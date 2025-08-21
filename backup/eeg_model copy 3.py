# eeg_model.py
# ---------------------------------------------------------
# EEGLAB(.set) 로드 & 2분 구간 추론
# - 누락 채널 0-패딩, CAR, bandpass, 조건부 notch
# - 품질(bad 세그) 가중 로짓 평균 → subject-level 확률/판정
# - 세그 평균 확률(prob_mean)도 함께 제공
# ---------------------------------------------------------
import os
import re
import json
from typing import Union, Optional, Dict

import numpy as np
import torch
import torch.nn as nn

# huggingface_hub
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download

# MNE
import mne

# ========== 모델 정의 ==========
class EEGNetV4(nn.Module):
    def __init__(self, n_chans, n_classes, F1=32, D=2, kernel_length=250, pool1=4, pool2=8, dropout_rate=0.3):
        super().__init__()
        F2 = F1 * D
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F2, (n_chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout_rate)
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 32), groups=F2, padding=(0, 16), bias=False),
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout_rate)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(F2, n_classes)

    def forward(self, x):  # x: (N,1,C,T)
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# =============== 라벨 맵 정규화 ===============
def _normalize_label_maps(cfg: dict) -> dict:
    default_idx2lab = {0: "CN", 1: "AD", 2: "FTD"}

    idx2 = cfg.get("idx_to_label") or default_idx2lab.copy()
    lab2 = cfg.get("label_to_idx")

    try:
        if isinstance(next(iter(idx2.keys())), str):
            idx2 = {int(k): v for k, v in idx2.items()}
    except StopIteration:
        idx2 = default_idx2lab.copy()
    except Exception:
        idx2 = default_idx2lab.copy()

    n = int(cfg.get("n_classes", len(idx2) if len(idx2) > 0 else 3))
    want_keys = set(range(n))
    if set(idx2.keys()) != want_keys:
        if isinstance(lab2, dict):
            try:
                any_v = next(iter(lab2.values()))
                if isinstance(any_v, str):
                    lab2 = {k: int(v) for k, v in lab2.items()}
            except StopIteration:
                lab2 = None
        if isinstance(lab2, dict) and set(lab2.values()) == want_keys:
            inv = {v: k for k, v in lab2.items()}
            idx2 = {i: inv.get(i, default_idx2lab.get(i, f"class_{i}")) for i in range(n)}
        else:
            idx2 = {i: default_idx2lab.get(i, f"class_{i}") for i in range(n)}

    lab2 = {v: k for k, v in idx2.items()}
    cfg["idx_to_label"] = idx2
    cfg["label_to_idx"] = lab2
    cfg["n_classes"] = n
    return cfg

# =============== HF 모델 로드 ===============
def load_hf_model(repo_id: str, device: Optional[str] = None, token: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json", token=token)
    wt_path  = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", token=token)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg = _normalize_label_maps(cfg)

    n_chans   = int(cfg.get("n_chans", len(cfg.get("pick_channels", [])) or 8))
    n_classes = int(cfg.get("n_classes", 3))
    model = EEGNetV4(
        n_chans=n_chans,
        n_classes=n_classes,
        F1=int(cfg.get("F1", 32)),
        D=int(cfg.get("D", 2)),
        kernel_length=int(cfg.get("kernel_length", 250)),
        pool1=int(cfg.get("pool1", 4)),
        pool2=int(cfg.get("pool2", 8)),
        dropout_rate=float(cfg.get("dropout_rate", 0.3)),
    ).to(device)

    state = torch.load(wt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg

# ==================== 전처리 유틸(MNE) ====================
def _apply_car(raw: mne.io.BaseRaw):
    picks = mne.pick_types(raw.info, eeg=True, misc=False)
    if len(picks) >= 2:
        raw.set_eeg_reference('average', projection=False, verbose=False)
    return raw

def _detect_mains_hz(raw: mne.io.BaseRaw, ratio_thresh: float = 3.0) -> int:
    try:
        psd = mne.time_frequency.psd_welch(raw, fmin=40, fmax=70, n_fft=2048, n_overlap=1024, verbose=False)
        if isinstance(psd, tuple):
            psd_vals, freqs = psd
        else:
            psd_vals = psd.get_data(); freqs = psd.freqs
        psd_median = np.median(psd_vals, axis=0)

        def band_med(f0, w=2.0):
            m = (freqs >= (f0 - w)) & (freqs <= (f0 + w))
            return np.median(psd_median[m]) if m.any() else 0.0

        p50 = band_med(50.0); base50 = np.median(psd_median[(freqs>=40)&(freqs<=46)]) if ((freqs>=40)&(freqs<=46)).any() else 1.0
        p60 = band_med(60.0); base60 = np.median(psd_median[(freqs>=64)&(freqs<=70)]) if ((freqs>=64)&(freqs<=70)).any() else 1.0
        r50, r60 = p50/(base50+1e-9), p60/(base60+1e-9)
        if r50 >= ratio_thresh and r50 >= r60: return 50
        if r60 >= ratio_thresh and r60 >  r50: return 60
    except Exception:
        pass
    return 0

def _read_filter_resample(filepath: str, cfg: dict) -> mne.io.BaseRaw:
    low = float(cfg.get("low_freq", 1.0))
    high = float(cfg.get("high_freq", 40.0))
    target_srate = int(cfg.get("target_srate", 250))
    enable_notch = bool(cfg.get("enable_conditional_notch", True))
    notch_harmonics = bool(cfg.get("notch_harmonics", True))

    raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)

    if enable_notch:
        mains = _detect_mains_hz(raw)
        if mains in (50, 60):
            freqs = [mains, mains*2] if notch_harmonics else [mains]
            try: raw.notch_filter(freqs=freqs, verbose=False)
            except Exception: pass

    raw.filter(l_freq=low, h_freq=high, verbose=False)
    raw.resample(sfreq=target_srate, npad="auto", verbose=False)
    raw = _apply_car(raw)
    return raw

# ---------- 채널 매칭(aliased name 허용) ----------
def _get_data_fixed_channels(raw: mne.io.BaseRaw, pick_channels: list) -> np.ndarray:
    """
    항상 pick_channels 순서 (C,T) 반환, 누락 채널 0-패딩.
    T5/T6↔T7/T8 등 이름 차이를 alias로 보정 시도.
    """
    ALIAS = {
        "T5": ["T5", "T7", "TP7", "P7"],
        "T6": ["T6", "T8", "TP8", "P8"],
        "F7": ["F7", "FT7"],
        "F8": ["F8", "FT8"],
        "O1": ["O1", "PO7"],
    }

    T = raw.n_times
    data_list = []
    for target in pick_channels:
        candidates = ALIAS.get(target, [target])
        picked = None
        for name in candidates:
            if name in raw.ch_names:
                picked = name
                break
        if picked is None:
            v = np.zeros(T, dtype=np.float32)
        else:
            v = raw.get_data(picks=[picked]).astype(np.float32)[0]
        data_list.append(v)

    return np.stack(data_list, axis=0)

def _segment_array(data: np.ndarray, srate: int, win_sec: float, hop_sec: float) -> np.ndarray:
    C, T = data.shape
    win = int(win_sec * srate)
    hop = int(hop_sec * srate)
    if T < win: return np.empty((0, C, win), dtype=np.float32)
    segs = [data[:, s:s+win] for s in range(0, T - win + 1, hop)]
    return np.stack(segs, axis=0).astype(np.float32) if segs else np.empty((0, C, win), dtype=np.float32)

# ---------- 품질 메트릭 & 최적 시작점 ----------
def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x); mad = np.median(np.abs(x - med)) + 1e-6
    return (x - med) / (1.4826 * mad)

def _segment_quality_metrics(segs: np.ndarray, srate: int, lf_env_win_sec: float = 0.5,
                             thr_ptp_z: float = 5.0, thr_lf_z: float = 3.0) -> Dict[str, np.ndarray]:
    if segs.shape[0] == 0:
        return {"bad": np.array([], dtype=bool)}
    ptp = segs.max(axis=2) - segs.min(axis=2)   # (N,C)
    ptp_max = ptp.max(axis=1)                   # (N,)
    ptp_z = _robust_z(ptp_max)

    k = max(1, int(lf_env_win_sec * srate))
    if k > 1:
        absx = np.abs(segs)
        csum = np.cumsum(absx, axis=2)
        env = (csum[:, :, k:] - csum[:, :, :-k]) / k  # (N,C,T-k)
        env_max = env.max(axis=2)                     # (N,C)
    else:
        env_max = np.abs(segs).mean(axis=2)
    lf_med = np.median(env_max, axis=1)              # (N,)
    lf_z = _robust_z(lf_med)

    bad = (ptp_z > thr_ptp_z) | (lf_z > thr_lf_z)
    return {"bad": bad}

def _two_min_need(win_sec: float, hop_sec: float) -> int:
    return int((120.0 - win_sec) / hop_sec) + 1

def _best_window_start(segs: np.ndarray, need: int, srate: int) -> int:
    N = segs.shape[0]
    if N < need: return -1
    qbad = _segment_quality_metrics(segs, srate=srate)["bad"]
    best_s, best_score = -1, -1.0
    for s in range(0, N - need + 1):
        e = s + need
        score = 1.0 - float(qbad[s:e].mean())
        if score > best_score:
            best_score, best_s = score, s
    return best_s

# ========================= 소프트맥스 & 유틸 =========================
def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def extract_subject_id(file_path: str) -> str:
    m = re.search(r"(sub-\d+)", file_path.replace("\\", "/"))
    return m.group(1) if m else "unknown"

def _canon_true_label(true_label, name_to_idx: Dict[str, int]) -> Optional[int]:
    """
    true_label: "CN"/"AD"/"FTD" 또는 "C"/"A"/"F" 또는 0/1/2
    name_to_idx: {"CN":0,"AD":1,"FTD":2} 같은 매핑
    """
    if true_label is None:
        return None
    if isinstance(true_label, (int, np.integer)):
        gi = int(true_label)
        return gi if gi in name_to_idx.values() else None
    tl = str(true_label).strip().upper()
    if tl in ("C", "A", "F"):
        tl = {"C":"CN","A":"AD","F":"FTD"}[tl]
    return name_to_idx.get(tl, None)

# ========================= 추론 루틴 =========================
@torch.no_grad()
def predict_from_eeglab_file(file_path: str,
                             model: nn.Module,
                             cfg: dict,
                             device: Optional[str] = None,
                             true_label: Union[str, int, None] = None,
                             enforce_two_minutes: bool = True):
    """
    EEGLAB .set 파일을 읽어 '정확히 2분' 윈도우에서 세그먼트 예측.
    - prob_mean: 세그 확률의 단순 평균
    - subject_probs: (품질가중) 로짓 평균 → softmax  → 최종 판정
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 채널/샘플링/윈도우 설정
    pick_channels = cfg.get("pick_channels", ["Fz","C3","Cz","C4","Pz","T5","T6","O1"])
    srate   = int(cfg.get("target_srate", 250))
    win_sec = float(cfg.get("segment_sec", 5.0))
    hop_sec = float(cfg.get("eval_hop_sec", 2.5))

    # 라벨 맵
    idx_to_label: Dict[int, str] = cfg.get("idx_to_label", {0: "CN", 1: "AD", 2: "FTD"})
    def _lab(i: int) -> str:
        return idx_to_label.get(i) or idx_to_label.get(str(i)) or f"class_{i}"
    name_to_idx = {(_lab(i)).upper(): i for i in range(int(cfg.get("n_classes", 3)))}

    # 로드 + 전처리
    raw  = _read_filter_resample(file_path, cfg)
    data = _get_data_fixed_channels(raw, pick_channels)               # (C,T)
    segs = _segment_array(data, srate=srate, win_sec=win_sec, hop_sec=hop_sec)  # (N,C,win)

    need = _two_min_need(win_sec, hop_sec)
    if segs.shape[0] < need:
        if enforce_two_minutes:
            raise ValueError(f"파일 길이가 부족합니다: 2분 커버에 필요한 세그먼트 {need}개, 실제 {segs.shape[0]}개")
        else:
            need = segs.shape[0]

    s_best = _best_window_start(segs, need=need, srate=srate)
    if s_best < 0:
        s_best = 0
    block = segs[s_best:s_best+need]  # (need, C, win)

    # 배치 추론 → per-seg 로짓/확률
    X = torch.from_numpy(block)[:, None, :, :].to(device)
    logits = []
    bs = int(cfg.get("inference_batch_size", 64))
    for p in range(0, X.size(0), bs):
        lb = model(X[p:p+bs])
        logits.append(lb.detach().cpu().numpy().astype(np.float32))
    logits = np.concatenate(logits, axis=0)   # (need, n_classes)
    probs  = _softmax_np(logits)              # (need, n_classes)
    pred   = probs.argmax(axis=1)             # (need,)

    # 세그먼트 카운트/확률 평균
    n_classes = int(cfg.get("n_classes", 3))
    counts = np.bincount(pred, minlength=n_classes)
    maj_idx = int(np.argmax(counts))
    maj_label = _lab(maj_idx)

    prob_mean = probs.mean(axis=0)  # shape=(n_classes,)
    prob_map = {_lab(i): float(prob_mean[i]) for i in range(prob_mean.shape[0])}
    counts_map = {_lab(i): int(counts[i]) for i in range(counts.shape[0]) if counts[i] > 0}

    # ---------- ✅ subject-level 확률/판정 ----------
    qm = _segment_quality_metrics(block, srate=srate)
    w  = np.where(qm["bad"], 1e-3, 1.0).astype(np.float32)           # (need,)
    w_sum = float(w.sum()) + 1e-8
    subj_logits = (logits * w[:, None]).sum(axis=0) / w_sum          # (n_classes,)
    subj_probs  = np.exp(subj_logits - np.max(subj_logits))
    subj_probs  = subj_probs / (subj_probs.sum() + 1e-12)
    subject_probs_map = {_lab(i): float(subj_probs[i]) for i in range(n_classes)}
    subject_pred_idx   = int(np.argmax(subj_probs))
    subject_pred_label = _lab(subject_pred_idx)

    # 실제 라벨(옵션) → subject-level accuracy(0/1)
    gt_idx = _canon_true_label(true_label, name_to_idx)
    subject_acc = None
    if gt_idx is not None and 0 <= gt_idx < n_classes:
        subject_acc = float(1.0 if subject_pred_idx == gt_idx else 0.0)

    # (옵션) Segment Accuracy (세그 기준) — true_label 제공 시
    seg_acc = None
    if gt_idx is not None and 0 <= gt_idx < n_classes:
        seg_acc = float(np.mean(pred == gt_idx))

    return {
        "file_path": os.path.abspath(file_path),
        "subject_id": extract_subject_id(file_path),

        # 세그먼트 기준
        "n_segments": int(block.shape[0]),
        "segment_counts": counts_map,
        "segment_majority_index": maj_idx,
        "segment_majority_label": maj_label,
        "segment_accuracy": seg_acc,

        # 세그 평균 확률
        "prob_mean": prob_map,

        # ✅ subject-level 확률/판정/정확도 (신규)
        "subject_probs": subject_probs_map,
        "subject_pred_index": subject_pred_idx,
        "subject_pred_label": subject_pred_label,
        "subject_accuracy": subject_acc,

        # 부가정보
        "window": {"start": int(s_best), "need": int(need)},
        "channels_used": list(pick_channels),
    }
