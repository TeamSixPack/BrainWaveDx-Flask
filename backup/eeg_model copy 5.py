# -*- coding: utf-8 -*-
"""
eeg_model.py
- Hugging Face 가중치 자동 다운로드(레포 규칙: ardor924/EEGNetV4-{ch}ch-{device}-Ver{ver})
- MuseLab CSV 로더 통합(채널 재배열 → 1–40 Hz 필터 → 256→250 Hz 리샘플링)
- Part10 스타일 추론(5s/2.5s, best 2분 윈도우, per-record z-score, 품질가중)
- 캘리브레이션/바이어스(온도, 프라이어, 결정바이어스) 적용
- .csv / .set 모두 지원
"""
from __future__ import annotations
import os, re, json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mne
from huggingface_hub import snapshot_download

# ========================= 기본 설정 =========================
CLASS_NAMES = ['CN', 'AD', 'FTD']

CHANNEL_GROUPS: Dict[str, List[str]] = {
    'muse': ['T5','T6','F7','F8'],
    'hybrid_black': ['Fz','C3','Cz','C4','Pz','T5','T6','O1'],
    'union10': ['T5','T6','F7','F8','Fz','C3','Cz','C4','Pz','O1'],
    'total19': ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2'],
}

LOW_FREQ, HIGH_FREQ = 1.0, 40.0
TARGET_SRATE = 250
SEG_SECONDS, EVAL_HOP_SEC = 5.0, 2.5  # 50% overlap
WINDOW_NEED_SECONDS = 120
BATCH_SIZE = int(os.getenv("EEG_BATCH_SIZE", "64"))

# ========================= 모델 정의(Compat) =========================
class EEGNetV4Compat(nn.Module):
    """
    체크포인트 키 네이밍(firstconv/depthwise/separable/classifier)과 호환.
    k1(첫 conv 커널), k2(separable depthwise 커널), F1/D/F2, pool, dropout을 주입형으로 설정.
    최종 GAP→Linear(F2→n_classes) 구조로 classifier.in_features=F2 고정.
    """
    def __init__(self, n_classes: int, Chans: int,
                 k1: int, k2: int, F1: int, D: int, F2: int,
                 pool1: int = 4, pool2: int = 8, dropout: float = 0.3):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False),
            nn.BatchNorm2d(F1)
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D)
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, k2), padding=(0, k2 // 2), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2)
        )
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(F2, n_classes)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwise(x); x = self.elu(x); x = self.pool1(x); x = self.drop(x)
        x = self.separable(x); x = self.elu(x); x = self.pool2(x); x = self.drop(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x

# ========================= 가중치 로드 유틸 =========================
def _strip_prefix(sd: dict, prefixes=("module.", "model.")) -> dict:
    out = {}
    for k, v in sd.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
        out[kk] = v
    return out

def _load_state_dict_generic(weights_path: str, map_location: str):
    ext = os.path.splitext(weights_path)[-1].lower()
    if ext == ".safetensors":
        from safetensors.torch import load_file
        sd = dict(load_file(weights_path))
    else:
        obj = torch.load(weights_path, map_location=map_location)
        if isinstance(obj, dict):
            for k in ["state_dict","model_state_dict","weights","params","model","net"]:
                if k in obj and isinstance(obj[k], dict):
                    sd = obj[k]; break
            else:
                if all(isinstance(v, torch.Tensor) for v in obj.values()):
                    sd = obj
                elif isinstance(obj.get("model", None), nn.Module):
                    sd = obj["model"].state_dict()
                else:
                    raise RuntimeError("state_dict를 찾지 못했습니다.")
        elif isinstance(obj, nn.Module):
            sd = obj.state_dict()
        else:
            raise RuntimeError("지원되지 않는 가중치 포맷")
    return _strip_prefix(sd)

def _looks_compat(sd: dict) -> bool:
    return any(k.startswith("firstconv.0.weight") for k in sd.keys())

def _infer_hparams_from_sd(sd: dict, chans: int):
    F1, D, F2, k1, k2, p1, p2 = 32, 2, 64, 250, 32, 4, 8
    try:
        w = sd["firstconv.0.weight"]; F1 = int(w.shape[0]); k1 = int(w.shape[-1])
        w = sd["depthwise.0.weight"]; D  = int(w.shape[0] // F1)
        if "separable.0.weight" in sd: k2 = int(sd["separable.0.weight"].shape[-1])
        if "separable.1.weight" in sd: F2 = int(sd["separable.1.weight"].shape[0])
        if "classifier.weight" in sd:  F2 = int(sd["classifier.weight"].shape[1])
    except:  # pragma: no cover
        pass
    return F1, D, F2, k1, k2, p1, p2

def _hf_download(repo_id: str, token: Optional[str]):
    allow = ["*.pt","*.pth","*.bin","*.safetensors","config.json","calibration.json"]
    local_dir = snapshot_download(repo_id=repo_id, allow_patterns=allow, token=token)
    weights = []
    for root, _, files in os.walk(local_dir):
        for fn in files:
            if fn.lower().endswith((".pt",".pth",".bin",".safetensors")):
                weights.append(os.path.join(root, fn))
    if not weights:
        raise FileNotFoundError(f"[HF] No weights in {repo_id}")
    weights.sort()
    cfg = {}
    for name in ("config.json","calibration.json"):
        p = os.path.join(local_dir, name)
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    cfg.update(json.load(f))
            except Exception:
                pass
    return weights[0], cfg

# ========================= CSV 로더 =========================
# CSV 채널 정의: eeg_1..4 = [TP9, AF7, AF8, TP10]
# 학습 순서: ['T5','T6','F7','F8'] = [TP9, TP10, AF7, AF8]
_MUSE_CSV_ORDER_DEFAULT = ("TP9","AF7","AF8","TP10")
_MUSE_TRAIN_ORDER = ("T5","T6","F7","F8")
_MUSE_MAP_DEFAULT = {"TP9":"T5","TP10":"T6","AF7":"F7","AF8":"F8"}

def _parse_csv_order_env(env_val: Optional[str]) -> Tuple[str,str,str,str]:
    """
    EEG_CSV_ORDER 환경변수 파싱: 예) "TP9,AF7,AF8,TP10"
    """
    if not env_val:
        return _MUSE_CSV_ORDER_DEFAULT
    items = [s.strip().upper() for s in env_val.split(",") if s.strip()]
    if len(items) != 4 or set(items) != {"TP9","AF7","AF8","TP10"}:
        return _MUSE_CSV_ORDER_DEFAULT
    return tuple(items)  # type: ignore

def _load_muselab_csv(file_path: str,
                      csv_order: Optional[Tuple[str,str,str,str]] = None) -> Tuple[np.ndarray, float]:
    df = pd.read_csv(file_path)
    # 필수 컬럼 점검
    need_cols = ['eeg_1','eeg_2','eeg_3','eeg_4','timestamps']
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"CSV column missing: {c}")

    sub = df[need_cols].dropna()
    ts = sub["timestamps"].to_numpy(dtype=np.float64)
    # 타임스탬프 정렬/중복 제거
    if np.any(np.diff(ts) <= 0):
        sub = sub.sort_values("timestamps")
        ts = sub["timestamps"].to_numpy(dtype=np.float64)

    dt = np.diff(ts)
    dt_med = np.median(dt[dt > 0])
    sfreq_est = float(1.0 / dt_med)

    X_raw = sub[['eeg_1','eeg_2','eeg_3','eeg_4']].to_numpy(dtype=np.float32).T  # (4, T)

    # 채널 재배열: 입력 CSV의 물리 채널 순서 → 학습 채널 순서
    # csv_order가 없으면: (TP9,AF7,AF8,TP10) 으로 간주
    order = csv_order or _parse_csv_order_env(os.getenv("EEG_CSV_ORDER"))
    idx_by_name = {"TP9":0, "AF7":1, "AF8":2, "TP10":3}
    # 학습 순서 T5,T6,F7,F8 ← [TP9,TP10,AF7,AF8]
    X_ord = np.stack([
        X_raw[idx_by_name["TP9"], :],   # T5
        X_raw[idx_by_name["TP10"], :],  # T6
        X_raw[idx_by_name["AF7"], :],   # F7
        X_raw[idx_by_name["AF8"], :],   # F8
    ], axis=0)

    # RawArray → 필터/리샘플
    info = mne.create_info(list(_MUSE_TRAIN_ORDER), sfreq=sfreq_est, ch_types='eeg')
    raw = mne.io.RawArray(X_ord, info, verbose='ERROR')
    raw.filter(LOW_FREQ, HIGH_FREQ, fir_design='firwin', verbose='ERROR')
    if abs(sfreq_est - TARGET_SRATE) > 1e-3:
        raw.resample(TARGET_SRATE, verbose='ERROR')
    return raw.get_data(), TARGET_SRATE  # (4, T_250), 250

# ========================= 세그먼트/보조 =========================
def _segment_overlap(data: np.ndarray, win_sec: float, hop_sec: float, sfreq: float) -> np.ndarray:
    C, T = data.shape
    win = int(round(win_sec * sfreq))
    hop = int(round(hop_sec * sfreq))
    if T < win:
        return np.empty((0, C, win), dtype=np.float32)
    idxs = list(range(0, T - win + 1, hop))
    return np.stack([data[:, i:i+win] for i in idxs], axis=0).astype(np.float32)

def _per_record_zscore(segs: np.ndarray) -> np.ndarray:
    mean = segs.mean(axis=(0,2), keepdims=True)
    std  = segs.std(axis=(0,2), keepdims=True) + 1e-7
    return (segs - mean) / std

def _quality_weights(segs: np.ndarray) -> np.ndarray:
    std = segs.std(axis=(1,2))
    med = np.median(std) + 1e-8
    return np.where(std < 0.2 * med, 1e-3, 1.0).astype(np.float32)

def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)

# ========================= 추론 엔진 =========================
class EEGInferenceEngine:
    """
    device_type: 'muse' | 'hybrid_black' | 'union10' | 'total19'
    version    : HF 레포 Ver (문자열)
    csv_order  : Muse CSV의 물리 채널 이름 순서 (TP9,AF7,AF8,TP10), 기본값은 환경변수 EEG_CSV_ORDER 또는 표준 순서
    """
    def __init__(self, device_type: str = 'muse',
                 version: Optional[str] = None,
                 torch_device: Optional[str] = None,
                 hf_token: Optional[str] = None,
                 csv_order: Optional[Tuple[str,str,str,str]] = None):
        self.device_type = device_type.lower().strip()
        if self.device_type not in CHANNEL_GROUPS:
            raise ValueError(f"Unknown device_type '{self.device_type}'. Choose one of {list(CHANNEL_GROUPS.keys())}")

        self.channels = CHANNEL_GROUPS[self.device_type]
        self.samples_per_seg = int(TARGET_SRATE * SEG_SECONDS)
        self.hop_samples = int(TARGET_SRATE * EVAL_HOP_SEC)
        self.torch_device = torch_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.version = str(version or os.getenv("EEG_WEIGHTS_VER", "14")).strip()
        self.hf_token = hf_token or os.getenv("HF_TOKEN", None)
        self.csv_order = csv_order  # only used for .csv inputs

        # HF 가중치 로드
        repo_id = f"ardor924/EEGNetV4-{len(self.channels)}ch-{self.device_type}-Ver{self.version}"
        weights_path, cfg = _hf_download(repo_id, token=self.hf_token)
        sd = _load_state_dict_generic(weights_path, map_location=self.torch_device)
        if not _looks_compat(sd):
            raise RuntimeError("Unsupported checkpoint (expected 'firstconv/depthwise/separable/...')")

        F1, D, F2, k1, k2, p1, p2 = _infer_hparams_from_sd(sd, chans=len(self.channels))
        # config.json 값으로 보정(존재 시)
        k1 = int(cfg.get("kernel_length", k1))
        k2 = int(cfg.get("sep_length", k2))
        F1 = int(cfg.get("F1", F1))
        D  = int(cfg.get("D", D))
        dropout = float(cfg.get("dropout_rate", 0.3))
        pool1 = int(cfg.get("pool1", 4)); pool2 = int(cfg.get("pool2", 8))

        self.model = EEGNetV4Compat(
            n_classes=len(CLASS_NAMES), Chans=len(self.channels),
            k1=k1, k2=k2, F1=F1, D=D, F2=F2,
            pool1=pool1, pool2=pool2, dropout=dropout
        ).to(self.torch_device)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()

        # ----- 캘리브레이션/바이어스 -----
        self.temperature     = float(os.getenv("EEG_TEMP",              cfg.get("temperature", 1.0)))
        self.prior_strength  = float(os.getenv("EEG_PRIOR_STRENGTH",    cfg.get("prior_strength", 0.0)))
        # class prior
        prior_cfg = cfg.get("class_prior", None)
        env_prior = os.getenv("EEG_CLASS_PRIOR", None)  # "CN:0.34,AD:0.33,FTD:0.33"
        if env_prior:
            try:
                d = {}
                for kv in env_prior.split(","):
                    k, v = kv.split(":"); d[k.strip().upper()] = float(v)
                prior_cfg = d
            except Exception:
                pass
        self.class_prior = None
        if prior_cfg:
            self.class_prior = np.array([float(prior_cfg.get(c, 1/len(CLASS_NAMES))) for c in CLASS_NAMES], dtype=np.float32)
            self.class_prior = np.clip(self.class_prior, 1e-6, 1.0)
            self.class_prior /= self.class_prior.sum()

        # decision bias
        bias_cfg = cfg.get("decision_bias", None)
        env_bias = os.getenv("EEG_DECISION_BIAS", None)  # "0,0.05,-0.05"
        if env_bias:
            try: bias_cfg = [float(x) for x in env_bias.split(",")]
            except Exception: pass
        self.decision_bias = np.array(bias_cfg, dtype=np.float32) if bias_cfg is not None else np.zeros(len(CLASS_NAMES), dtype=np.float32)

    # ----- 내부 보조 -----
    def _apply_calib(self, logits: np.ndarray) -> np.ndarray:
        z = logits / max(1e-3, self.temperature)
        if self.class_prior is not None and self.prior_strength > 0:
            z = z + self.prior_strength * np.log(self.class_prior[None, :])
        if self.decision_bias is not None:
            z = z - self.decision_bias[None, :]
        return z

    def _choose_best_window(self, probs_all: np.ndarray, need: int) -> Tuple[int, int]:
        top1 = probs_all.max(axis=1)
        cs = np.concatenate([[0.0], np.cumsum(top1)])
        best, best_sum = 0, -1.0
        for s in range(0, len(top1) - need + 1):
            sm = cs[s + need] - cs[s]
            if sm > best_sum:
                best, best_sum = s, sm
        return best, need

    def _read_any(self, file_path: str) -> Tuple[np.ndarray, float]:
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".csv":
            data, srate = _load_muselab_csv(file_path, csv_order=self.csv_order)
            return data, srate  # (4,T), 250
        elif ext == ".set":
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose='ERROR')
            miss = [ch for ch in self.channels if ch not in raw.ch_names]
            if miss:
                raise ValueError(f"Channels missing in file: {miss}\nPresent: {raw.ch_names}\nExpected: {self.channels}")
            raw.pick_channels(self.channels)
            raw.filter(LOW_FREQ, HIGH_FREQ, fir_design='firwin', verbose='ERROR')
            raw.resample(TARGET_SRATE, verbose='ERROR')
            return raw.get_data(), TARGET_SRATE
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ----- 공개 API -----
    @torch.no_grad()
    def infer(self, file_path: str,
              subject_id: Optional[str] = None,
              true_label: Optional[str] = None,
              enforce_two_minutes: bool = True) -> Dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"EEG file not found: {file_path}")

        data, srate = self._read_any(file_path)  # (C,T), 250
        segs = _segment_overlap(data, SEG_SECONDS, EVAL_HOP_SEC, srate)
        need = int((WINDOW_NEED_SECONDS - SEG_SECONDS) / EVAL_HOP_SEC) + 1  # = 47
        N = segs.shape[0]
        if N == 0:
            raise ValueError("No segments could be formed from the recording.")
        if enforce_two_minutes and N < need:
            raise ValueError(f"Too short for 2-minute window: need {need}, got {N}")

        segs_z = _per_record_zscore(segs)
        x = torch.from_numpy(segs_z)[:, None, :, :].to(self.torch_device)
        # batched logits
        outs = []
        for i in range(0, x.size(0), BATCH_SIZE):
            outs.append(self.model(x[i:i+BATCH_SIZE]).detach().cpu().numpy().astype(np.float32))
        logits_all = np.concatenate(outs, axis=0)
        probs_all  = _softmax_np(self._apply_calib(logits_all))

        if N < need:
            s_best, use = 0, N
        else:
            s_best, use = self._choose_best_window(probs_all, need)

        # 세그먼트 지표
        block_logits = logits_all[s_best:s_best+use]
        block_probs  = probs_all[s_best:s_best+use]
        y_pred = block_probs.argmax(axis=1)
        counts = {CLASS_NAMES[i]: int((y_pred == i).sum()) for i in range(len(CLASS_NAMES))}
        maj_idx = int(np.bincount(y_pred, minlength=len(CLASS_NAMES)).argmax())
        maj_lbl = CLASS_NAMES[maj_idx]

        # subject-level: 품질가중 + 로짓 평균 → softmax
        w = _quality_weights(segs[s_best:s_best+use])
        wsum = float(w.sum()) + 1e-8
        subj_logit = (self._apply_calib(block_logits) * w[:, None]).sum(axis=0) / wsum
        subj_prob  = _softmax_np(subj_logit[None, :])[0]

        # (옵션) 세그 정확도
        seg_acc = None
        if true_label:
            tl = str(true_label).strip().upper()
            if tl in ("C","A","F"): tl = {"C":"CN","A":"AD","F":"FTD"}[tl]
            if tl in CLASS_NAMES:
                tl_idx = CLASS_NAMES.index(tl)
                seg_acc = float((y_pred == tl_idx).mean())

        # subject_id 추정
        sid = subject_id
        if not sid:
            m = re.search(r"(sub-\d+)", file_path, flags=re.IGNORECASE)
            sid = m.group(1) if m else None

        return {
            "channels_used": self.channels,
            "file_path": file_path,
            "n_segments": int(use),
            "prob_mean": {CLASS_NAMES[i]: float(subj_prob[i]) for i in range(len(CLASS_NAMES))},
            "segment_accuracy": seg_acc,
            "segment_counts": counts,
            "segment_majority_index": maj_idx,
            "segment_majority_label": maj_lbl,
            "subject_id": sid,
            "window": {"start": int(s_best * EVAL_HOP_SEC), "need": int(WINDOW_NEED_SECONDS)}
        }
