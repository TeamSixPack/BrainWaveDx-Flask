# -*- coding: utf-8 -*-
"""
eeg_model3class.py
- 3진분류(CN/AD/FTD) 전용 추론 엔진
- 2클 가중치를 주입하면 명확한 에러 메시지로 안내
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
CLASS_NAMES = ['CN', 'AD', 'FTD']   # 3-class 고정

CHANNEL_GROUPS: Dict[str, List[str]] = {
    'muse': ['T5','T6','F7','F8'],
    'hybrid_black': ['Fz','C3','Cz','C4','Pz','T5','T6','O1'],
    'union10': ['T5','T6','F7','F8','Fz','C3','Cz','C4','Pz','O1'],
    'total19': ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2'],
}

LOW_FREQ, HIGH_FREQ   = 1.0, 40.0
TARGET_SRATE          = 250
SEG_SECONDS           = 5.0
EVAL_HOP_SEC          = 2.5
WINDOW_NEED_SECONDS   = 120
BATCH_SIZE            = int(os.getenv("EEG_BATCH_SIZE", "64"))

# 기본값(ENV → 기본)
DEFAULT_DEVICE  = os.getenv("EEG_DEVICE_DEFAULT", "muse").strip().lower()
DEFAULT_VER     = os.getenv("EEG_WEIGHTS_VER", "29").strip()
DEFAULT_COMMENT = os.getenv("EEG_WEIGHTS_COMMENT", "").strip()

# ========================= 모델 정의 =========================
class EEGNetV4Compat(nn.Module):
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

# ========================= 유틸 =========================
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
    except:
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

def _build_repo_id(ch_len: int, device: str, ver: str, comment: Optional[str]) -> str:
    base = f"ardor924/EEGNetV4-{ch_len}ch-{device}-{ver}"
    if comment is not None and str(comment).strip() != "":
        return f"{base}-{comment.strip()}"
    return base

# ----- CSV/SET 로더 -----
_MUSE_CSV_ORDER_DEFAULT = ("TP9","AF7","AF8","TP10")
_MUSE_TRAIN_ORDER       = ("T5","T6","F7","F8")

def _parse_csv_order_env(env_val: Optional[str]) -> Tuple[str,str,str,str]:
    if not env_val:
        return _MUSE_CSV_ORDER_DEFAULT
    items = [s.strip().upper() for s in env_val.split(",") if s.strip()]
    if len(items) != 4 or not set(items).issubset({"TP9","AF7","AF8","TP10"}):
        return _MUSE_CSV_ORDER_DEFAULT
    return tuple(items)  # type: ignore

def _robust_median_dt(ts: np.ndarray) -> float:
    dt = np.diff(ts)
    dt = dt[dt > 0]
    if dt.size == 0:
        raise ValueError("Invalid timestamps: non-increasing or empty.")
    q1, q3 = np.quantile(dt, [0.25, 0.75])
    iqr = max(1e-9, q3 - q1)
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    dt_clipped = dt[(dt >= max(1e-4, low)) & (dt <= min(1.0, high))]
    if dt_clipped.size == 0:
        dt_clipped = dt
    return float(np.median(dt_clipped))

def _detect_mains_hz_raw(raw: mne.io.BaseRaw, ratio_thresh: float = 3.0) -> int:
    try:
        psd = mne.time_frequency.psd_welch(raw, fmin=45, fmax=65, n_fft=4096,
                                           n_overlap=1024, verbose="ERROR")
        if isinstance(psd, tuple):
            psd_vals, freqs = psd
        else:
            psd_vals = psd.get_data(); freqs = psd.freqs
        med = np.median(psd_vals, axis=0)
        def band_pow(f0, w=1.5):
            m = (freqs >= f0 - w) & (freqs <= f0 + w)
            return np.median(med[m]) if m.any() else 0.0
        p50 = band_pow(50.0); p60 = band_pow(60.0)
        base = np.median(med[(freqs >= 46) & (freqs <= 64)])
        r50 = p50 / (base + 1e-9); r60 = p60 / (base + 1e-9)
        if r50 >= ratio_thresh and r50 >= r60: return 50
        if r60 >= ratio_thresh and r60 >  r50: return 60
    except Exception:
        pass
    return 0

def _maybe_notch(raw: mne.io.BaseRaw):
    env = os.getenv("EEG_MAINS", "").strip()
    mains = int(env) if env in ("50","60") else _detect_mains_hz_raw(raw, ratio_thresh=3.0)
    if mains in (50, 60):
        try:
            raw.notch_filter(freqs=[mains], verbose="ERROR")
        except Exception:
            pass

def _load_muselab_csv(file_path: str, csv_order: Optional[Tuple[str,str,str,str]] = None) -> Tuple[np.ndarray, float]:
    df = pd.read_csv(file_path)
    need_cols = ['eeg_1','eeg_2','eeg_3','eeg_4','timestamps']
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"CSV column missing: {c}")
    sub = df[need_cols].dropna()
    ts = sub["timestamps"].to_numpy(dtype=np.float64)
    if np.any(np.diff(ts) <= 0):
        sub = sub.sort_values("timestamps")
        ts = sub["timestamps"].to_numpy(dtype=np.float64)
    dt_med = _robust_median_dt(ts)
    sfreq_est = float(1.0 / max(dt_med, 1e-6))
    X_raw = sub[['eeg_1','eeg_2','eeg_3','eeg_4']].to_numpy(dtype=np.float32).T

    order = csv_order or _parse_csv_order_env(os.getenv("EEG_CSV_ORDER"))
    idx_by_name = {name: i for i, name in enumerate(order)}
    X_ord = np.stack([
        X_raw[idx_by_name["TP9"], :],   # T5
        X_raw[idx_by_name["TP10"], :],  # T6
        X_raw[idx_by_name["AF7"], :],   # F7
        X_raw[idx_by_name["AF8"], :],   # F8
    ], axis=0)

    info = mne.create_info(list(_MUSE_TRAIN_ORDER), sfreq=sfreq_est, ch_types='eeg')
    raw = mne.io.RawArray(X_ord, info, verbose='ERROR')
    _maybe_notch(raw)
    raw.filter(LOW_FREQ, HIGH_FREQ, fir_design='firwin', verbose='ERROR')
    if abs(sfreq_est - TARGET_SRATE) > 1e-3:
        raw.resample(TARGET_SRATE, verbose='ERROR')
    try:
        raw.set_eeg_reference('average', projection=False, verbose='ERROR')
    except Exception:
        pass
    return raw.get_data(), TARGET_SRATE

def _norm(name: str) -> str:
    import re as _re
    return _re.sub(r'[^A-Z0-9]', '', str(name).upper())

def _load_device_csv(file_path: str, channels: List[str]) -> Tuple[np.ndarray, float]:
    df = pd.read_csv(file_path)
    if 'timestamps' in df.columns:
        sub = df.dropna(subset=['timestamps']).copy()
        ts = sub['timestamps'].to_numpy(dtype=np.float64)
        if np.any(np.diff(ts) <= 0):
            sub = sub.sort_values('timestamps')
            ts = sub['timestamps'].to_numpy(dtype=np.float64)
        dt_med = _robust_median_dt(ts)
        sfreq_est = float(1.0 / max(dt_med, 1e-6))
    else:
        sub = df.copy()
        sfreq_est = float(os.getenv("EEG_CSV_SFREQ", TARGET_SRATE))

    norm2orig = { _norm(c): c for c in sub.columns }
    X_list, missing = [], []
    for ch in channels:
        key = _norm(ch)
        if key in norm2orig:
            X_list.append(sub[norm2orig[key]].to_numpy(dtype=np.float32))
        else:
            cand = None
            for k, orig in norm2orig.items():
                if k.endswith(key):
                    cand = orig; break
            if cand is not None:
                X_list.append(sub[cand].to_numpy(dtype=np.float32))
            else:
                missing.append(ch)
    if missing:
        raise ValueError(f"CSV missing channels: {missing} / expected={channels}")

    X_ord = np.stack(X_list, axis=0)
    info = mne.create_info(channels, sfreq=sfreq_est, ch_types='eeg')
    raw = mne.io.RawArray(X_ord, info, verbose='ERROR')
    _maybe_notch(raw)
    raw.filter(LOW_FREQ, HIGH_FREQ, fir_design='firwin', verbose='ERROR')
    if abs(sfreq_est - TARGET_SRATE) > 1e-3:
        raw.resample(TARGET_SRATE, verbose='ERROR')
    try:
        raw.set_eeg_reference('average', projection=False, verbose='ERROR')
    except Exception:
        pass
    return raw.get_data(), TARGET_SRATE

# ========================= 보조 함수 =========================
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

# ========================= 엔진 =========================
class EEGInferenceEngine3Class:
    def __init__(self,
                 device_type: Optional[str] = None,
                 version: Optional[str] = None,
                 comment: Optional[str] = None,
                 torch_device: Optional[str] = None,
                 hf_token: Optional[str] = None,
                 csv_order: Optional[Tuple[str,str,str,str]] = None):

        self.device_type = (device_type or DEFAULT_DEVICE).strip().lower()
        if self.device_type not in CHANNEL_GROUPS:
            raise ValueError(f"Unknown device_type '{self.device_type}'. Choose one of {list(CHANNEL_GROUPS.keys())}")

        self.channels = CHANNEL_GROUPS[self.device_type]
        self.torch_device = torch_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.version = str(version or DEFAULT_VER).strip()
        self.comment = comment if comment is not None else DEFAULT_COMMENT
        self.hf_token = hf_token or os.getenv("HF_TOKEN", None)
        self.csv_order = csv_order

        # HF 가중치
        ch_len = len(self.channels)
        repo = _build_repo_id(ch_len, self.device_type, self.version, self.comment)
        try:
            weights_path, cfg = _hf_download(repo, token=self.hf_token)
        except Exception as e_primary:
            legacy = f"ardor924/EEGNetV4-{ch_len}ch-{self.device_type}-Ver{self.version}"
            try:
                weights_path, cfg = _hf_download(legacy, token=self.hf_token)
            except Exception as e_legacy:
                raise FileNotFoundError(f"Failed to download weights\n  tried: {repo}\n  and  : {legacy}\n{e_primary} | {e_legacy}")

        sd = _load_state_dict_generic(weights_path, map_location=self.torch_device)
        if not _looks_compat(sd):
            raise RuntimeError("Unsupported checkpoint format")

        # 출력 차원 검사(3클 고정)
        if "classifier.weight" in sd:
            n_out = int(sd["classifier.weight"].shape[0])
        elif "classifier.bias" in sd:
            n_out = int(sd["classifier.bias"].shape[0])
        else:
            n_out = 3
        if n_out != 3:
            raise RuntimeError(
                f"This endpoint expects 3-class weights, but checkpoint has {n_out} outputs. "
                f"→ Use /infer2class or load 3-class weights (comment/ver)."
            )

        # 하이퍼 파라미터
        F1, D, F2, k1, k2, p1, p2 = _infer_hparams_from_sd(sd, chans=ch_len)
        k1 = int(cfg.get("kernel_length", k1))
        k2 = int(cfg.get("sep_length", k2))
        F1 = int(cfg.get("F1", F1))
        D  = int(cfg.get("D", D))
        dropout = float(cfg.get("dropout_rate", 0.3))
        pool1 = int(cfg.get("pool1", 4)); pool2 = int(cfg.get("pool2", 8))

        self.model = EEGNetV4Compat(
            n_classes=3, Chans=ch_len, k1=k1, k2=k2, F1=F1, D=D, F2=F2,
            pool1=pool1, pool2=pool2, dropout=dropout
        ).to(self.torch_device)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()

        # 캘리브레이션/바이어스
        self.temperature     = float(os.getenv("EEG_TEMP",           cfg.get("temperature", 1.0)))
        self.prior_strength  = float(os.getenv("EEG_PRIOR_STRENGTH", cfg.get("prior_strength", 0.0)))
        prior_cfg = cfg.get("class_prior", None)
        env_prior = os.getenv("EEG_CLASS_PRIOR", None)
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
            arr = np.array([float(prior_cfg.get(nm, 1/3)) for nm in CLASS_NAMES], dtype=np.float32)
            arr = np.clip(arr, 1e-6, 1.0); arr /= arr.sum()
            self.class_prior = arr

        env_bias = os.getenv("EEG_DECISION_BIAS", None)
        bias_cfg = cfg.get("decision_bias", None)
        if env_bias:
            try: bias_cfg = [float(x) for x in env_bias.split(",")]
            except Exception: pass
        self.decision_bias = np.array(bias_cfg, dtype=np.float32) if isinstance(bias_cfg, (list, tuple)) else np.zeros(3, dtype=np.float32)

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
            if self.device_type == "muse":
                data, srate = _load_muselab_csv(file_path, csv_order=self.csv_order)
            else:
                data, srate = _load_device_csv(file_path, channels=self.channels)
            return data, srate
        elif ext == ".set":
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose='ERROR')
            miss = [ch for ch in self.channels if ch not in raw.ch_names]
            if miss:
                raise ValueError(f"Channels missing in file: {miss}\nPresent: {raw.ch_names}\nExpected: {self.channels}")
            raw.pick_channels(self.channels)
            _maybe_notch(raw)
            raw.filter(LOW_FREQ, HIGH_FREQ, fir_design='firwin', verbose='ERROR')
            raw.resample(TARGET_SRATE, verbose='ERROR')
            try:
                raw.set_eeg_reference('average', projection=False, verbose='ERROR')
            except Exception:
                pass
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

        data, srate = self._read_any(file_path)
        segs = _segment_overlap(data, SEG_SECONDS, EVAL_HOP_SEC, srate)
        need = int((WINDOW_NEED_SECONDS - SEG_SECONDS) / EVAL_HOP_SEC) + 1  # = 47
        N = segs.shape[0]
        if N == 0:
            raise ValueError("No segments could be formed from the recording.")
        if enforce_two_minutes and N < need:
            raise ValueError(f"Too short for 2-minute window: need {need}, got {N}")

        segs_z = _per_record_zscore(segs)
        x = torch.from_numpy(segs_z)[:, None, :, :].to(self.torch_device)

        outs = []
        for i in range(0, x.size(0), BATCH_SIZE):
            outs.append(self.model(x[i:i+BATCH_SIZE]).detach().cpu().numpy().astype(np.float32))
        logits_all = np.concatenate(outs, axis=0)
        probs_all  = _softmax_np(self._apply_calib(logits_all))

        if N < need:
            s_best, use = 0, N
        else:
            s_best, use = self._choose_best_window(probs_all, need)

        block_logits = logits_all[s_best:s_best+use]
        block_probs  = probs_all[s_best:s_best+use]
        y_pred = block_probs.argmax(axis=1)

        counts = {CLASS_NAMES[i]: int((y_pred == i).sum()) for i in range(3)}
        maj_idx = int(np.bincount(y_pred, minlength=3).argmax())
        maj_lbl = CLASS_NAMES[maj_idx]

        w = _quality_weights(segs[s_best:s_best+use])
        wsum = float(w.sum()) + 1e-8
        subj_logit = (self._apply_calib(block_logits) * w[:, None]).sum(axis=0) / wsum
        subj_prob  = _softmax_np(subj_logit[None, :])[0]

        seg_acc = None
        if true_label:
            tl = str(true_label).strip().upper()
            if tl in ("C","A","F"): tl = {"C":"CN","A":"AD","F":"FTD"}[tl]
            if tl in CLASS_NAMES:
                tl_idx = CLASS_NAMES.index(tl)
                seg_acc = float((y_pred == tl_idx).mean())

        sid = subject_id
        if not sid:
            m = re.search(r"(sub-\d+)", file_path, flags=re.IGNORECASE)
            sid = m.group(1) if m else None

        return {
            "channels_used": self.channels,
            "file_path": file_path,
            "n_segments": int(use),
            "prob_mean": {CLASS_NAMES[i]: float(subj_prob[i]) for i in range(3)},
            "segment_accuracy": seg_acc,
            "segment_counts": counts,
            "segment_majority_index": maj_idx,
            "segment_majority_label": maj_lbl,
            "subject_id": sid,
            "window": {"start": int(EVAL_HOP_SEC * s_best), "need": int(WINDOW_NEED_SECONDS)}
        }
