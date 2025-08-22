# -*- coding: utf-8 -*-
"""
eeg_model.py
- HuggingFace에서 가중치 자동 다운로드
- 체크포인트 키/하이퍼파라미터(k1=kernel_length, k2=sep_length, F1/D/F2) 자동 추정
- Part10 스타일: 5s 세그먼트, 2.5s hop(50% 겹침), best 2분 창 선택
"""
import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
from huggingface_hub import snapshot_download

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None

# ========================= 기본 설정 =========================
CLASS_NAMES = ['CN', 'AD', 'FTD']

CHANNEL_GROUPS: Dict[str, List[str]] = {
    'muse': ['T5','T6','F7','F8'],  # 선생님 코드 순서에 맞춤
    'hybrid_black': ['Fz','C3','Cz','C4','Pz','T5','T6','O1'],
    'union10': ['T5','T6','F7','F8','Fz','C3','Cz','C4','Pz','O1'],
    'total19': ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2'],
}

LOW_FREQ = 1.0
HIGH_FREQ = 40.0
TARGET_SRATE = 250
SEG_SECONDS = 5.0
EVAL_HOP_SEC = 2.5          # ✔ 50% 겹침
WINDOW_NEED_SECONDS = 120
BATCH_SIZE = 64
EVAL_PER_RECORD_ZSCORE = True

DECISION_BIAS_VEC = np.zeros(len(CLASS_NAMES), dtype=np.float32)
def apply_calibration(logits: np.ndarray) -> np.ndarray:
    return logits  # 필요시 온도스케일링 등 삽입


# ========================= 추가됨 =========================

class EEGInferenceEngine:
    def __init__(self, device_type: str = 'muse',
                 version: Optional[str] = None,
                 torch_device: Optional[str] = None,
                 hf_token: Optional[str] = None,
                 csv_order: Optional[Tuple[str]] = None):   # ✅ 추가
        self.device_type = device_type.lower().strip()
        if self.device_type not in CHANNEL_GROUPS:
            raise ValueError(f"Unknown device_type '{self.device_type}'. Choose one of {list(CHANNEL_GROUPS.keys())}")
        
        # 기본 채널 정의
        self.channels = CHANNEL_GROUPS[self.device_type]

        # ✅ 사용자가 csv_order 지정 시 반영
        if csv_order is not None:
            if len(csv_order) != len(self.channels):
                raise ValueError(f"csv_order length {len(csv_order)} != expected {len(self.channels)} for {self.device_type}")
            self.channels = list(csv_order)

        self.samples_per_seg = int(TARGET_SRATE * SEG_SECONDS)
        self.hop_samples = int(TARGET_SRATE * EVAL_HOP_SEC)
        self.torch_device = torch_device or ('cuda' if torch.cuda.is_available() else 'cpu')

# ========================= 모델 정의 =========================
class EEGNetV4Compat(nn.Module):
    """
    체크포인트 키: firstconv / depthwise / separable / classifier
    k1, k2, F1, D, F2를 외부에서 주입 (HF config.json 또는 state_dict에서 추정)
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

# ========================= 유틸 =========================
def extract_subject_id(file_path: str) -> Optional[str]:
    m = re.search(r"(sub-\d+)", file_path)
    return m.group(1) if m else None

def truthy(x) -> bool:
    if isinstance(x, bool): return x
    if x is None: return False
    return str(x).strip().lower() in ('1','true','on','yes','y')

def _strip_prefix(sd: dict, prefixes=("module.","model.")) -> dict:
    out = {}
    for k, v in sd.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p): kk = kk[len(p):]
        out[kk] = v
    return out

def _load_state_dict(weights_path: str, map_location: str):
    ext = os.path.splitext(weights_path)[-1].lower()
    if ext == ".safetensors":
        if safe_load_file is None:
            raise RuntimeError("pip install safetensors 필요")
        sd = dict(safe_load_file(weights_path))
    else:
        obj = torch.load(weights_path, map_location=map_location)
        if isinstance(obj, dict):
            for k in ["state_dict","model_state_dict","weights","params","model","net"]:
                if k in obj and isinstance(obj[k], dict):
                    sd = obj[k]; break
            else:
                # 순수 state_dict
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
    # firstconv.0.weight 존재 여부로 판정
    return any(k.startswith("firstconv.0.weight") or k.endswith("firstconv.0.weight") for k in sd.keys())

def _infer_hparams_from_sd(sd: dict, chans: int) -> Tuple[int,int,int,int,int,int,int]:
    """
    F1, D, F2, k1, k2, pool1, pool2 추정
    - k1: firstconv.0.weight[..., k1]
    - k2: separable.0.weight[..., k2]  ✔ (수정 포인트)
    - F1: firstconv.0.weight.shape[0]
    - D : depthwise.0.weight.shape[0] // F1
    - F2: classifier.weight.shape[1] (우선) / separable.1.weight.shape[0] (보조)
    - pool1/pool2: 일반적으로 (4,8) — sd로 직접 추정 어려워 기본값 사용
    """
    # 기본값(선생님 코드에 맞춤)
    F1, D, F2, k1, k2, pool1, pool2 = 32, 2, 64, 250, 32, 4, 8
    try:
        if "firstconv.0.weight" in sd:
            w = sd["firstconv.0.weight"]           # [F1, 1, 1, k1]
            F1 = int(w.shape[0]); k1 = int(w.shape[-1])
        if "depthwise.0.weight" in sd:
            w = sd["depthwise.0.weight"]           # [F1*D, 1, chans, 1]
            D = int(w.shape[0] // F1)
        if "separable.0.weight" in sd:
            w = sd["separable.0.weight"]           # [F1*D, 1, 1, k2]
            k2 = int(w.shape[-1])                  # ✔ k2를 체크포인트에서 직접 추정
        if "separable.1.weight" in sd:
            w = sd["separable.1.weight"]           # [F2, F1*D, 1, 1]
            F2 = int(w.shape[0])
        if "classifier.weight" in sd:
            F2 = int(sd["classifier.weight"].shape[1])  # 최종 확정
    except Exception:
        pass
    return F1, D, F2, k1, k2, pool1, pool2

def _load_hf_config(repo_dir: str) -> dict:
    # config.json이 있으면 읽어 하이퍼파라미터 보정
    cfg_path = os.path.join(repo_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            import json
            with open(cfg_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _hf_download(device_type: str, channel_len: int, ver: str, token: Optional[str]) -> Tuple[str, dict]:
    repo_id = f"ardor924/EEGNetV4-{channel_len}ch-{device_type}-Ver{ver}"
    allow_patterns = ["*.pt", "*.pth", "*.bin", "*.safetensors", "config.json"]
    local_dir = snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns, token=token)
    # 가중치 파일 선택
    preferred_exts = (".pt",".pth",".bin",".safetensors")
    weights = []
    for root, _, files in os.walk(local_dir):
        for fn in files:
            if fn.lower().endswith(preferred_exts):
                weights.append(os.path.join(root, fn))
    if not weights:
        raise FileNotFoundError(f"[HF] {repo_id} 에 가중치 파일이 없습니다.")
    weights.sort()
    cfg = _load_hf_config(local_dir)
    return weights[0], cfg

# ========================= 엔진 =========================
class EEGInferenceEngine:
    def __init__(self, device_type: str = 'muse',
                 version: Optional[str] = None,
                 torch_device: Optional[str] = None,
                 hf_token: Optional[str] = None):
        self.device_type = device_type.lower().strip()
        if self.device_type not in CHANNEL_GROUPS:
            raise ValueError(f"Unknown device_type '{self.device_type}'. Choose one of {list(CHANNEL_GROUPS.keys())}")
        self.channels = CHANNEL_GROUPS[self.device_type]
        self.samples_per_seg = int(TARGET_SRATE * SEG_SECONDS)
        self.hop_samples = int(TARGET_SRATE * EVAL_HOP_SEC)
        self.torch_device = torch_device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.version = str(version or os.getenv("EEG_WEIGHTS_VER", "14")).strip()
        self.hf_token = hf_token or os.getenv("HF_TOKEN", None)

        # 1) HF에서 다운로드 + config 로딩
        weights_path, cfg = _hf_download(
            device_type=self.device_type,
            channel_len=len(self.channels),
            ver=self.version,
            token=self.hf_token,
        )

        # 2) state_dict 로딩 후 구조/하이퍼 추정
        sd = _load_state_dict(weights_path, map_location=self.torch_device)
        if not _looks_compat(sd):
            raise RuntimeError("이 체크포인트는 지원되는 키(firstconv/depthwise/separable/classifier)가 아닙니다.")

        F1, D, F2, k1, k2, pool1, pool2 = _infer_hparams_from_sd(sd, chans=len(self.channels))
        # config.json 값이 있으면 우선 적용(안전 보정)
        k1 = int(cfg.get("kernel_length", k1))
        k2 = int(cfg.get("sep_length", k2))
        F1 = int(cfg.get("F1", F1))
        D  = int(cfg.get("D", D))
        pool1 = int(cfg.get("pool1", pool1))
        pool2 = int(cfg.get("pool2", pool2))
        dropout = float(cfg.get("dropout_rate", 0.3))

        # 3) 모델 구성/로드
        self.model = EEGNetV4Compat(
            n_classes=len(CLASS_NAMES),
            Chans=len(self.channels),
            k1=k1, k2=k2, F1=F1, D=D, F2=F2,
            pool1=pool1, pool2=pool2, dropout=dropout
        ).to(self.torch_device)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()

    # -------- 전처리 --------
    def load_raw_fixed_channels(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"EEG file not found: {file_path}")
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose='ERROR')
        missing = [ch for ch in self.channels if ch not in raw.ch_names]
        if missing:
            raise ValueError(f"Channels missing in file: {missing}\nPresent: {raw.ch_names}\nExpected: {self.channels}")
        raw.pick_channels(self.channels)
        raw.filter(LOW_FREQ, HIGH_FREQ, fir_design='firwin', verbose='ERROR')
        raw.resample(TARGET_SRATE)
        return raw

    def _segment_array(self, data: np.ndarray) -> np.ndarray:
        C, T = data.shape
        win = self.samples_per_seg
        hop = max(1, self.hop_samples)
        if T < win:
            return np.empty((0, C, win), dtype=np.float32)
        idxs = list(range(0, T - win + 1, hop))
        segs = np.stack([data[:, i:i+win] for i in idxs], axis=0).astype(np.float32)
        return segs

    def _per_record_zscore(self, segs: np.ndarray) -> np.ndarray:
        mean = segs.mean(axis=(0,2), keepdims=True)
        std = segs.std(axis=(0,2), keepdims=True) + 1e-7
        return (segs - mean) / std

    def _segment_quality_metrics(self, segs: np.ndarray) -> Dict[str, np.ndarray]:
        std_per_seg = segs.std(axis=(1,2))
        med = np.median(std_per_seg)
        bad = std_per_seg < (0.2 * med + 1e-8)
        return {"bad": bad}

    @torch.no_grad()
    def _logits_for_segments(self, segs: np.ndarray) -> np.ndarray:
        if segs.shape[0] == 0:
            return np.empty((0, len(CLASS_NAMES)), dtype=np.float32)
        x = torch.from_numpy(segs)[:, None, :, :].to(self.torch_device, non_blocking=True)
        outs = []
        for i in range(0, x.size(0), BATCH_SIZE):
            outs.append(self.model(x[i:i+BATCH_SIZE]).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0)

    @staticmethod
    def _softmax_np(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)

    def _choose_best_window(self, logits_all: np.ndarray, need_eval: int) -> Tuple[int, int]:
        if logits_all.shape[0] < need_eval:
            return 0, logits_all.shape[0]
        probs = self._softmax_np(logits_all)
        top1 = probs.max(axis=1)
        cumsum = np.concatenate([[0.0], np.cumsum(top1)])
        best, best_sum = 0, -1.0
        for s in range(0, len(top1) - need_eval + 1):
            sm = cumsum[s+need_eval] - cumsum[s]
            if sm > best_sum:
                best_sum, best = sm, s
        return best, need_eval

    # -------- 엔드투엔드 --------
    def infer(self, file_path: str,
              subject_id: Optional[str] = None,
              true_label: Optional[str] = None,
              enforce_two_minutes: bool = True) -> Dict:
        raw = self.load_raw_fixed_channels(file_path)
        data = raw.get_data()  # (C, T)
        segs = self._segment_array(data)

        need_eval = int((WINDOW_NEED_SECONDS - SEG_SECONDS) / EVAL_HOP_SEC) + 1
        N = segs.shape[0]
        if enforce_two_minutes and N < need_eval:
            raise ValueError(f"Recording too short for 2-minute window: need {need_eval} segments, got {N}")

        if EVAL_PER_RECORD_ZSCORE:
            segs = self._per_record_zscore(segs)

        logits_all = self._logits_for_segments(segs)
        if N == 0:
            raise ValueError("No segments could be formed from the recording.")

        s_best, use = (0, N) if N < need_eval else self._choose_best_window(logits_all, need_eval)
        block_logits = logits_all[s_best:s_best+use]
        block_segs   = segs[s_best:s_best+use]

        qm = self._segment_quality_metrics(block_segs)
        w = np.where(qm["bad"], 1e-3, 1.0).astype(np.float32)

        logits_cal  = apply_calibration(block_logits)
        logits_bias = logits_cal - DECISION_BIAS_VEC[None, :]
        probs       = self._softmax_np(logits_bias)

        y_pred_seg = probs.argmax(axis=1)
        counts = {CLASS_NAMES[i]: int((y_pred_seg == i).sum()) for i in range(len(CLASS_NAMES))}
        majority_idx = int(np.bincount(y_pred_seg, minlength=len(CLASS_NAMES)).argmax())
        majority_label = CLASS_NAMES[majority_idx]

        w_sum = float(w.sum()) + 1e-8
        subj_logit = (logits_bias * w[:, None]).sum(axis=0) / w_sum
        subj_prob  = self._softmax_np(subj_logit[None, :])[0]
        prob_mean  = {CLASS_NAMES[i]: float(subj_prob[i]) for i in range(len(CLASS_NAMES))}

        segment_acc = None
        if true_label:
            tl = str(true_label).strip().upper()
            if tl in ("C","A","F"): tl = {"C":"CN","A":"AD","F":"FTD"}[tl]
            if tl in CLASS_NAMES:
                tl_idx = CLASS_NAMES.index(tl)
                segment_acc = float((y_pred_seg == tl_idx).mean())

# -*- coding: utf-8 -*-
"""
eeg_model.py
- HuggingFace에서 가중치 자동 다운로드
- 체크포인트 키/하이퍼파라미터(k1=kernel_length, k2=sep_length, F1/D/F2) 자동 추정
- Part10 스타일: 5s 세그먼트, 2.5s hop(50% 겹침), best 2분 창 선택
"""
import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mne
from huggingface_hub import snapshot_download

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None

# ========================= 기본 설정 =========================
CLASS_NAMES = ['CN', 'AD', 'FTD']

CHANNEL_GROUPS: Dict[str, List[str]] = {
    'muse': ['T5','T6','F7','F8'],  # 선생님 코드 순서에 맞춤
    'hybrid_black': ['Fz','C3','Cz','C4','Pz','T5','T6','O1'],
    'union10': ['T5','T6','F7','F8','Fz','C3','Cz','C4','Pz','O1'],
    'total19': ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2'],
}

LOW_FREQ = 1.0
HIGH_FREQ = 40.0
TARGET_SRATE = 250
SEG_SECONDS = 5.0
EVAL_HOP_SEC = 2.5          # ✔ 50% 겹침
WINDOW_NEED_SECONDS = 120
BATCH_SIZE = 64
EVAL_PER_RECORD_ZSCORE = True

DECISION_BIAS_VEC = np.zeros(len(CLASS_NAMES), dtype=np.float32)
def apply_calibration(logits: np.ndarray) -> np.ndarray:
    return logits  # 필요시 온도스케일링 등 삽입

# ========================= 모델 정의 =========================
class EEGNetV4Compat(nn.Module):
    """
    체크포인트 키: firstconv / depthwise / separable / classifier
    k1, k2, F1, D, F2를 외부에서 주입 (HF config.json 또는 state_dict에서 추정)
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

# ========================= 유틸 =========================
def extract_subject_id(file_path: str) -> Optional[str]:
    m = re.search(r"(sub-\d+)", file_path)
    return m.group(1) if m else None

def truthy(x) -> bool:
    if isinstance(x, bool): return x
    if x is None: return False
    return str(x).strip().lower() in ('1','true','on','yes','y')

def _strip_prefix(sd: dict, prefixes=("module.","model.")) -> dict:
    out = {}
    for k, v in sd.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p): kk = kk[len(p):]
        out[kk] = v
    return out

def _load_state_dict(weights_path: str, map_location: str):
    ext = os.path.splitext(weights_path)[-1].lower()
    if ext == ".safetensors":
        if safe_load_file is None:
            raise RuntimeError("pip install safetensors 필요")
        sd = dict(safe_load_file(weights_path))
    else:
        obj = torch.load(weights_path, map_location=map_location)
        if isinstance(obj, dict):
            for k in ["state_dict","model_state_dict","weights","params","model","net"]:
                if k in obj and isinstance(obj[k], dict):
                    sd = obj[k]; break
            else:
                # 순수 state_dict
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
    # firstconv.0.weight 존재 여부로 판정
    return any(k.startswith("firstconv.0.weight") or k.endswith("firstconv.0.weight") for k in sd.keys())

def _infer_hparams_from_sd(sd: dict, chans: int) -> Tuple[int,int,int,int,int,int,int]:
    """
    F1, D, F2, k1, k2, pool1, pool2 추정
    - k1: firstconv.0.weight[..., k1]
    - k2: separable.0.weight[..., k2]  ✔ (수정 포인트)
    - F1: firstconv.0.weight.shape[0]
    - D : depthwise.0.weight.shape[0] // F1
    - F2: classifier.weight.shape[1] (우선) / separable.1.weight.shape[0] (보조)
    - pool1/pool2: 일반적으로 (4,8) — sd로 직접 추정 어려워 기본값 사용
    """
    # 기본값(선생님 코드에 맞춤)
    F1, D, F2, k1, k2, pool1, pool2 = 32, 2, 64, 250, 32, 4, 8
    try:
        if "firstconv.0.weight" in sd:
            w = sd["firstconv.0.weight"]           # [F1, 1, 1, k1]
            F1 = int(w.shape[0]); k1 = int(w.shape[-1])
        if "depthwise.0.weight" in sd:
            w = sd["depthwise.0.weight"]           # [F1*D, 1, chans, 1]
            D = int(w.shape[0] // F1)
        if "separable.0.weight" in sd:
            w = sd["separable.0.weight"]           # [F1*D, 1, 1, k2]
            k2 = int(w.shape[-1])                  # ✔ k2를 체크포인트에서 직접 추정
        if "separable.1.weight" in sd:
            w = sd["separable.1.weight"]           # [F2, F1*D, 1, 1]
            F2 = int(w.shape[0])
        if "classifier.weight" in sd:
            F2 = int(sd["classifier.weight"].shape[1])  # 최종 확정
    except Exception:
        pass
    return F1, D, F2, k1, k2, pool1, pool2

def _load_hf_config(repo_dir: str) -> dict:
    # config.json이 있으면 읽어 하이퍼파라미터 보정
    cfg_path = os.path.join(repo_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            import json
            with open(cfg_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _hf_download(device_type: str, channel_len: int, ver: str, token: Optional[str]) -> Tuple[str, dict]:
    repo_id = f"ardor924/EEGNetV4-{channel_len}ch-{device_type}-Ver{ver}"
    allow_patterns = ["*.pt", "*.pth", "*.bin", "*.safetensors", "config.json"]
    local_dir = snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns, token=token)
    # 가중치 파일 선택
    preferred_exts = (".pt",".pth",".bin",".safetensors")
    weights = []
    for root, _, files in os.walk(local_dir):
        for fn in files:
            if fn.lower().endswith(preferred_exts):
                weights.append(os.path.join(root, fn))
    if not weights:
        raise FileNotFoundError(f"[HF] {repo_id} 에 가중치 파일이 없습니다.")
    weights.sort()
    cfg = _load_hf_config(local_dir)
    return weights[0], cfg

# ========================= 엔진 =========================
class EEGInferenceEngine:
    def __init__(self, device_type: str = 'muse',
                 version: Optional[str] = None,
                 torch_device: Optional[str] = None,
                 hf_token: Optional[str] = None):
        self.device_type = device_type.lower().strip()
        if self.device_type not in CHANNEL_GROUPS:
            raise ValueError(f"Unknown device_type '{self.device_type}'. Choose one of {list(CHANNEL_GROUPS.keys())}")
        self.channels = CHANNEL_GROUPS[self.device_type]
        self.samples_per_seg = int(TARGET_SRATE * SEG_SECONDS)
        self.hop_samples = int(TARGET_SRATE * EVAL_HOP_SEC)
        self.torch_device = torch_device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.version = str(version or os.getenv("EEG_WEIGHTS_VER", "14")).strip()
        self.hf_token = hf_token or os.getenv("HF_TOKEN", None)

        # 1) HF에서 다운로드 + config 로딩
        weights_path, cfg = _hf_download(
            device_type=self.device_type,
            channel_len=len(self.channels),
            ver=self.version,
            token=self.hf_token,
        )

        # 2) state_dict 로딩 후 구조/하이퍼 추정
        sd = _load_state_dict(weights_path, map_location=self.torch_device)
        if not _looks_compat(sd):
            raise RuntimeError("이 체크포인트는 지원되는 키(firstconv/depthwise/separable/classifier)가 아닙니다.")

        F1, D, F2, k1, k2, pool1, pool2 = _infer_hparams_from_sd(sd, chans=len(self.channels))
        # config.json 값이 있으면 우선 적용(안전 보정)
        k1 = int(cfg.get("kernel_length", k1))
        k2 = int(cfg.get("sep_length", k2))
        F1 = int(cfg.get("F1", F1))
        D  = int(cfg.get("D", D))
        pool1 = int(cfg.get("pool1", pool1))
        pool2 = int(cfg.get("pool2", pool2))
        dropout = float(cfg.get("dropout_rate", 0.3))

        # 3) 모델 구성/로드
        self.model = EEGNetV4Compat(
            n_classes=len(CLASS_NAMES),
            Chans=len(self.channels),
            k1=k1, k2=k2, F1=F1, D=D, F2=F2,
            pool1=pool1, pool2=pool2, dropout=dropout
        ).to(self.torch_device)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()

    # -------- 전처리 --------
    def load_raw_fixed_channels(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"EEG file not found: {file_path}")
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose='ERROR')
        missing = [ch for ch in self.channels if ch not in raw.ch_names]
        if missing:
            raise ValueError(f"Channels missing in file: {missing}\nPresent: {raw.ch_names}\nExpected: {self.channels}")
        raw.pick_channels(self.channels)
        raw.filter(LOW_FREQ, HIGH_FREQ, fir_design='firwin', verbose='ERROR')
        raw.resample(TARGET_SRATE)
        return raw

    def _segment_array(self, data: np.ndarray) -> np.ndarray:
        C, T = data.shape
        win = self.samples_per_seg
        hop = max(1, self.hop_samples)
        if T < win:
            return np.empty((0, C, win), dtype=np.float32)
        idxs = list(range(0, T - win + 1, hop))
        segs = np.stack([data[:, i:i+win] for i in idxs], axis=0).astype(np.float32)
        return segs

    def _per_record_zscore(self, segs: np.ndarray) -> np.ndarray:
        mean = segs.mean(axis=(0,2), keepdims=True)
        std = segs.std(axis=(0,2), keepdims=True) + 1e-7
        return (segs - mean) / std

    def _segment_quality_metrics(self, segs: np.ndarray) -> Dict[str, np.ndarray]:
        std_per_seg = segs.std(axis=(1,2))
        med = np.median(std_per_seg)
        bad = std_per_seg < (0.2 * med + 1e-8)
        return {"bad": bad}

    @torch.no_grad()
    def _logits_for_segments(self, segs: np.ndarray) -> np.ndarray:
        if segs.shape[0] == 0:
            return np.empty((0, len(CLASS_NAMES)), dtype=np.float32)
        x = torch.from_numpy(segs)[:, None, :, :].to(self.torch_device, non_blocking=True)
        outs = []
        for i in range(0, x.size(0), BATCH_SIZE):
            outs.append(self.model(x[i:i+BATCH_SIZE]).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outs, axis=0)

    @staticmethod
    def _softmax_np(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)

    def _choose_best_window(self, logits_all: np.ndarray, need_eval: int) -> Tuple[int, int]:
        if logits_all.shape[0] < need_eval:
            return 0, logits_all.shape[0]
        probs = self._softmax_np(logits_all)
        top1 = probs.max(axis=1)
        cumsum = np.concatenate([[0.0], np.cumsum(top1)])
        best, best_sum = 0, -1.0
        for s in range(0, len(top1) - need_eval + 1):
            sm = cumsum[s+need_eval] - cumsum[s]
            if sm > best_sum:
                best_sum, best = sm, s
        return best, need_eval

    # -------- 엔드투엔드 --------
    def infer(self, file_path: str,
              subject_id: Optional[str] = None,
              true_label: Optional[str] = None,
              enforce_two_minutes: bool = True) -> Dict:
        raw = self.load_raw_fixed_channels(file_path)
        data = raw.get_data()  # (C, T)
        segs = self._segment_array(data)

        need_eval = int((WINDOW_NEED_SECONDS - SEG_SECONDS) / EVAL_HOP_SEC) + 1
        N = segs.shape[0]
        if enforce_two_minutes and N < need_eval:
            raise ValueError(f"Recording too short for 2-minute window: need {need_eval} segments, got {N}")

        if EVAL_PER_RECORD_ZSCORE:
            segs = self._per_record_zscore(segs)

        logits_all = self._logits_for_segments(segs)
        if N == 0:
            raise ValueError("No segments could be formed from the recording.")

        s_best, use = (0, N) if N < need_eval else self._choose_best_window(logits_all, need_eval)
        block_logits = logits_all[s_best:s_best+use]
        block_segs   = segs[s_best:s_best+use]

        qm = self._segment_quality_metrics(block_segs)
        w = np.where(qm["bad"], 1e-3, 1.0).astype(np.float32)

        logits_cal  = apply_calibration(block_logits)
        logits_bias = logits_cal - DECISION_BIAS_VEC[None, :]
        probs       = self._softmax_np(logits_bias)

        # -------------------------------
        # 세그먼트 단위
        # -------------------------------
        y_pred_seg = probs.argmax(axis=1)
        counts = {CLASS_NAMES[i]: int((y_pred_seg == i).sum()) for i in range(len(CLASS_NAMES))}
        majority_idx = int(np.bincount(y_pred_seg, minlength=len(CLASS_NAMES)).argmax())
        majority_label = CLASS_NAMES[majority_idx]

        # -------------------------------
        # Subject-level 계산
        # -------------------------------
        w_sum = float(w.sum()) + 1e-8
        subj_logit = (logits_bias * w[:, None]).sum(axis=0) / w_sum
        subj_prob  = self._softmax_np(subj_logit[None, :])[0]
        subject_probs = {CLASS_NAMES[i]: float(subj_prob[i]) for i in range(len(CLASS_NAMES))}

        subj_pred_idx = int(subj_prob.argmax())
        subj_pred_label = CLASS_NAMES[subj_pred_idx]

        # -------------------------------
        # Accuracy 계산
        # -------------------------------
        segment_acc = None
        subject_acc = None
        if true_label:
            tl = str(true_label).strip().upper()
            if tl in ("C","A","F"):
                tl = {"C":"CN","A":"AD","F":"FTD"}[tl]
            if tl in CLASS_NAMES:
                tl_idx = CLASS_NAMES.index(tl)
                segment_acc = float((y_pred_seg == tl_idx).mean())
                subject_acc = 1.0 if subj_pred_label == tl else 0.0

        # -------------------------------
        # 결과 반환
        # -------------------------------
        return {
            "channels_used": self.channels,
            "file_path": file_path,
            "subject_id": subject_id or extract_subject_id(file_path),
            "n_segments": int(block_segs.shape[0]),

            # Probabilities
            "prob_mean": subject_probs,              # (이전 호환성용)
            "subject_probs": subject_probs,          # ✅ 명확히 subject-level
            "segment_counts": counts,

            # Labels
            "segment_majority_index": majority_idx,
            "segment_majority_label": majority_label,
            "subject_pred_label": subj_pred_label,   # ✅ 최종 subject-level label

            # Accuracy
            "segment_accuracy": segment_acc,
            "subject_accuracy": subject_acc,
            "true_label": true_label,

            # Info
            "window": {"start": int(s_best * EVAL_HOP_SEC), "need": int(WINDOW_NEED_SECONDS)}
        }