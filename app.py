# -*- coding: utf-8 -*-
import os
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

# --- 엔진/채널 ---
from eeg_model3class import EEGInferenceEngine3Class as EEGEngine3, CHANNEL_GROUPS
from eeg_model2class import EEGInferenceEngine2Class as EEGEngine2

PORT = int(os.getenv("FLASK_PORT", "8000"))
app = Flask(__name__)

# 동일 (device, ver, comment, csv_order) 조합 재사용
_ENGINES2 = {}  # 2-class 캐시
_ENGINES3 = {}  # 3-class 캐시

def _truthy(v, default=True):
    if v is None: return default
    s = str(v).strip().lower()
    return s in ("1","true","on","yes","y")

def _parse_common_params():
    p = request.get_json(force=True) or {}
    file_path = p.get("file_path")
    if not file_path:
        return None, ("file_path is required", 400)

    # device
    device = p.get("device")
    if isinstance(device, str) and device.strip():
        device = device.strip().lower()
        if device not in CHANNEL_GROUPS:
            return None, (f"Unsupported device '{device}'", 400)
    else:
        device = None  # 엔진이 DEFAULT_DEVICE 사용

    # ver/comment
    ver = p.get("ver")
    ver = ver.strip() if isinstance(ver, str) and ver.strip() else None
    comment = p.get("comment")
    comment = (str(comment).strip() if comment is not None else None)

    subject_id = p.get("subject_id")
    true_label = p.get("true_label")
    enforce_two_minutes = _truthy(p.get("enforce_two_minutes"), True)

    # Muse CSV 물리 채널 순서(옵션)
    csv_order_str = p.get("csv_order")
    csv_order = None
    if isinstance(csv_order_str, str) and csv_order_str.strip():
        items = [s.strip().upper() for s in csv_order_str.split(",") if s.strip()]
        if len(items) == 4:
            csv_order = tuple(items)

    parsed = {
        "file_path": file_path,
        "device": device,
        "ver": ver,
        "comment": comment,
        "subject_id": subject_id,
        "true_label": true_label,
        "enforce_two_minutes": enforce_two_minutes,
        "csv_order": csv_order
    }
    return parsed, None

def _engine3(device, ver, comment, csv_order):
    cache_key = (device or "__auto__", ver or "__auto__", comment or "__auto__", csv_order)
    eng = _ENGINES3.get(cache_key)
    if eng is None:
        eng = EEGEngine3(device_type=device, version=ver, comment=comment, csv_order=csv_order)
        _ENGINES3[cache_key] = eng
    return eng

def _engine2(device, ver, comment, csv_order):
    cache_key = (device or "__auto__", ver or "__auto__", comment or "__auto__", csv_order)
    eng = _ENGINES2.get(cache_key)
    if eng is None:
        eng = EEGEngine2(device_type=device, version=ver, comment=comment, csv_order=csv_order)
        _ENGINES2[cache_key] = eng
    return eng

def _normalize_true_label_3(tl: str | None):
    if not tl: return None
    tl = tl.strip().upper()
    if tl in ["C", "CN"]:  return "CN"
    if tl in ["A", "AD"]:  return "AD"
    if tl in ["F", "FTD"]: return "FTD"
    return tl

def _normalize_true_label_2(tl: str | None):
    if not tl: return None
    tl = tl.strip().upper()
    if tl in ["C", "CN"]: return "CN"
    if tl in ["A", "AD"]: return "AD"
    # FTD는 2진분류에서 제외
    return None

@app.get("/health")
def health():
    return jsonify({"status": "flask-ok", "routes": ["/infer(3-class)", "/infer2class(2-class)", "/infer3class(3-class)"]}), 200

def _infer_common(engine_kind: str):
    try:
        parsed, err = _parse_common_params()
        if err:
            msg, code = err
            return jsonify({"status":"error","error":msg}), code

        file_path       = parsed["file_path"]
        device          = parsed["device"]
        ver             = parsed["ver"]
        comment         = parsed["comment"]
        subject_id      = parsed["subject_id"]
        true_label_in   = parsed["true_label"]
        enforce_2min    = parsed["enforce_two_minutes"]
        csv_order       = parsed["csv_order"]

        if engine_kind == "2c":
            engine = _engine2(device, ver, comment, csv_order)
        else:
            engine = _engine3(device, ver, comment, csv_order)

        # 추론
        result = engine.infer(
            file_path=file_path,
            subject_id=subject_id,
            true_label=true_label_in,
            enforce_two_minutes=enforce_2min
        )
        result['class_mode'] = (2 if engine_kind == "2c" else 3)

        # subject-level 예측 레이블
        prob_mean = result.get('prob_mean', {})
        if not prob_mean:
            return jsonify({"status":"error","error":"empty prob_mean"}), 500
        subject_pred_label = max(prob_mean.items(), key=lambda x: x[1])[0]
        result['subject_pred_label'] = subject_pred_label

        # 정확도(옵션)
        if engine_kind == "2c":
            tl_std = _normalize_true_label_2(true_label_in)
            result['true_label'] = tl_std
            result['subject_accuracy'] = (None if tl_std is None
                                          else (1.0 if subject_pred_label == tl_std else 0.0))
        else:
            tl_std = _normalize_true_label_3(true_label_in)
            result['true_label'] = tl_std
            if tl_std is None:
                result['subject_accuracy'] = None
            else:
                result['subject_accuracy'] = 1.0 if subject_pred_label == tl_std else 0.0

        # 편의 필드
        result['subject_probs'] = result['prob_mean']

        return jsonify({"status": "ok", "result": result}), 200

    except FileNotFoundError as e:
        return jsonify({"status":"error","error":str(e)}), 404
    except (ValueError, AssertionError) as e:
        return jsonify({"status":"error","error":str(e)}), 400
    except HTTPException as e:
        return jsonify({"status":"error","error":f"{e.name}: {e.description}"}), e.code
    except Exception as e:
        return jsonify({"status":"error","error":repr(e)}), 500

# --- 라우트 ---
# (1) 기본 /infer: 항상 3진분류
@app.post("/infer")
def infer_default_3():
    return _infer_common("3c")

# (2) 강제 3진분류
@app.post("/infer3class")
def infer_3():
    return _infer_common("3c")

# (3) 강제 2진분류(CN/AD)
@app.post("/infer2class")
def infer_2():
    return _infer_common("2c")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
