# -*- coding: utf-8 -*-
"""
app.py
- Flask 서버 (포트 8000)
- POST /infer : 스프링 InferRequest DTO 준수 + device/ver 옵션
"""
import os
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

from eeg_model import EEGInferenceEngine, truthy, CHANNEL_GROUPS

PORT = int(os.getenv("FLASK_PORT", "8000"))

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "flask-ok"}), 200


def parse_bool(value, default=True):
    if value is None:
        return default
    return truthy(value)


@app.post("/infer")
def infer():
    """
    요청 JSON:
      {
        "file_path": ".../sub-044_task-eyesclosed_eeg.set",
        "true_label": "CN" | "FTD" | "AD",       (옵션)
        "subject_id": "sub-044",                 (옵션)
        "enforce_two_minutes": true | false,     (옵션, 기본 true)
        "device": "muse|hybrid_black|union10|total19", (옵션, 기본 "muse")
        "ver": "14"                              (옵션, 기본 ENV EEG_WEIGHTS_VER 또는 "14")
      }
    """
    try:
        payload = request.get_json(force=True, silent=False) or {}
        file_path = payload.get("file_path")
        true_label = payload.get("true_label")
        subject_id = payload.get("subject_id")
        enforce_two_minutes = parse_bool(payload.get("enforce_two_minutes"), default=True)

        device_type = (payload.get("device") or "muse").strip().lower()
        if device_type not in CHANNEL_GROUPS:
            return jsonify({
                "status": "error",
                "error": f"Unsupported device '{device_type}'. "
                         f"Choose one of {list(CHANNEL_GROUPS.keys())}"
            }), 400

        ver = str(payload.get("ver") or os.getenv("EEG_WEIGHTS_VER", "14")).strip()

        if not file_path:
            return jsonify({"status": "error", "error": "file_path is required"}), 400

        engine = EEGInferenceEngine(
            device_type=device_type,
            version=ver,
        )
        result = engine.infer(
            file_path=file_path,
            subject_id=subject_id,
            true_label=true_label,
            enforce_two_minutes=enforce_two_minutes
        )
        
        # 추가된 로직: prob_mean에서 가장 높은 확률의 레이블을 찾아서 추가
        if 'prob_mean' in result:
            subject_probs = result['prob_mean']
            subject_pred_label = max(subject_probs, key=subject_probs.get)
            result['subject_pred_label'] = subject_pred_label

        return jsonify({"status": "ok", "result": result}), 200

    except FileNotFoundError as e:
        return jsonify({"status": "error", "error": str(e)}), 404
    except (ValueError, AssertionError) as e:
        return jsonify({"status": "error", "error": str(e)}), 400
    except HTTPException as e:
        return jsonify({"status": "error", "error": f"{e.name}: {e.description}"}), e.code
    except Exception as e:
        return jsonify({"status": "error", "error": repr(e)}), 500
    

    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
