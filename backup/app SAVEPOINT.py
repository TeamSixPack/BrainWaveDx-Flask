# -*- coding: utf-8 -*-
"""
app.py
- /health, /infer
- 요청 예:
  {
    "file_path": "/path/to/muselab.csv or .set",
    "device": "muse",            // 기본: "muse"
    "ver": "14",                 // 기본: ENV EEG_WEIGHTS_VER 또는 "14"
    "subject_id": "sub-089",     // 옵션
    "true_label": "CN",          // 옵션: "CN"|"AD"|"FTD"|("C"|"A"|"F")
    "enforce_two_minutes": true, // 기본 true
    "csv_order": "TP9,AF7,AF8,TP10" // 옵션(생략 가능). 기본은 표준 MuseLab 순서
  }
"""
import os
from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException
from eeg_model import EEGInferenceEngine, CHANNEL_GROUPS

PORT = int(os.getenv("FLASK_PORT", "8000"))
app = Flask(__name__)

# 간단 엔진 캐시(동일 (device, ver, csv_order) 조합 재사용)
_ENGINES = {}

def _truthy(v, default=True):
    if v is None: return default
    s = str(v).strip().lower()
    return s in ("1","true","on","yes","y")

@app.get("/health")
def health():
    return jsonify({"status": "flask-ok"}), 200

@app.post("/infer")
def infer():
    try:
        p = request.get_json(force=True) or {}
        file_path = p.get("file_path")
        if not file_path:
            return jsonify({"status":"error","error":"file_path is required"}), 400
        
        device = (p.get("device") or "muse").strip().lower()
        if device not in CHANNEL_GROUPS:
            return jsonify({"status":"error","error":f"Unsupported device '{device}'"}), 400
        
        ver = str(p.get("ver") or os.getenv("EEG_WEIGHTS_VER","14")).strip()
        subject_id = p.get("subject_id")
        true_label = p.get("true_label")
        enforce_two_minutes = _truthy(p.get("enforce_two_minutes"), True)
        
        # Muse CSV 물리 채널 순서(옵션): "TP9,AF7,AF8,TP10"
        csv_order_str = p.get("csv_order")
        csv_order = None
        if csv_order_str:
            items = [s.strip().upper() for s in str(csv_order_str).split(",") if s.strip()]
            if len(items) == 4:
                csv_order = tuple(items)
        
        # 엔진 캐시 키
        cache_key = (device, ver, csv_order)
        engine = _ENGINES.get(cache_key)
        if engine is None:
            engine = EEGInferenceEngine(device_type=device, version=ver, csv_order=csv_order)
            _ENGINES[cache_key] = engine
        
        result = engine.infer(file_path=file_path, subject_id=subject_id, true_label=true_label, enforce_two_minutes=enforce_two_minutes)
        
        # subject-level 예측 레이블 계산
        prob_mean = result['prob_mean']
        subject_pred_label = max(prob_mean.items(), key=lambda x: x[1])[0]
        result['subject_pred_label'] = subject_pred_label
        
        # true_label이 제공된 경우 정확도 계산
        if true_label:
            # true_label 표준화 (C->CN, A->AD, F->FTD)
            tl = true_label.strip().upper()
            if tl in ["C", "CN"]:
                tl_std = "CN"
            elif tl in ["A", "AD"]:
                tl_std = "AD"
            elif tl in ["F", "FTD"]:
                tl_std = "FTD"
            else:
                tl_std = tl
            
            result['true_label'] = tl_std
            result['subject_accuracy'] = 1.0 if subject_pred_label == tl_std else 0.0
        else:
            result['true_label'] = None
            result['subject_accuracy'] = None
        
        # subject_probs 필드 추가 (prob_mean과 동일)
        result['subject_probs'] = result['prob_mean']
        
        return jsonify({"status": "ok", "result": result}), 200
        
    except FileNotFoundError as e:
        return jsonify({"status":"error","error":str(e)}), 404
    except (ValueError, AssertionError) as e:
        return jsonify({"status":"error","error":str(e)}), 400
    except HTTPException as e:
        return jsonify({"status":"error","error":f"{e.name}: {e.description}"}), e.code
    except Exception as e:
        # 예기치 않은 오류도 추적 가능하도록 repr 포함
        return jsonify({"status":"error","error":repr(e)}), 500

if __name__ == "__main__":
    # 디버그 서버는 단일 스레드라 캐시 race 이슈가 없지만,
    # 운영 시에는 WSGI(예: gunicorn) 사용을 권장드립니다.
    app.run(host="0.0.0.0", port=PORT, debug=False)