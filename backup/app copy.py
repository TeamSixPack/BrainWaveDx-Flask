# app.py
# ---------------------------------------------------------
# Flask 딥러닝 추론 서버 (Spring 연동용)
# - 시작 시 Hugging Face에서 모델 로드(디바이스별 자동 repo 선택)
# - /infer : JSON 입력 받아 동기/비동기 추론 후 결과 반환/콜백
# - prob_mean 포함 전체 결과를 Spring(기본 8090)으로 자동 POST
# ---------------------------------------------------------
import os
import uuid
import json
import traceback
from concurrent.futures import ThreadPoolExecutor

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
from eeg_model import (
    load_hf_model,
    predict_from_eeglab_file,
    extract_subject_id,
)

# -----------------------------
# 기본 설정(환경변수로 덮어쓰기 가능)
# -----------------------------
# 디바이스: hybrid_black | muse | union10
EEG_DEVICE = os.getenv("EEG_DEVICE", "muse").strip()

# 사용자/버전(필요 시 변경)
HF_USERNAME = os.getenv("HF_USERNAME", "ardor924").strip()
MODEL_VER   = os.getenv("MODEL_VER", "Ver10").strip()

# 채널 개수(디바이스별 기본값)
DEVICE_TO_CHANNELS = {
    "hybrid_black": 8,
    "muse": 4,
    "union10": 10,
}
CHANNEL_LEN = str(DEVICE_TO_CHANNELS.get(EEG_DEVICE, 8))

# repo_id 자동 구성 (환경변수 HF_REPO_ID가 있으면 그것을 우선 사용)
DEFAULT_REPO = f"{HF_USERNAME}/EEGNetV4-{CHANNEL_LEN}ch-{EEG_DEVICE}-{MODEL_VER}"
HF_REPO_ID = os.getenv("HF_REPO_ID", DEFAULT_REPO).strip()

# 허깅페이스 토큰(비공개 레포면 필요)
HF_TOKEN   = os.getenv("HF_TOKEN", None)               # ex) hf_xxx...
DEVICE_OPT = os.getenv("DEVICE", None)                 # 'cuda' | 'cpu' | None
THREADS    = int(os.getenv("WORKERS", "2"))
DATA_ROOT  = os.getenv("DATA_ROOT", ".")               # 상대 경로 처리용(옵션)
ENFORCE_2MIN_DEFAULT = os.getenv("ENFORCE_2MIN", "true").lower() == "true"

# Spring 자동 POST 설정
SPRING_URL       = os.getenv("SPRING_URL", "http://localhost:8090/eeg/result").strip()
SPRING_AUTO_POST = os.getenv("SPRING_AUTO_POST", "true").lower() == "true"

EXEC = ThreadPoolExecutor(max_workers=THREADS)
JOBS = {}

def create_app():
    app = Flask(__name__)
    CORS(app)

    # -----------------------------
    # 모델 로드 (서버 시작 시 1회)
    # -----------------------------
    global MODEL, CFG, DEVICE_RESOLVED
    try:
        MODEL, CFG = load_hf_model(HF_REPO_ID, device=DEVICE_OPT, token=HF_TOKEN)
    except Exception as e:
        # 레포 접근 실패 시, 친절한 에러 메시지
        raise SystemExit(
            f"\n❌ Hugging Face 모델 로드 실패\n"
            f"   - repo_id: {HF_REPO_ID}\n"
            f"   - reason : {e}\n"
            f"   - 확인사항: 1) repo_id가 정확한지  2) 비공개 레포라면 HF_TOKEN 환경변수 설정\n"
            f"              3) EEG_DEVICE/HF_USERNAME/MODEL_VER 조합이 실제 레포와 일치하는지\n"
        )

    DEVICE_RESOLVED = "cuda" if torch.cuda.is_available() else "cpu"
    app.logger.info(
        f"✅ HF model loaded: {HF_REPO_ID} | device={DEVICE_RESOLVED} | chans={CFG.get('n_chans')} | labels={CFG.get('idx_to_label')}"
    )

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "device": DEVICE_RESOLVED,
            "repo": HF_REPO_ID,
            "spring_auto_post": SPRING_AUTO_POST,
            "spring_url": SPRING_URL,
            "eeg_device": EEG_DEVICE,
            "channels": CFG.get("n_chans"),
        }), 200

    @app.route("/infer", methods=["POST"])
    def infer():
        try:
            payload = request.get_json(force=True)
            if payload is None:
                return jsonify({"error": "no JSON payload"}), 400

            file_path = str(payload.get("file_path", "")).strip()
            if not file_path:
                return jsonify({"error": "file_path is required (.set)"}), 400
            if not os.path.isabs(file_path):
                file_path = os.path.normpath(os.path.join(DATA_ROOT, file_path))
            if not os.path.exists(file_path):
                return jsonify({"error": f"file not found: {file_path}"}), 404

            # ✅ 무조건 파일 경로에서 subject_id를 추출 (요청 값과 무관)
            sid_from_path = extract_subject_id(file_path)
            req_sid = payload.get("subject_id")
            if req_sid and req_sid != sid_from_path:
                app.logger.warning(f"subject_id mismatch ignored: req={req_sid} path={sid_from_path}")
            subject_id   = sid_from_path

            true_label   = payload.get("true_label", None)
            callback_url = payload.get("callback_url", None)
            enforce_2min = payload.get("enforce_two_minutes", ENFORCE_2MIN_DEFAULT)


            # 비동기 콜백 요청이 들어온 경우
            if callback_url:
                job_id = str(uuid.uuid4())
                JOBS[job_id] = {"status": "queued", "result": None}
                EXEC.submit(
                    _run_infer_and_callback,
                    job_id, file_path, subject_id, true_label, callback_url, enforce_2min
                )
                # 별도로 Spring 자동 POST도 설정되어 있으면, 그쪽에도 보내기
                if SPRING_AUTO_POST and (callback_url.strip() != SPRING_URL.strip()):
                    # 같은 스레드 안에서 중복 전송 방지: 콜백 완료 후 내부에서 보내도록 처리
                    pass
                return jsonify({"status": "accepted", "job_id": job_id, "file_path": file_path}), 202

            # 동기 처리
            result = _run_infer(file_path, subject_id, true_label, enforce_2min)

            # Spring 자동 POST (옵션)
            if SPRING_AUTO_POST:
                EXEC.submit(_post_to_spring, SPRING_URL, {"result": result})

            return jsonify({"status": "ok", "result": result}), 200

        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/result/<job_id>", methods=["GET"])
    def result(job_id):
        info = JOBS.get(job_id)
        if not info:
            return jsonify({"error": "invalid job_id"}), 404
        return jsonify(info), 200

    return app

# =========================
# 내부 유틸
# =========================
def _post_to_spring(url: str, payload: dict):
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        # 실패해도 서버는 죽지 않게 로깅만(실환경이면 로거 사용 권장)
        print(f"[WARN] Spring POST failed → {url} | reason={e}")

def _run_infer(file_path: str, subject_id: str, true_label, enforce_2min: bool):
    out = predict_from_eeglab_file(
        file_path=file_path,
        model=MODEL,
        cfg=CFG,
        device=DEVICE_RESOLVED,
        true_label=true_label,
        enforce_two_minutes=enforce_2min
    )
    res = {
        # ✅ 모델이 반환한 subject_id를 신뢰
        "subject_id": out["subject_id"],

        "file_path": out["file_path"],
        "n_segments": out["n_segments"],
        "segment_counts": out["segment_counts"],
        "segment_majority_index": out["segment_majority_index"],
        "segment_majority_label": out["segment_majority_label"],
        "segment_accuracy": out["segment_accuracy"],
        "subject_pred_index": out.get("subject_pred_index"),
        "subject_pred_label": out.get("subject_pred_label"),
        "subject_accuracy": out.get("subject_accuracy"),
        "prob_mean": out["prob_mean"],
        "window": out["window"],
        "channels_used": out["channels_used"],
    }
    return res


def _run_infer_and_callback(job_id: str, file_path: str, subject_id: str, true_label, callback_url: str, enforce_2min: bool):
    JOBS[job_id]["status"] = "running"
    try:
        result = _run_infer(file_path, subject_id, true_label, enforce_2min)
        JOBS[job_id]["result"] = result
        JOBS[job_id]["status"] = "done"

        # 1) 사용자 제공 콜백에 전송
        try:
            resp = requests.post(callback_url, json={"job_id": job_id, "result": result}, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"[WARN] user callback POST failed → {callback_url} | reason={e}")

        # 2) ✅ Spring에도 자동 POST (prob_mean 포함)
        if SPRING_AUTO_POST and (callback_url.strip() != SPRING_URL.strip()):
            _post_to_spring(SPRING_URL, {"job_id": job_id, "result": result})

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["result"] = {"error": str(e)}

# =========================
# 엔트리 포인트
# =========================
if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", "8000"))
    app = create_app()
    # app.run(host=host, port=port, debug=False)
    app.run(host="127.0.0.1", port=8000) # 강제실행

