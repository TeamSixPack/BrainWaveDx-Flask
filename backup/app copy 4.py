# ---------------------------------------------------------
# Flask 딥러닝 추론 서버 (Spring 연동용)
# - /infer : JSON 입력 받아 동기/비동기 추론 후 결과 반환/콜백
# - 디바이스별 채널 세트 강제 주입 + subject-level 확률/판정 포함
# ---------------------------------------------------------
import os
import uuid
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
# 기본 디바이스(환경변수로 덮어쓰기 가능)
# -----------------------------
DEVICE_NAME = 'muse'
# DEVICE_NAME = 'hybrid_black'
# DEVICE_NAME = 'union10'

EEG_DEVICE = os.getenv("EEG_DEVICE", DEVICE_NAME).strip()
HF_USERNAME = os.getenv("HF_USERNAME", "ardor924").strip()
MODEL_VER   = os.getenv("MODEL_VER", "Ver14").strip()

DEVICE_TO_CHANNELS = {
    "hybrid_black": 8,
    "muse": 4,
    "union10": 10,
}

# 디바이스별 채널 이름(순서 고정)
DEVICE_TO_PICK_CHANNELS = {
    "union10":      ['T5', 'T6', 'F7', 'F8', 'Fz', 'C3', 'Cz', 'C4', 'Pz', 'O1'],
    "muse":         ['T5', 'T6', 'F7', 'F8'],
    "hybrid_black": ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'T5', 'T6', 'O1'],
}
def get_pick_channels(device_name: str):
    return DEVICE_TO_PICK_CHANNELS.get(device_name, DEVICE_TO_PICK_CHANNELS[DEVICE_NAME])

CHANNEL_LEN = str(DEVICE_TO_CHANNELS.get(EEG_DEVICE, 8))
DEFAULT_REPO = f"{HF_USERNAME}/EEGNetV4-{CHANNEL_LEN}ch-{EEG_DEVICE}-{MODEL_VER}"
HF_REPO_ID = os.getenv("HF_REPO_ID", DEFAULT_REPO).strip()

HF_TOKEN   = os.getenv("HF_TOKEN", None)     # ex) hf_xxx...
DEVICE_OPT = os.getenv("DEVICE", None)       # 'cuda' | 'cpu' | None
THREADS    = int(os.getenv("WORKERS", "2"))
DATA_ROOT  = os.getenv("DATA_ROOT", ".")
ENFORCE_2MIN_DEFAULT = os.getenv("ENFORCE_2MIN", "true").lower() == "true"

SPRING_URL       = os.getenv("SPRING_URL", "http://localhost:8090/eeg/result").strip()
SPRING_AUTO_POST = os.getenv("SPRING_AUTO_POST", "true").lower() == "true"

EXEC = ThreadPoolExecutor(max_workers=THREADS)
JOBS = {}

def _boollike(x, default=False):
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default

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
        raise SystemExit(
            f"\n❌ Hugging Face 모델 로드 실패\n"
            f"   - repo_id: {HF_REPO_ID}\n"
            f"   - reason : {e}\n"
            f"   - 확인사항: 1) repo_id 정확성  2) 비공개 레포면 HF_TOKEN  3) EEG_DEVICE/HF_USERNAME/MODEL_VER 일치\n"
        )

    DEVICE_RESOLVED = "cuda" if torch.cuda.is_available() else "cpu"
    app.logger.info(
        f"✅ HF model loaded: {HF_REPO_ID} | device={DEVICE_RESOLVED} | n_chans={CFG.get('n_chans')} | labels={CFG.get('idx_to_label')}"
    )
    app.logger.info(f"✅ server EEG_DEVICE={EEG_DEVICE}, server pick_channels={get_pick_channels(EEG_DEVICE)}")

    expected_ch = DEVICE_TO_CHANNELS.get(EEG_DEVICE)
    cfg_ch = int(CFG.get("n_chans", expected_ch))
    if expected_ch != cfg_ch:
        app.logger.warning(f"⚠ 모델 n_chans({cfg_ch}) vs 서버 디바이스({EEG_DEVICE}:{expected_ch}) 불일치 가능 — HF_REPO_ID 확인 필요.")

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
            "server_pick_channels": get_pick_channels(EEG_DEVICE),
        }), 200

    @app.route("/infer", methods=["POST"])
    def infer():
        try:
            payload = request.get_json(force=True)
            if payload is None:
                return jsonify({"error": "no JSON payload"}), 400

            # 🔎 들어온 값 로깅(필요하면 주석 처리)
            app.logger.info(f"[infer] payload keys={list(payload.keys())}")


            # --- 입력 키: camelCase/snake_case 모두 허용 ---
            file_path = (payload.get("file_path")
                         or payload.get("filePath") or "").strip()
            if not file_path:
                return jsonify({"error": "file_path is required (.set)"}), 400
            if not os.path.isabs(file_path):
                file_path = os.path.normpath(os.path.join(DATA_ROOT, file_path))
            if not os.path.exists(file_path):
                return jsonify({"error": f"file not found: {file_path}"}), 404

            req_device = (payload.get("device")
                          or payload.get("channel_name")
                          or payload.get("channelName")
                          or EEG_DEVICE).strip()
            if req_device != EEG_DEVICE:
                return jsonify({
                    "error": "device mismatch",
                    "message": f"Server loaded for '{EEG_DEVICE}'. You requested '{req_device}'. "
                               f"Restart with EEG_DEVICE={req_device} (and matching HF_REPO_ID)."
                }), 400

            pick_channels = get_pick_channels(EEG_DEVICE)
            cfg_for_run = dict(CFG)
            cfg_for_run["pick_channels"] = pick_channels
            cfg_for_run["n_chans"] = len(pick_channels)

            # subject_id: 경로에서 추출이 진실
            sid_from_path = extract_subject_id(file_path)
            req_sid = payload.get("subject_id") or payload.get("subjectId")
            if req_sid and req_sid != sid_from_path:
                app.logger.warning(f"subject_id mismatch ignored: req={req_sid} path={sid_from_path}")
            subject_id = sid_from_path

            # true label & 2분 옵션 (둘 다 양식 허용)
            true_label   = payload.get("true_label") or payload.get("trueLabel")
            enforce_2min = payload.get("enforce_two_minutes")
            if enforce_2min is None:
                enforce_2min = payload.get("enforceTwoMinutes", ENFORCE_2MIN_DEFAULT)
            enforce_2min = _boollike(enforce_2min, ENFORCE_2MIN_DEFAULT)

            callback_url = payload.get("callback_url") or payload.get("callbackUrl")

            # 비동기
            if callback_url:
                job_id = str(uuid.uuid4())
                JOBS[job_id] = {"status": "queued", "result": None}
                EXEC.submit(_run_infer_and_callback, job_id, file_path, subject_id,
                            true_label, callback_url, enforce_2min, cfg_for_run)
                return jsonify({"status": "accepted", "job_id": job_id, "file_path": file_path}), 202

            # 동기
            out = predict_from_eeglab_file(
                file_path=file_path, model=MODEL, cfg=cfg_for_run,
                device=DEVICE_RESOLVED, true_label=true_label,
                enforce_two_minutes=enforce_2min
            )

            def camelize(d: dict) -> dict:
                def ckey(k: str) -> str:
                    parts = k.split('_')
                    return parts[0] + ''.join(p.title() for p in parts[1:])
                return {ckey(k): v for k, v in d.items()}

            res = {
                "subject_id": out["subject_id"],
                "file_path": out["file_path"],
                "n_segments": out["n_segments"],
                "segment_counts": out["segment_counts"],
                "segment_majority_index": out["segment_majority_index"],
                "segment_majority_label": out["segment_majority_label"],
                "segment_accuracy": out["segment_accuracy"],
                "prob_mean": out["prob_mean"],
                "subject_probs": out["subject_probs"],
                "subject_pred_index": out["subject_pred_index"],
                "subject_pred_label": out["subject_pred_label"],
                "subject_accuracy": out["subject_accuracy"],
                "true_label": out.get("true_label"),
                "true_label_raw": out.get("true_label_raw"),  # 🔎 디버깅용
                "channels_used": out["channels_used"],
                "channels_map": out["channels_map"],
                "raw_channel_names": out["raw_channel_names"],
                "window": out["window"],
            }
            res.update(camelize(res))

            if SPRING_AUTO_POST:
                EXEC.submit(_post_to_spring, SPRING_URL, {"result": res})

            return jsonify({"status": "ok", "result": res}), 200

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
        print(f"[WARN] Spring POST failed → {url} | reason={e}")

def _run_infer(file_path: str, subject_id: str, true_label, enforce_2min: bool, cfg_for_run: dict):
    out = predict_from_eeglab_file(
        file_path=file_path,
        model=MODEL,
        cfg=cfg_for_run,  # ← 이 요청에 맞춘 pick_channels 강제 사용
        device="cuda" if torch.cuda.is_available() else "cpu",
        true_label=true_label,
        enforce_two_minutes=enforce_2min
    )
    # snake+camel 모두 가진 결과를 그대로 반환하려면 app.infer()에서 만든 포맷을 재사용하는 편이 낫지만
    # 여기서는 테스트/콜백 용도로 최소 필드만 유지합니다.
    res = {
        "subject_id": out["subject_id"],
        "file_path": out["file_path"],
        "n_segments": out["n_segments"],
        "segment_counts": out["segment_counts"],
        "segment_majority_index": out["segment_majority_index"],
        "segment_majority_label": out["segment_majority_label"],
        "segment_accuracy": out["segment_accuracy"],
        "prob_mean": out["prob_mean"],
        "subject_probs": out["subject_probs"],
        "subject_pred_index": out["subject_pred_index"],
        "subject_pred_label": out["subject_pred_label"],
        "subject_accuracy": out["subject_accuracy"],
        "true_label": out.get("true_label"),
        "channels_used": out["channels_used"],
        "channels_map": out["channels_map"],
        "raw_channel_names": out["raw_channel_names"],
        "window": out["window"],
    }
    return res


def _run_infer_and_callback(job_id: str, file_path: str, subject_id: str, true_label, callback_url: str, enforce_2min: bool, cfg_for_run: dict):
    JOBS[job_id]["status"] = "running"
    try:
        result = _run_infer(file_path, subject_id, true_label, enforce_2min, cfg_for_run)
        JOBS[job_id]["result"] = result
        JOBS[job_id]["status"] = "done"

        try:
            resp = requests.post(callback_url, json={"job_id": job_id, "result": result}, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"[WARN] user callback POST failed → {callback_url} | reason={e}")

        if SPRING_AUTO_POST and (callback_url.strip() != SPRING_URL.strip()):
            _post_to_spring(SPRING_URL, {"job_id": job_id, "result": result})

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["result"] = {"error": str(e)}

# =========================
# 엔트리 포인트
# =========================
if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=8000)
