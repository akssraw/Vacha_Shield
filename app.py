from __future__ import annotations

import base64
import datetime
import io
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import numpy as np
import speech_recognition as sr
import torch
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory
from flask_cors import CORS
from groq import Groq
from werkzeug.utils import secure_filename

from deepfake_detector import predict_deepfake_from_file
from model import AudioCNN

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = os.getenv("VACHA_ENV_FILE")
if ENV_FILE:
    load_dotenv(dotenv_path=ENV_FILE)
else:
    load_dotenv(dotenv_path=BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

UPLOAD_DIR = BASE_DIR / "temp_uploads"
FLAGGED_DIR = BASE_DIR / "flagged_calls"
CONTINUOUS_DATASET_DIR = BASE_DIR / "continuous_learning_dataset"
CALIBRATION_PATH = BASE_DIR / "model_calibration.json"
STT_COMPATIBLE_EXTENSIONS = {".wav", ".flac", ".aif", ".aiff", ".aifc"}
ALERT_DEFAULT_FLOOR = float(os.getenv("ALERT_MIN_THRESHOLD", "0.62"))
DEFAULT_ANALYSIS_PROFILE = os.getenv("DEFAULT_ANALYSIS_PROFILE", "strict").strip().lower()
SEMANTIC_OVERRIDE_ENABLED = os.getenv("SEMANTIC_OVERRIDE_ENABLED", "false").strip().lower() == "true"

ANALYSIS_PROFILES: dict[str, dict[str, float]] = {
    "conservative": {
        "sensitivity": 0.40,
        "decision_floor": 0.66,
        "model_weight": 0.84,
        "artifact_weight": 0.16,
        "chunk_seconds": 1.0,
        "hop_seconds": 0.5,
        "borderline_margin": 0.03,
    },
    "balanced": {
        "sensitivity": 0.48,
        "decision_floor": ALERT_DEFAULT_FLOOR,
        "model_weight": 0.82,
        "artifact_weight": 0.18,
        "chunk_seconds": 1.0,
        "hop_seconds": 0.5,
        "borderline_margin": 0.04,
    },
    "strict": {
        "sensitivity": 0.62,
        "decision_floor": 0.58,
        "model_weight": 0.78,
        "artifact_weight": 0.22,
        "chunk_seconds": 0.95,
        "hop_seconds": 0.4,
        "borderline_margin": 0.06,
    },
    "forensic": {
        "sensitivity": 0.80,
        "decision_floor": 0.58,
        "model_weight": 0.72,
        "artifact_weight": 0.28,
        "chunk_seconds": 0.8,
        "hop_seconds": 0.3,
        "borderline_margin": 0.08,
    },
}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _find_lovable_dist() -> Path | None:
    candidates = sorted(
        BASE_DIR.glob("lovable-project-*/dist"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


LOVABLE_DIST_DIR = _find_lovable_dist()

# Serve built Lovable UI when available, otherwise keep legacy static folder behavior.
app = Flask(
    __name__,
    static_folder=str(LOVABLE_DIST_DIR) if LOVABLE_DIST_DIR else "static",
    static_url_path="/static",
    template_folder="templates",
)
CORS(app, resources={r"/*": {"origins": "*"}})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running Inference API on: {DEVICE}")

def _load_base_threshold(default: float = 0.50) -> float:
    if not CALIBRATION_PATH.exists():
        return default
    try:
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        threshold = float(payload.get("threshold", default))
        if 0.1 <= threshold <= 0.9:
            print(f"Loaded calibrated threshold from model_calibration.json: {threshold:.3f}")
            return threshold
    except Exception as e:
        print(f"Could not load model calibration file: {e}")
    return default


def _coerce_float(value, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return float(np.clip(parsed, minimum, maximum))


def _resolve_analysis_profile(profile_name: str | None) -> tuple[str, dict[str, float]]:
    candidate = (profile_name or DEFAULT_ANALYSIS_PROFILE).strip().lower()
    if candidate not in ANALYSIS_PROFILES:
        candidate = DEFAULT_ANALYSIS_PROFILE if DEFAULT_ANALYSIS_PROFILE in ANALYSIS_PROFILES else "balanced"
    return candidate, dict(ANALYSIS_PROFILES[candidate])


def _extract_analysis_params(form_data) -> dict[str, float | str]:
    profile_name, preset = _resolve_analysis_profile(form_data.get("analysis_profile"))

    chunk_seconds = _coerce_float(form_data.get("chunk_seconds"), preset["chunk_seconds"], 0.35, 3.0)
    hop_seconds = _coerce_float(form_data.get("hop_seconds"), preset["hop_seconds"], 0.1, 2.0)
    hop_seconds = min(hop_seconds, max(0.1, chunk_seconds * 0.9))

    model_weight = _coerce_float(form_data.get("model_weight"), preset["model_weight"], 0.05, 0.95)
    artifact_weight = _coerce_float(form_data.get("artifact_weight"), preset["artifact_weight"], 0.05, 0.95)

    return {
        "profile": profile_name,
        "sensitivity": _coerce_float(form_data.get("sensitivity"), preset["sensitivity"], 0.0, 1.0),
        "decision_floor": _coerce_float(form_data.get("decision_floor"), preset["decision_floor"], 0.35, 0.8),
        "model_weight": model_weight,
        "artifact_weight": artifact_weight,
        "chunk_seconds": chunk_seconds,
        "hop_seconds": hop_seconds,
        "borderline_margin": _coerce_float(form_data.get("borderline_margin"), preset["borderline_margin"], 0.02, 0.2),
    }

# Base decision threshold. The detector applies small per-clip adaptive changes.
DEEPFAKE_THRESHOLD = _load_base_threshold(0.50)

try:
    model = AudioCNN(num_classes=1)
    model_path = BASE_DIR / "model.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Loaded model weights from 'model.pth'.")
    else:
        print("WARNING: 'model.pth' not found. The model is using random weights.")
    model.to(DEVICE)
    model.eval()
except Exception as e:
    print(f"Critical error loading model: {e}")
    model = None


def generate_spectrogram_base64(audio_path: Path) -> str | None:
    """Generate a compact base64 mel spectrogram image for UI rendering."""
    try:
        y, sr_rate = librosa.load(str(audio_path), sr=16000, res_type="kaiser_fast")
        mel = librosa.feature.melspectrogram(y=y, sr=sr_rate, n_mels=30)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        plt.figure(figsize=(8, 3))
        plt.style.use("dark_background")
        librosa.display.specshow(mel_db, sr=sr_rate, x_axis="time", y_axis="mel", cmap="magma")
        plt.tight_layout(pad=0)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", transparent=True, dpi=72, bbox_inches="tight")
        plt.close("all")

        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        return None


def _semantic_keyword_hit(transcript: str) -> bool:
    text = transcript.lower().strip()
    phrases = [
        "i am an ai",
        "i'm an ai",
        "i am a bot",
        "i'm a bot",
        "i am a virtual assistant",
        "i am an artificial intelligence",
        "this is an ai voice",
        "this voice is generated",
    ]
    return any(phrase in text for phrase in phrases)


def semantic_deepfake_check(audio_path: Path) -> bool:
    """
    Optional semantic override:
    if transcript explicitly self-identifies as AI, raise a guaranteed flag.
    """
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(str(audio_path)) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            print(f"[STT] '{text}'")

        if _semantic_keyword_hit(text):
            print("[SEMANTIC] Explicit AI self-identification detected via keyword rules.")
            return True

        if not GROQ_API_KEY:
            return False

        client = Groq(api_key=GROQ_API_KEY)
        prompt = (
            "Classify this transcript as TRUE or FALSE only. "
            "TRUE if the speaker explicitly identifies as AI, bot, or virtual assistant. "
            f"Transcript: {text!r}"
        )
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0,
            max_tokens=5,
        )

        result = completion.choices[0].message.content.strip().upper()
        semantic_hit = "TRUE" in result
        if semantic_hit:
            print("[SEMANTIC] LLM override triggered.")
        return semantic_hit
    except sr.UnknownValueError:
        return False
    except sr.RequestError as e:
        print(f"STT unavailable: {e}")
        return False
    except Exception as e:
        print(f"Semantic engine error: {e}")
        return False


def _cleanup_paths(*paths: Path) -> None:
    for path in paths:
        if path and path.exists():
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass


def _convert_webm_to_wav(source_webm: Path, target_wav: Path) -> Path:
    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i",
                str(source_webm),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(target_wav),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return target_wav
    except Exception as e:
        print(f"WEBM conversion failed, falling back to original input: {e}")
        return source_webm


@app.route("/health", methods=["GET"])
def health() -> tuple[dict, int]:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "ui_mode": "lovable-dist" if LOVABLE_DIST_DIR else "legacy-templates",
        "default_analysis_profile": DEFAULT_ANALYSIS_PROFILE,
        "default_alert_floor": ALERT_DEFAULT_FLOOR,
        "semantic_override_enabled": SEMANTIC_OVERRIDE_ENABLED,
    }, 200


@app.route("/legacy", methods=["GET"])
def legacy_index():
    return render_template("index.html")


@app.route("/legacy/mobile", methods=["GET"])
def legacy_mobile():
    return render_template("mobile.html")


@app.route("/detect_voice", methods=["POST"])
def detect_voice():
    if model is None:
        return jsonify({"error": "AI model failed to load."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No 'file' part provided in request."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    request_id = uuid.uuid4().hex
    ext = Path(file.filename).suffix.lower() or ".wav"
    raw_path = UPLOAD_DIR / f"{request_id}{ext}"
    wav_path = UPLOAD_DIR / f"{request_id}.wav"

    force_alert = request.form.get("force_alert", "false").lower() == "true"
    analysis_params = _extract_analysis_params(request.form)

    process_path = raw_path

    try:
        file.save(raw_path)

        if ext == ".webm":
            process_path = _convert_webm_to_wav(raw_path, wav_path)

        result = predict_deepfake_from_file(
            audio_path=str(process_path),
            model=model,
            device=DEVICE,
            threshold=DEEPFAKE_THRESHOLD,
            chunk_seconds=float(analysis_params["chunk_seconds"]),
            hop_seconds=float(analysis_params["hop_seconds"]),
            sensitivity=float(analysis_params["sensitivity"]),
            model_weight=float(analysis_params["model_weight"]),
            artifact_weight=float(analysis_params["artifact_weight"]),
        )

        if force_alert:
            result["synthetic_probability"] = 0.98
            result["human_probability"] = 0.02
            result["alert"] = True
            result["threshold"] = max(DEEPFAKE_THRESHOLD, float(analysis_params["decision_floor"]))
            print("[DEMO MODE] Forced alert enabled.")
        else:
            if SEMANTIC_OVERRIDE_ENABLED:
                if result.get("max_amplitude", 0.0) >= 0.005 and process_path.suffix.lower() in STT_COMPATIBLE_EXTENSIONS:
                    if semantic_deepfake_check(process_path):
                        result["synthetic_probability"] = 0.99
                        result["human_probability"] = 0.01
                        result["alert"] = True
                        print("[SEMANTIC OVERRIDE] Speaker claimed to be AI.")

        # Guardrail is tunable per request via analysis parameters.
        effective_threshold = max(float(result.get("threshold", DEEPFAKE_THRESHOLD)), float(analysis_params["decision_floor"]))
        result["threshold"] = round(effective_threshold, 4)
        result["alert"] = bool(result["synthetic_probability"] > effective_threshold)
        result["analysis_profile"] = str(analysis_params["profile"])
        merged_analysis_params = dict(result.get("analysis_parameters") or {})
        merged_analysis_params.update(
            {
                "decision_floor": round(float(analysis_params["decision_floor"]), 4),
                "borderline_margin": round(float(analysis_params["borderline_margin"]), 4),
            }
        )
        result["analysis_parameters"] = merged_analysis_params
        if result["alert"]:
            result["verdict"] = "ai_clone"
        elif result["synthetic_probability"] >= (effective_threshold - float(analysis_params["borderline_margin"])):
            result["verdict"] = "borderline_human"
        else:
            result["verdict"] = "human"
        result["spectrogram_base64"] = generate_spectrogram_base64(process_path)

        if result["alert"]:
            FLAGGED_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prob = int(float(result["synthetic_probability"]) * 100)
            flagged_path = FLAGGED_DIR / f"deepfake_log_{timestamp}_{request_id[:8]}_prob{safe_prob}.wav"
            shutil.copy(process_path, flagged_path)
            print(f"[FORENSICS] Logged suspicious call: {flagged_path}")

        print(
            "[INFERENCE] "
            f"synthetic={result['synthetic_probability']:.4f} "
            f"model={result['model_probability']:.4f} "
            f"artifact={result['artifact_probability']:.4f} "
            f"threshold={result['threshold']:.4f} "
            f"chunks={result['chunk_count']} "
            f"profile={analysis_params['profile']} "
            f"sensitivity={analysis_params['sensitivity']:.2f}"
        )

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if process_path == raw_path:
            _cleanup_paths(raw_path)
        else:
            _cleanup_paths(raw_path, process_path)


@app.route("/feedback", methods=["POST"])
def feedback():
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    label = request.form.get("label")
    if label not in {"human", "ai"}:
        return jsonify({"error": "Invalid label. Use 'human' or 'ai'."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Invalid file."}), 400

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cleaned_name = secure_filename(file.filename)
    save_dir = CONTINUOUS_DATASET_DIR / label
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"feedback_{timestamp}_{uuid.uuid4().hex[:8]}_{cleaned_name}"
    file.save(save_path)

    print(f"[CONTINUOUS LEARNING] Stored labeled clip -> {label.upper()} at {save_path}")

    return jsonify({"success": True, "message": "Feedback recorded."})


if LOVABLE_DIST_DIR:

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_lovable(path: str):
        # Protect API routes from being shadowed by the SPA fallback.
        api_prefixes = ("detect_voice", "feedback", "health", "legacy")
        if path.startswith(api_prefixes):
            abort(404)

        if path:
            asset_path = LOVABLE_DIST_DIR / path
            if asset_path.exists() and asset_path.is_file():
                return send_from_directory(str(LOVABLE_DIST_DIR), path)

        return send_from_directory(str(LOVABLE_DIST_DIR), "index.html")

else:

    @app.route("/", methods=["GET"])
    def index_fallback():
        return redirect("/legacy")

    @app.route("/mobile", methods=["GET"])
    def mobile_fallback():
        return redirect("/legacy/mobile")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "false").strip().lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
