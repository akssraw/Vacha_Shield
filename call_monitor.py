import datetime
import json
import os
import shutil
import time
import wave
from pathlib import Path

import pyaudio
import torch

from deepfake_detector import predict_deepfake_from_file
from model import AudioCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
CHUNK_SECONDS = 4
FRAMES_PER_BUFFER = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

BASE_DIR = Path(__file__).resolve().parent
TEMP_FILE = BASE_DIR / "ambient_temp_call.wav"
FLAGGED_DIR = BASE_DIR / "flagged_calls"
CALIBRATION_PATH = BASE_DIR / "model_calibration.json"
MODEL_PATH = BASE_DIR / "model.pth"

# Borderline outputs should not trigger hard alert.
ALERT_FLOOR_THRESHOLD = 0.55
CONSECUTIVE_ALERTS_REQUIRED = 2
ALERT_COOLDOWN_SECONDS = 12
MONITOR_SENSITIVITY = 0.72
MONITOR_MODEL_WEIGHT = 0.76
MONITOR_ARTIFACT_WEIGHT = 0.24
MONITOR_CHUNK_SECONDS = 0.9
MONITOR_HOP_SECONDS = 0.35

FLAGGED_DIR.mkdir(parents=True, exist_ok=True)


def load_base_threshold(default: float = 0.5) -> float:
    if not CALIBRATION_PATH.exists():
        return default
    try:
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        threshold = float(payload.get("threshold", default))
        return threshold if 0.1 <= threshold <= 0.9 else default
    except Exception:
        return default


def load_model() -> AudioCNN | None:
    try:
        if not MODEL_PATH.exists():
            print("[!] model.pth not found. Train model first.")
            return None
        model = AudioCNN(num_classes=1)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as exc:
        print(f"[!] Failed to load model: {exc}")
        return None


def record_chunk(temp_file: Path) -> bool:
    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
        )

        frames = []
        total_frames = int(SAMPLE_RATE / FRAMES_PER_BUFFER * CHUNK_SECONDS)
        for _ in range(total_frames):
            data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        with wave.open(str(temp_file), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b"".join(frames))

        return True
    except OSError as exc:
        print(f"[!] Microphone error: {exc}")
        return False
    finally:
        p.terminate()


def log_flagged_clip(temp_file: Path, probability: float) -> None:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prob = int(probability * 100)
    destination = FLAGGED_DIR / f"deepfake_log_{timestamp}_prob{safe_prob}.wav"
    shutil.copy(str(temp_file), str(destination))
    print(f"[+] Forensic clip saved: {destination}")


def simulate_call_monitor() -> None:
    model = load_model()
    if model is None:
        return

    base_threshold = load_base_threshold(0.5)
    print("=" * 64)
    print("VACHA-SHIELD CALL MONITOR (BACKGROUND MODE)")
    print("=" * 64)
    print(f"[*] Device: {DEVICE}")
    print(f"[*] Base threshold: {base_threshold:.2f}")
    print(f"[*] Alert floor threshold: {ALERT_FLOOR_THRESHOLD:.2f}")
    print(f"[*] Consecutive alerts required: {CONSECUTIVE_ALERTS_REQUIRED}")

    try:
        input("Press ENTER to start call monitoring...")
    except KeyboardInterrupt:
        return

    streak = 0
    cycle = 1
    last_alert_time = 0.0

    try:
        while True:
            if not record_chunk(TEMP_FILE):
                break

            result = predict_deepfake_from_file(
                audio_path=str(TEMP_FILE),
                model=model,
                device=DEVICE,
                threshold=base_threshold,
                chunk_seconds=MONITOR_CHUNK_SECONDS,
                hop_seconds=MONITOR_HOP_SECONDS,
                sensitivity=MONITOR_SENSITIVITY,
                model_weight=MONITOR_MODEL_WEIGHT,
                artifact_weight=MONITOR_ARTIFACT_WEIGHT,
            )

            effective_threshold = max(float(result.get("threshold", base_threshold)), ALERT_FLOOR_THRESHOLD)
            synthetic = float(result["synthetic_probability"])
            human = float(result["human_probability"])

            if synthetic > effective_threshold:
                streak += 1
            else:
                streak = 0

            now = time.time()
            can_alert = (now - last_alert_time) >= ALERT_COOLDOWN_SECONDS

            if streak >= CONSECUTIVE_ALERTS_REQUIRED and can_alert:
                print(
                    f"\n[ALERT] cycle={cycle} | AI={synthetic:.3f} | Human={human:.3f} | "
                    f"threshold={effective_threshold:.3f} | streak={streak}"
                )
                log_flagged_clip(TEMP_FILE, synthetic)
                last_alert_time = now
                streak = 0
            else:
                print(
                    f"[cycle {cycle}] human={human:.3f} ai={synthetic:.3f} "
                    f"thr={effective_threshold:.3f} streak={streak}",
                    end="\r",
                )

            if TEMP_FILE.exists():
                TEMP_FILE.unlink(missing_ok=True)

            cycle += 1

    except KeyboardInterrupt:
        print("\n[*] Monitoring stopped by user.")
    finally:
        if TEMP_FILE.exists():
            TEMP_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    simulate_call_monitor()
