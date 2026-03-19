import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import librosa
import torch

from feature_extraction import extract_dual_channel_from_waveform


def _clamp(value: float, low: float, high: float) -> float:
    return float(np.clip(float(value), low, high))


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _chunk_signal(audio: np.ndarray, chunk_samples: int, hop_samples: int) -> list[np.ndarray]:
    if audio is None or len(audio) == 0:
        return []

    chunks = []
    for start in range(0, len(audio), hop_samples):
        end = start + chunk_samples
        chunk = audio[start:end]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk.astype(np.float32))
        if end >= len(audio):
            break
    return chunks


def _voice_activity_ratio(audio: np.ndarray, frame_length: int = 400, hop_length: int = 160) -> float:
    if audio is None or len(audio) == 0:
        return 0.0
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length).flatten()
    if rms.size == 0:
        return 0.0
    threshold = max(0.0015, float(np.mean(rms) * 0.45))
    return float(np.mean(rms > threshold))


def _chunk_quality_weight(chunk: np.ndarray) -> float:
    if chunk is None or len(chunk) == 0:
        return 0.2

    rms = float(np.sqrt(np.mean(np.square(chunk))) + 1e-8)
    vad_ratio = _voice_activity_ratio(chunk)

    # We prefer chunks that are audible and contain active speech.
    rms_score = float(np.clip(rms * 6.0, 0.2, 1.0))
    vad_score = float(np.clip(vad_ratio * 1.4, 0.2, 1.0))
    return float(np.clip(0.55 * rms_score + 0.45 * vad_score, 0.2, 1.0))


def _load_audio_for_inference(audio_path: str, sample_rate: int) -> tuple[np.ndarray, Path | None]:
    source_path = Path(audio_path)
    load_path = source_path
    temp_wav_path: Path | None = None

    if source_path.suffix.lower() == ".webm":
        import imageio_ffmpeg

        temp_fd, temp_name = tempfile.mkstemp(prefix="vacha_infer_", suffix=".wav")
        os.close(temp_fd)
        temp_wav_path = Path(temp_name)

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i",
                str(source_path),
                "-ar",
                str(sample_rate),
                "-ac",
                "1",
                str(temp_wav_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        load_path = temp_wav_path

    y, _ = librosa.load(str(load_path), sr=sample_rate, res_type="kaiser_fast")
    return np.asarray(y, dtype=np.float32), temp_wav_path


def _artifact_probability(audio: np.ndarray, sample_rate: int) -> tuple[float, dict]:
    """
    Handcrafted anti-spoof cues that complement the CNN on unseen clone engines.
    Returns (probability, diagnostics).
    """
    if audio is None or len(audio) == 0:
        return 0.5, {}

    try:
        rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=160).flatten()
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=400, hop_length=160).flatten()

        rms_mean = float(np.mean(rms)) if rms.size else 0.0
        rms_std = float(np.std(rms)) if rms.size else 0.0
        rms_cv = rms_std / (rms_mean + 1e-8)
        zcr_std = float(np.std(zcr)) if zcr.size else 0.0

        stft = np.abs(librosa.stft(audio, n_fft=1024, hop_length=160)) ** 2
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=1024)
        high_mask = freqs >= 4000
        total_energy = float(np.sum(stft) + 1e-8)
        high_energy = float(np.sum(stft[high_mask, :])) if stft.size else 0.0
        high_freq_ratio = high_energy / total_energy

        spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
        contrast_mean = float(np.mean(spectral_contrast)) if spectral_contrast.size else 0.0
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, roll_percent=0.85, hop_length=160).flatten()
        rolloff_mean = float(np.mean(rolloff)) if rolloff.size else 0.0
        spectral_flatness = librosa.feature.spectral_flatness(y=audio, n_fft=1024, hop_length=160).flatten()
        flatness_mean = float(np.mean(spectral_flatness)) if spectral_flatness.size else 0.0

        vad_ratio = _voice_activity_ratio(audio)
        duration_seconds = float(len(audio) / sample_rate)

        score = 0.5

        # Deep-cloned speech often has lower short-term dynamics and smoother contours.
        if rms_cv < 0.20:
            score += 0.14
        elif rms_cv < 0.28:
            score += 0.07
        elif rms_cv > 0.46:
            score -= 0.10

        if zcr_std < 0.014:
            score += 0.10
        elif zcr_std < 0.020:
            score += 0.05
        elif zcr_std > 0.035:
            score -= 0.08

        if high_freq_ratio < 0.07:
            score += 0.10
        elif high_freq_ratio < 0.11:
            score += 0.05
        elif high_freq_ratio > 0.19:
            score -= 0.06

        if contrast_mean < 15.0:
            score += 0.08
        elif contrast_mean > 21.5:
            score -= 0.06

        if vad_ratio > 0.93:
            score += 0.05
        elif vad_ratio < 0.38:
            score += 0.03

        if duration_seconds < 2.5:
            score += 0.04

        score = float(np.clip(score, 0.05, 0.95))

        diagnostics = {
            "rms_cv": round(rms_cv, 5),
            "zcr_std": round(zcr_std, 5),
            "high_freq_ratio": round(high_freq_ratio, 5),
            "spectral_contrast": round(contrast_mean, 5),
            "rolloff_mean": round(rolloff_mean, 2),
            "spectral_flatness": round(flatness_mean, 5),
            "vad_ratio": round(vad_ratio, 5),
        }

        return score, diagnostics
    except Exception:
        return 0.5, {}


def predict_deepfake_from_file(
    audio_path: str,
    model,
    device,
    threshold: float = 0.50,
    chunk_seconds: float = 1.0,
    hop_seconds: float = 0.5,
    sensitivity: float = 0.50,
    model_weight: float = 0.82,
    artifact_weight: float = 0.18,
    low_amplitude_guard: float = 0.01,
    stability_bonus: float = 0.02,
) -> dict:
    """
    Hybrid deepfake scoring:
      1) CNN probability on overlapping chunks
      2) acoustic artifact probability
      3) fusion with adaptive thresholding
    """
    sample_rate = 16000
    chunk_seconds = _clamp(_safe_float(chunk_seconds, 1.0), 0.35, 3.0)
    hop_seconds = _clamp(_safe_float(hop_seconds, 0.5), 0.1, 2.5)
    if hop_seconds >= chunk_seconds:
        hop_seconds = max(0.1, chunk_seconds * 0.8)

    sensitivity = _clamp(_safe_float(sensitivity, 0.5), 0.0, 1.0)
    model_weight = _clamp(_safe_float(model_weight, 0.82), 0.05, 0.95)
    artifact_weight = _clamp(_safe_float(artifact_weight, 0.18), 0.05, 0.95)
    weight_sum = model_weight + artifact_weight
    model_weight = model_weight / weight_sum
    artifact_weight = artifact_weight / weight_sum
    low_amplitude_guard = _clamp(_safe_float(low_amplitude_guard, 0.01), 0.001, 0.05)
    stability_bonus = _clamp(_safe_float(stability_bonus, 0.02), 0.0, 0.08)

    chunk_samples = int(sample_rate * chunk_seconds)
    hop_samples = int(sample_rate * hop_seconds)

    temp_wav_path: Path | None = None
    try:
        y, temp_wav_path = _load_audio_for_inference(audio_path, sample_rate=sample_rate)

        if y.size == 0:
            return {
                "synthetic_probability": 0.0,
                "human_probability": 1.0,
                "alert": False,
                "threshold": round(threshold, 4),
                "model_probability": 0.0,
                "artifact_probability": 0.5,
                "chunk_probability_mean": 0.0,
                "chunk_probability_std": 0.0,
                "chunk_count": 0,
                "audio_duration_seconds": 0.0,
                "max_amplitude": 0.0,
                "artifact_signals": {},
                "analysis_parameters": {
                    "chunk_seconds": round(chunk_seconds, 3),
                    "hop_seconds": round(hop_seconds, 3),
                    "sensitivity": round(sensitivity, 3),
                    "model_weight": round(model_weight, 4),
                    "artifact_weight": round(artifact_weight, 4),
                },
            }

        max_amp = float(np.max(np.abs(y)))
        if max_amp > 0.0:
            y = librosa.util.normalize(y)

        chunk_probs: list[float] = []
        chunk_weights: list[float] = []
        chunks = _chunk_signal(y, chunk_samples=chunk_samples, hop_samples=hop_samples)

        with torch.no_grad():
            for chunk in chunks:
                features = extract_dual_channel_from_waveform(chunk, sample_rate=sample_rate, max_pad_len=400)
                if features is None:
                    continue

                tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
                logit = model(tensor)
                prob = float(torch.sigmoid(logit).item())

                chunk_probs.append(prob)
                chunk_weights.append(_chunk_quality_weight(chunk))

        if not chunk_probs:
            chunk_probs = [0.0]
            chunk_weights = [1.0]

        probs_np = np.asarray(chunk_probs, dtype=np.float32)
        weights_np = np.asarray(chunk_weights, dtype=np.float32)

        weighted_mean = float(np.average(probs_np, weights=weights_np)) if np.sum(weights_np) > 0 else float(np.mean(probs_np))
        top_k = max(1, int(np.ceil(len(probs_np) * 0.33)))
        top_k_mean = float(np.mean(np.sort(probs_np)[-top_k:]))
        model_probability = float(np.clip(0.65 * weighted_mean + 0.35 * top_k_mean, 0.0, 1.0))

        artifact_probability, artifact_diagnostics = _artifact_probability(y, sample_rate=sample_rate)

        fused_probability = float(
            np.clip(model_weight * model_probability + artifact_weight * artifact_probability, 0.0, 1.0)
        )

        duration_seconds = float(len(y) / sample_rate)
        rolloff_mean = float(artifact_diagnostics.get("rolloff_mean", 0.0))
        flatness_mean = float(artifact_diagnostics.get("spectral_flatness", 0.0))
        robustness_penalty = 0.0
        if duration_seconds < 6.5 and model_probability > 0.84 and flatness_mean > 0.018 and rolloff_mean > 2700:
            robustness_penalty = 0.13
            fused_probability = float(np.clip(fused_probability - robustness_penalty, 0.0, 1.0))

        chunk_std = float(np.std(probs_np)) if probs_np.size > 1 else 0.0
        dynamic_threshold = threshold

        if max_amp < low_amplitude_guard:
            dynamic_threshold += 0.05

        if probs_np.size >= 4 and chunk_std < 0.05:
            dynamic_threshold -= stability_bonus

        dynamic_threshold += (0.5 - sensitivity) * 0.14
        dynamic_threshold = float(np.clip(dynamic_threshold, 0.40, 0.72))
        alert = bool(fused_probability > dynamic_threshold)

        return {
            "synthetic_probability": round(fused_probability, 4),
            "human_probability": round(1.0 - fused_probability, 4),
            "alert": alert,
            "threshold": round(dynamic_threshold, 4),
            "model_probability": round(model_probability, 4),
            "artifact_probability": round(artifact_probability, 4),
            "chunk_probability_mean": round(float(np.mean(probs_np)), 4),
            "chunk_probability_std": round(chunk_std, 4),
            "chunk_count": int(len(probs_np)),
            "audio_duration_seconds": round(duration_seconds, 3),
            "max_amplitude": round(max_amp, 5),
            "artifact_signals": artifact_diagnostics,
            "analysis_parameters": {
                "chunk_seconds": round(chunk_seconds, 3),
                "hop_seconds": round(hop_seconds, 3),
                "sensitivity": round(sensitivity, 3),
                "model_weight": round(model_weight, 4),
                "artifact_weight": round(artifact_weight, 4),
                "low_amplitude_guard": round(low_amplitude_guard, 4),
                "stability_bonus": round(stability_bonus, 4),
                "robustness_penalty": round(robustness_penalty, 4),
            },
        }
    finally:
        if temp_wav_path and temp_wav_path.exists():
            temp_wav_path.unlink(missing_ok=True)
