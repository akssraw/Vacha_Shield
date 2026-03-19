import argparse
import asyncio
import datetime
import json
import os
import random
import shutil
import time
from pathlib import Path

import edge_tts
import librosa
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from dotenv import load_dotenv
from edge_tts import VoicesManager
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from approved_sources import CATEGORY_AI, CATEGORY_HUMAN, iter_registered_audio_files, list_sources, write_registry_index
from feature_extraction import extract_dual_channel_from_waveform
from model import AudioCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_yesno_transcript(file_path: Path) -> str:
    # File format: 0_1_0_1_... where 0=no and 1=yes
    token_map = {"0": "no", "1": "yes"}
    return " ".join(token_map.get(part, part) for part in file_path.stem.split("_"))


def download_yesno(root: Path) -> tuple[list[Path], list[str]]:
    # Reuse existing download if available.
    candidates = [
        root / "waves_yesno",
        root.parent / "waves_yesno",
        Path("data") / "internet_raw" / "waves_yesno",
    ]
    for candidate in candidates:
        if candidate.exists():
            existing = sorted(candidate.glob("*.wav"))
            if existing:
                print(f"[+] Reusing existing YESNO data from {candidate}")
                transcripts = [parse_yesno_transcript(path) for path in existing]
                return existing, transcripts

    print("[*] Downloading YESNO dataset from internet...")
    root.mkdir(parents=True, exist_ok=True)

    # This triggers download/extract. We do not call __getitem__ because some setups
    # require torchcodec for torchaudio's internal decoding.
    for attempt in range(3):
        try:
            torchaudio.datasets.YESNO(root=str(root), download=True)
            break
        except PermissionError as exc:
            if attempt == 2:
                raise
            print(f"[!] YESNO download lock detected, retrying ({attempt + 1}/3): {exc}")
            time.sleep(2.0)

    wav_dir = root / "waves_yesno"
    audio_files = sorted(wav_dir.glob("*.wav"))
    transcripts = [parse_yesno_transcript(path) for path in audio_files]

    print(f"[+] YESNO ready: {len(audio_files)} human clips")
    return audio_files, transcripts


def download_librispeech(root: Path, subset: str, max_files: int, seed: int) -> tuple[list[Path], list[str]]:
    subset_dir = root / "LibriSpeech" / subset
    if subset_dir.exists():
        print(f"[+] Reusing existing LibriSpeech '{subset}' from {subset_dir}")
    else:
        print(f"[*] Downloading LibriSpeech '{subset}' from internet...")
    root.mkdir(parents=True, exist_ok=True)

    if not subset_dir.exists():
        for attempt in range(3):
            try:
                torchaudio.datasets.LIBRISPEECH(root=str(root), url=subset, download=True)
                break
            except PermissionError as exc:
                if attempt == 2:
                    raise
                print(f"[!] LibriSpeech download lock detected, retrying ({attempt + 1}/3): {exc}")
                time.sleep(2.0)
            except Exception as exc:
                # Download may still succeed even if runtime decode backend is missing.
                print(f"[!] LibriSpeech loader warning: {exc}")
                break

    if not subset_dir.exists():
        print("[!] LibriSpeech folder not found after download; skipping this source.")
        return [], []

    transcript_map: dict[str, str] = {}
    for trans_file in subset_dir.rglob("*.trans.txt"):
        with open(trans_file, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    transcript_map[parts[0]] = parts[1]

    flac_files = list(subset_dir.rglob("*.flac"))
    if not flac_files:
        print("[!] No LibriSpeech .flac files found; skipping this source.")
        return [], []

    rng = random.Random(seed)
    rng.shuffle(flac_files)
    selected = flac_files[:max_files]
    transcripts = [transcript_map.get(path.stem, "") for path in selected]

    print(f"[+] LibriSpeech ready: {len(selected)} human clips")
    return selected, transcripts


SCAM_PHRASES = [
    "Hello, this is your bank security team. Please verify the one-time password now.",
    "I lost my wallet and phone, can you urgently transfer money to this account?",
    "Your account has suspicious activity. Share your card details to avoid suspension.",
    "I need you to update the vendor payment immediately before the deadline.",
    "This is an emergency call from your family member, please respond quickly.",
    "Your KYC is expired. Confirm your PIN and date of birth right now.",
    "We detected a failed transaction. Approve this transfer immediately.",
    "I am calling from support, install this app so I can fix your device remotely.",
]

HINDI_SCAM_PHRASES = [
    "??????, ??? ???? ???? ?? ??????? ??? ?? ??? ??? ???? ????? ??? ????? ??????",
    "???? ??????? ?????? ?? ?? ??, ????? ???? ??? ?? ???????? ????? ???? ?????",
    "???? ???? ??? ??????? ???-??? ???? ??, ?????? ????? ?? ??? ????? ?? ?????? ?????",
    "??? ?????? ?????? ?? ??? ??? ???, ???? ??? ?? ?????? ?? ???? ?? ??? ?? ?? ??????? ??????",
    "???? ?????? ??? ???????? ??, ????? ?? ???? ?? ???? ??????",
]

HINGLISH_SCAM_PHRASES = [
    "Hello ma'am, aapke account me suspicious activity detect hui hai, please OTP abhi share kijiye.",
    "Sir, KYC update nahi hua hai, warna aapka account temporary hold ho jayega.",
    "Main support team se bol raha hoon, payment fail hua hai, ek baar PIN confirm kar dijiye.",
    "Aapke family member ne emergency transfer ke liye bola hai, please abhi UPI kar dijiye.",
    "Aapka card block hone wala hai, verification ke liye date of birth aur OTP batayein.",
]

HINDI_SUPPORT_PHRASES = [
    "??????, ??? ???? ??? ?? ??? ???? ???? ????? ???? ?????? ???? ?? ??????",
    "?? ?? ?????? ????? ??? ??? ??, ????? ??? ?? ???????",
    "???? ???? ??? ?? ??????? ?? ??? ??? ????? ???? ??? ?????? ???? ?? ?????",
]

HINGLISH_SUPPORT_PHRASES = [
    "Hi, aap apni problem clearly bata do, main step by step help karunga.",
    "Kal client call hai, please final presentation deck shaam tak bhej dena.",
    "Agar budget thoda flexible hai to hum better option shortlist kar sakte hain.",
]


def build_multilingual_prompt_pool(transcripts: list[str]) -> list[tuple[str, str]]:
    prompt_pool: list[tuple[str, str]] = []

    for text in transcripts:
        cleaned = text.strip()
        if cleaned:
            prompt_pool.append((cleaned, "en"))

    prompt_pool.extend((text, "en") for text in SCAM_PHRASES)
    prompt_pool.extend((text, "hi") for text in HINDI_SCAM_PHRASES)
    prompt_pool.extend((text, "hi") for text in HINDI_SUPPORT_PHRASES)
    prompt_pool.extend((text, "hinglish") for text in HINGLISH_SCAM_PHRASES)
    prompt_pool.extend((text, "hinglish") for text in HINGLISH_SUPPORT_PHRASES)

    return prompt_pool


def _dedupe_voice_names(voices: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for voice in voices:
        if not voice or voice in seen:
            continue
        seen.add(voice)
        unique.append(voice)
    return unique


def build_edge_voice_buckets(voice_manager: VoicesManager) -> dict[str, list[str]]:
    raw_voices = list(getattr(voice_manager, "voices", []) or [])
    if not raw_voices:
        raw_voices.extend(voice_manager.find(Language="en"))
        raw_voices.extend(voice_manager.find(Language="hi"))

    buckets = {"en": [], "hi": [], "multi": [], "all": []}
    for voice in raw_voices:
        name = voice.get("Name") or voice.get("ShortName")
        if not name:
            continue

        descriptor = " ".join(
            str(voice.get(key, ""))
            for key in ("Locale", "Language", "Name", "ShortName", "FriendlyName")
        ).lower()

        buckets["all"].append(name)
        if "multilingual" in descriptor:
            buckets["multi"].append(name)
        if any(tag in descriptor for tag in ("hi-in", " hindi", "hindi ", "language: hi")) or descriptor.startswith("hi"):
            buckets["hi"].append(name)
        if any(tag in descriptor for tag in ("en-us", "en-gb", "en-in", " english", "language: en")) or descriptor.startswith("en"):
            buckets["en"].append(name)

    for key, voices in buckets.items():
        buckets[key] = _dedupe_voice_names(voices)

    if not buckets["en"]:
        buckets["en"] = buckets["multi"][:]
    if not buckets["hi"]:
        buckets["hi"] = _dedupe_voice_names(buckets["multi"] + buckets["en"])
    if not buckets["all"]:
        raise RuntimeError("No usable voices returned by edge-tts service")

    return buckets


def choose_edge_voice(language_hint: str, buckets: dict[str, list[str]], rng: random.Random) -> str:
    if language_hint == "hi":
        candidates = buckets["hi"] or buckets["multi"] or buckets["all"]
    elif language_hint == "hinglish":
        candidates = _dedupe_voice_names(buckets["hi"] + buckets["multi"] + buckets["en"]) or buckets["all"]
    else:
        candidates = buckets["en"] or buckets["multi"] or buckets["all"]
    return rng.choice(candidates)


PREMIUM_TTS_BASE_URL = "https://api.elevenlabs.io/v1"


def list_local_ai_files(local_ai_dir: Path) -> list[Path]:
    if not local_ai_dir.exists():
        return []
    files: list[Path] = []
    for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.webm"):
        files.extend(local_ai_dir.glob(ext))
    return sorted(files)


def _premium_tts_headers(api_key: str) -> dict[str, str]:
    return {
        "xi-api-key": api_key,
        "accept": "application/json",
        "content-type": "application/json",
    }


def fetch_premium_tts_voice_ids(api_key: str, max_voices: int = 12) -> list[str]:
    try:
        response = requests.get(
            f"{PREMIUM_TTS_BASE_URL}/voices",
            headers=_premium_tts_headers(api_key),
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        voices = data.get("voices", [])
        voice_ids = [voice.get("voice_id") for voice in voices if voice.get("voice_id")]
        return voice_ids[:max_voices]
    except Exception as exc:
        print(f"[!] Could not fetch premium provider voices: {exc}")
        return []


def generate_premium_tts_clones(
    output_dir: Path,
    transcripts: list[str],
    target_count: int,
    seed: int,
    api_key: str | None,
    model_id: str = "eleven_multilingual_v2",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("*.mp3")) + sorted(output_dir.glob("*.wav"))
    if len(existing) >= target_count:
        print(f"[+] Reusing {target_count} existing premium provider AI clips")
        return existing[:target_count]

    if target_count <= 0:
        return existing

    if not api_key:
        print("[!] Provider API key missing. Skipping premium provider generation.")
        return existing

    print(f"[*] Generating {target_count} premium provider deepclone clips...")
    voice_ids = fetch_premium_tts_voice_ids(api_key=api_key, max_voices=16)
    if not voice_ids:
        print("[!] No provider voices available. Skipping premium provider generation.")
        return existing

    pool = build_multilingual_prompt_pool(transcripts)
    if not pool:
        pool = [(text, "en") for text in SCAM_PHRASES]

    rng = random.Random(seed)
    generated = list(existing)

    for idx in range(len(existing), target_count):
        text, language_hint = rng.choice(pool)
        voice_id = rng.choice(voice_ids)

        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": round(rng.uniform(0.25, 0.75), 2),
                "similarity_boost": round(rng.uniform(0.55, 0.95), 2),
                "style": round(rng.uniform(0.0, 0.4), 2),
                "use_speaker_boost": True,
            },
        }

        out_path = output_dir / f"premium_clone_{idx:04d}.mp3"
        success = False

        for attempt in range(5):
            try:
                response = requests.post(
                    f"{PREMIUM_TTS_BASE_URL}/text-to-speech/{voice_id}",
                    params={"output_format": "mp3_44100_128"},
                    headers={
                        "xi-api-key": api_key,
                        "accept": "audio/mpeg",
                        "content-type": "application/json",
                    },
                    json=payload,
                    timeout=60,
                )
                if response.status_code == 429:
                    wait_seconds = 2.0 + attempt * 2.0
                    print(
                        f"[!] Provider rate limit ({idx + 1}/{target_count}), "
                        f"retry {attempt + 1}/5 in {wait_seconds:.1f}s"
                    )
                    time.sleep(wait_seconds)
                    continue

                response.raise_for_status()
                with open(out_path, "wb") as handle:
                    handle.write(response.content)
                generated.append(out_path)
                success = True
                break
            except Exception as exc:
                wait_seconds = 1.5 + attempt * 1.5
                print(
                    f"[!] Premium provider generation failed ({idx + 1}/{target_count}), "
                    f"retry {attempt + 1}/5: {exc}"
                )
                time.sleep(wait_seconds)

        if (idx + 1) % 10 == 0:
            print(f"  -> generated {idx + 1}/{target_count} premium provider clips")

        if not success:
            continue

    final_files = sorted(output_dir.glob("*.mp3")) + sorted(output_dir.glob("*.wav"))
    return final_files


async def generate_ai_clones(output_dir: Path, transcripts: list[str], target_count: int, seed: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("*.mp3")) + sorted(output_dir.glob("*.wav"))
    if len(existing) >= target_count:
        print(f"[+] Reusing {target_count} existing AI clips")
        return existing[:target_count]

    print(f"[*] Generating {target_count} cloud TTS deepclone clips from internet voices...")

    voice_manager = await VoicesManager.create()
    voice_buckets = build_edge_voice_buckets(voice_manager)

    pool = build_multilingual_prompt_pool(transcripts)
    if not pool:
        pool = [(text, "en") for text in SCAM_PHRASES]

    rng = random.Random(seed)

    generated: list[Path] = []
    for idx in range(target_count):
        text, language_hint = rng.choice(pool)
        voice = choose_edge_voice(language_hint, voice_buckets, rng)
        rate = f"{rng.randint(-20, 20):+d}%"
        pitch = f"{rng.randint(-15, 15):+d}Hz"

        out_path = output_dir / f"internet_tts_{idx:04d}.mp3"
        success = False
        for attempt in range(5):
            try:
                comm = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
                await comm.save(str(out_path))
                generated.append(out_path)
                success = True
                break
            except Exception as exc:
                wait_seconds = 1.5 + attempt * 1.5
                print(f"[!] TTS generation failed ({idx + 1}/{target_count}), retry {attempt + 1}/5: {exc}")
                await asyncio.sleep(wait_seconds)
                voice = choose_edge_voice(language_hint, voice_buckets, rng)

        if not success:
            # Keep going; we'll use whatever got generated successfully.
            continue

        if (idx + 1) % 10 == 0:
            print(f"  -> generated {idx + 1}/{target_count} AI clips")

    final_files = sorted(output_dir.glob("*.mp3")) + sorted(output_dir.glob("*.wav"))
    return final_files


def list_local_human_files(local_human_dir: Path) -> list[Path]:
    if not local_human_dir.exists():
        return []
    files: list[Path] = []
    for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.webm"):
        files.extend(local_human_dir.glob(ext))
    return sorted(files)


def parse_source_ids(raw_value: str, category: str) -> list[str]:
    available = [source.source_id for source in list_sources(category)]
    normalized = (raw_value or "all").strip().lower()
    if normalized == "all":
        return available
    if normalized == "none":
        return []

    chosen = [part.strip() for part in raw_value.split(",") if part.strip()]
    invalid = [source_id for source_id in chosen if source_id not in available]
    if invalid:
        raise ValueError(f"Unknown {category} approved sources: {', '.join(invalid)}")
    return chosen


def dedupe_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def count_registered_files(base_dir: Path, category: str, source_ids: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source_id in source_ids:
        counts[source_id] = len(iter_registered_audio_files(base_dir, category, [source_id]))
    return counts


def write_training_report(
    base_dir: Path,
    args: argparse.Namespace,
    human_source_counts: dict[str, int],
    ai_source_counts: dict[str, int],
    balanced_human_count: int,
    balanced_ai_count: int,
    calibration_path: Path,
) -> Path:
    runs_dir = base_dir / "data" / "training_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = runs_dir / f"internet_training_{timestamp}.json"

    calibration = {}
    if calibration_path.exists():
        with open(calibration_path, "r", encoding="utf-8") as handle:
            calibration = json.load(handle)

    payload = {
        "created_at": datetime.datetime.now().isoformat(),
        "args": vars(args),
        "human_source_counts": human_source_counts,
        "ai_source_counts": ai_source_counts,
        "balanced_training_counts": {
            "human": balanced_human_count,
            "ai": balanced_ai_count,
        },
        "calibration": calibration,
    }

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return report_path


def load_audio(path: Path, sample_rate: int = 16000) -> np.ndarray | None:
    try:
        y, _ = librosa.load(str(path), sr=sample_rate, res_type="kaiser_fast")
        if y is None or len(y) == 0:
            return None
        y = np.asarray(y, dtype=np.float32)
        if np.max(np.abs(y)) > 0:
            y = librosa.util.normalize(y)
        return y
    except Exception:
        return None


def augment_for_channel(audio: np.ndarray, sample_rate: int, rng: random.Random) -> list[np.ndarray]:
    # Focus on realistic telephony variations.
    variants = [audio]

    # Mild time stretch.
    stretch = rng.uniform(0.92, 1.08)
    try:
        variants.append(librosa.effects.time_stretch(audio, rate=stretch))
    except Exception:
        pass

    # Mild pitch shift.
    semitones = rng.uniform(-1.2, 1.2)
    try:
        variants.append(librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones))
    except Exception:
        pass

    # Add low-level background noise.
    noise = np.random.normal(0, 0.0025, size=audio.shape).astype(np.float32)
    variants.append(np.clip(audio + noise, -1.0, 1.0))

    # Simulated phone-band emphasis/de-emphasis.
    try:
        pre = librosa.effects.preemphasis(audio, coef=0.95)
        variants.append(np.asarray(pre, dtype=np.float32))
    except Exception:
        pass

    return variants


def to_feature_chunks(
    path: Path,
    label: float,
    rng: random.Random,
    sample_rate: int = 16000,
    chunk_seconds: float = 1.0,
    hop_seconds: float = 0.5,
    max_chunks_per_file: int = 8,
) -> tuple[list[np.ndarray], list[float]]:
    y = load_audio(path, sample_rate=sample_rate)
    if y is None:
        return [], []

    chunk_len = int(sample_rate * chunk_seconds)
    hop_len = int(sample_rate * hop_seconds)

    feats: list[np.ndarray] = []
    labels: list[float] = []

    variants = augment_for_channel(y, sample_rate, rng)

    for aug_y in variants:
        per_file_chunks = 0
        for start in range(0, len(aug_y), hop_len):
            end = start + chunk_len
            chunk = aug_y[start:end]
            if len(chunk) < chunk_len:
                chunk = np.pad(chunk, (0, chunk_len - len(chunk)))

            # Skip near-silent windows.
            if float(np.max(np.abs(chunk))) < 0.002:
                if end >= len(aug_y):
                    break
                continue

            feat = extract_dual_channel_from_waveform(chunk, sample_rate=sample_rate, max_pad_len=400)
            if feat is not None:
                feats.append(feat)
                labels.append(label)
                per_file_chunks += 1

            if per_file_chunks >= max_chunks_per_file:
                break
            if end >= len(aug_y):
                break

    return feats, labels


def build_feature_matrix(
    human_files: list[Path],
    ai_files: list[Path],
    seed: int,
    max_chunks_per_file: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)

    feats: list[np.ndarray] = []
    labels: list[float] = []

    print(f"[*] Building features from {len(human_files)} human and {len(ai_files)} AI files...")

    for idx, path in enumerate(human_files, start=1):
        f, y = to_feature_chunks(path, 0.0, rng, max_chunks_per_file=max_chunks_per_file)
        feats.extend(f)
        labels.extend(y)
        if idx % 20 == 0:
            print(f"  -> processed human files: {idx}/{len(human_files)}")

    for idx, path in enumerate(ai_files, start=1):
        f, y = to_feature_chunks(path, 1.0, rng, max_chunks_per_file=max_chunks_per_file)
        feats.extend(f)
        labels.extend(y)
        if idx % 20 == 0:
            print(f"  -> processed AI files: {idx}/{len(ai_files)}")

    if not feats:
        raise RuntimeError("No features extracted. Check dataset audio files.")

    x = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32).reshape(-1, 1)

    print(f"[+] Feature matrix built: X={x.shape}, y={y.shape}")
    return x, y


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.30, 0.80, 51):
        preds = (probs >= t).astype(np.float32)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = float(score)
            best_t = float(t)
    return best_t, best_f1


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    model_path: Path,
    calibration_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y.flatten(),
    )

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32).to(device)
    y_val_np = y_val.astype(np.float32).flatten()

    model = AudioCNN(num_classes=1).to(device)
    if model_path.exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("[*] Loaded existing model weights for fine-tuning")
        except Exception as exc:
            print(f"[!] Could not load existing weights ({exc}), training from scratch")

    pos_ratio = float(np.mean(y_train))
    neg_ratio = 1.0 - pos_ratio
    pos_weight_value = (neg_ratio / max(pos_ratio, 1e-6)) if pos_ratio > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_state = None
    best_val_f1 = -1.0
    best_threshold = 0.5

    print("[*] Starting internet-based training...")
    for epoch in range(epochs):
        model.train()

        perm = torch.randperm(x_train_t.size(0))
        x_train_t = x_train_t[perm]
        y_train_t = y_train_t[perm]

        running_loss = 0.0
        for start in range(0, x_train_t.size(0), batch_size):
            end = start + batch_size
            xb = x_train_t[start:end].to(device)
            yb = y_train_t[start:end].to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)

        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(x_val_t)
            probs_val = torch.sigmoid(logits_val).cpu().numpy().flatten()

        threshold, val_f1 = find_best_threshold(y_val_np, probs_val)
        avg_loss = running_loss / max(1, x_train_t.size(0))

        print(
            f"  -> epoch {epoch + 1}/{epochs} | "
            f"train_loss={avg_loss:.4f} | val_f1={val_f1:.4f} | best_threshold={threshold:.2f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = threshold
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_path.exists():
        backup_path = model_path.with_name(f"model_backup_{timestamp}.pth")
        shutil.copy(model_path, backup_path)
        print(f"[*] Backed up previous model to {backup_path.name}")

    torch.save(best_state, model_path)
    print(f"[+] Saved best model to {model_path}")

    calibration = {
        "threshold": round(float(best_threshold), 4),
        "val_f1": round(float(best_val_f1), 4),
        "updated_at": datetime.datetime.now().isoformat(),
    }
    with open(calibration_path, "w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2)
    print(f"[+] Saved calibration to {calibration_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Vacha-Shield using approved human sources, approved AI datasets, and generated clone audio"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_yesno", type=int, default=60)
    parser.add_argument("--max_librispeech", type=int, default=120)
    parser.add_argument("--max_local_human", type=int, default=120)
    parser.add_argument("--target_edge_ai", type=int, default=120)
    parser.add_argument(
        "--target_premium_ai",
        "--target_elevenlabs_ai",
        dest="target_premium_ai",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--provider_model_id",
        "--elevenlabs_model_id",
        dest="provider_model_id",
        type=str,
        default="eleven_multilingual_v2",
    )
    parser.add_argument("--max_chunks_per_file", type=int, default=6)
    parser.add_argument("--librispeech_subset", type=str, default="dev-clean")
    parser.add_argument(
        "--disable_multilingual_prompts",
        action="store_true",
        help="Disable Hindi and Hinglish prompt pools during AI sample generation.",
    )
    parser.add_argument(
        "--approved_human_sources",
        type=str,
        default="all",
        help="Comma-separated approved human source ids, or use all/none.",
    )
    parser.add_argument(
        "--approved_ai_sources",
        type=str,
        default="all",
        help="Comma-separated approved AI source ids, or use all/none.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    load_dotenv(dotenv_path=base_dir / ".env")
    load_dotenv(dotenv_path=base_dir.parent / ".env", override=False)
    provider_api_key = os.getenv("AI_PROVIDER_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
    registry_index_path = write_registry_index(base_dir)

    internet_root = base_dir / "data" / "internet_raw"
    generated_ai_dir = base_dir / "data" / "internet_generated_ai"
    generated_premium_ai_dir = base_dir / "data" / "internet_generated_premium_ai"
    legacy_generated_premium_ai_dir = base_dir / "data" / "internet_generated_legacy_ai"

    model_path = base_dir / "model.pth"
    calibration_path = base_dir / "model_calibration.json"

    approved_human_source_ids = parse_source_ids(args.approved_human_sources, CATEGORY_HUMAN)
    approved_ai_source_ids = parse_source_ids(args.approved_ai_sources, CATEGORY_AI)

    print("=" * 70)
    print("VACHA-SHIELD INTERNET TRAINER")
    print("Approved human sources + approved AI corpora + generated English/Hindi/Hinglish clone audio")
    print(f"[*] Source registry: {registry_index_path}")
    print("=" * 70)

    yesno_files, yesno_texts = download_yesno(internet_root / "yesno")
    yesno_files = yesno_files[: args.max_yesno]

    libri_files: list[Path] = []
    libri_texts: list[str] = []
    if args.max_librispeech > 0:
        libri_files, libri_texts = download_librispeech(
            internet_root / "librispeech",
            subset=args.librispeech_subset,
            max_files=args.max_librispeech,
            seed=args.seed,
        )
    else:
        print("[*] Skipping LibriSpeech download (--max_librispeech=0)")

    local_human_files = list_local_human_files(base_dir / "continuous_learning_dataset" / "human")
    local_human_files = local_human_files[: args.max_local_human]
    approved_human_files = iter_registered_audio_files(base_dir, CATEGORY_HUMAN, approved_human_source_ids)

    human_files = dedupe_paths(yesno_files + libri_files + local_human_files + approved_human_files)
    human_files = [p for p in human_files if p.exists()]

    if len(human_files) < 10:
        raise RuntimeError("Too few human files found. Need at least 10 to train reliably.")

    transcripts = yesno_texts + libri_texts
    if not args.disable_multilingual_prompts:
        transcripts = transcripts + SCAM_PHRASES + HINDI_SCAM_PHRASES + HINGLISH_SCAM_PHRASES + HINDI_SUPPORT_PHRASES + HINGLISH_SUPPORT_PHRASES
    edge_ai_files = asyncio.run(
        generate_ai_clones(
            output_dir=generated_ai_dir,
            transcripts=transcripts,
            target_count=args.target_edge_ai,
            seed=args.seed,
        )
    )

    premium_ai_files = generate_premium_tts_clones(
        output_dir=generated_premium_ai_dir,
        transcripts=transcripts,
        target_count=args.target_premium_ai,
        seed=args.seed,
        api_key=provider_api_key,
        model_id=args.provider_model_id,
    )

    legacy_premium_ai_files = list_local_ai_files(legacy_generated_premium_ai_dir)
    local_ai_files = list_local_ai_files(base_dir / "continuous_learning_dataset" / "ai")
    approved_ai_files = iter_registered_audio_files(base_dir, CATEGORY_AI, approved_ai_source_ids)

    ai_files = dedupe_paths(
        edge_ai_files + premium_ai_files + legacy_premium_ai_files + local_ai_files + approved_ai_files
    )
    ai_files = [p for p in ai_files if p.exists()]
    if len(ai_files) < 10:
        raise RuntimeError("Too few AI files generated. Need at least 10 to train.")

    human_source_counts = {
        "internet_yesno": len(yesno_files),
        f"internet_librispeech_{args.librispeech_subset}": len(libri_files),
        "local_feedback_human": len(local_human_files),
        **count_registered_files(base_dir, CATEGORY_HUMAN, approved_human_source_ids),
    }
    ai_source_counts = {
        "generated_edge_tts": len(edge_ai_files),
        "generated_premium_tts": len(premium_ai_files),
        "legacy_generated_ai": len(legacy_premium_ai_files),
        "local_feedback_ai": len(local_ai_files),
        **count_registered_files(base_dir, CATEGORY_AI, approved_ai_source_ids),
    }

    class_count = min(len(human_files), len(ai_files))
    rng = random.Random(args.seed)
    rng.shuffle(human_files)
    rng.shuffle(ai_files)
    human_files = human_files[:class_count]
    ai_files = ai_files[:class_count]

    print(f"[*] Human source counts: {human_source_counts}")
    print(f"[*] AI source counts: {ai_source_counts}")
    print(f"[*] Using balanced internet dataset: {len(human_files)} human / {len(ai_files)} AI")

    x, y = build_feature_matrix(
        human_files=human_files,
        ai_files=ai_files,
        seed=args.seed,
        max_chunks_per_file=args.max_chunks_per_file,
    )

    train_model(
        x=x,
        y=y,
        model_path=model_path,
        calibration_path=calibration_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    report_path = write_training_report(
        base_dir=base_dir,
        args=args,
        human_source_counts=human_source_counts,
        ai_source_counts=ai_source_counts,
        balanced_human_count=len(human_files),
        balanced_ai_count=len(ai_files),
        calibration_path=calibration_path,
    )
    print(f"[+] Internet training completed successfully. Report: {report_path}")


if __name__ == "__main__":
    main()
