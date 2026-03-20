import os
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
from model import AudioCNN
from feature_extraction import extract_dual_channel_from_waveform
import datetime
import random

import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_DIR = "continuous_learning_dataset"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return int(default)
    return max(0, value)


# Optional fast-training knobs (defaults keep legacy behavior unchanged).
MAX_AI_FILES = _env_int("KB_MAX_AI_FILES", 0)
MAX_HUMAN_FILES = _env_int("KB_MAX_HUMAN_FILES", 0)
MAX_SLICES_PER_FILE = _env_int("KB_MAX_SLICES_PER_FILE", 0)
TRAIN_EPOCHS = _env_int("KB_EPOCHS", 30)
TRAIN_BATCH_SIZE = _env_int("KB_BATCH_SIZE", 32)
RANDOM_SEED = _env_int("KB_RANDOM_SEED", 42)


# =============================================================================
# UPGRADE 2: FOCAL LOSS
# =============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification (Lin et al., 2017).

    WHY THIS BEATS BCEWithLogitsLoss FOR THIS DEMO:
    ─────────────────────────────────────────────────
    Standard BCE penalizes every wrong prediction equally. But for live demo
    detection of a fresh high-fidelity clone, two failure modes are NOT equal:
      - False Negative (clone called "Human") → demo fails catastrophically
      - False Positive (human called "AI")    → mildly awkward, recoverable

    Focal Loss solves this two ways:
      1. gamma (focusing): Down-weights "easy" correct predictions so the model
         stops wasting gradient budget on audio it already knows. All effort goes
         toward hard uncertain examples — exactly what a new voice clone is.
      2. alpha (class weight): We set alpha=0.75 to give the AI/spoof class 75%
         of the gradient attention, counteracting any human-heavy data imbalance
         in the continuous learning dataset.

    Args:
        alpha (float): Weight for positive (AI/spoof) class. 0.75 means the model
                       is penalized 3x harder for missing a deepfake vs a human.
        gamma (float): Focusing exponent. 2.0 is the standard recommended value.
                       At gamma=2, a prediction with 70% confidence receives only
                       9% of the loss weight it would under standard BCE.
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Standard BCE loss (unreduced so we can apply focal weighting per-sample)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        # p_t: probability of the CORRECT class for each sample
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)

        # alpha_t: up-weight the AI/spoof class (targets==1), down-weight human (targets==0)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Focal weight: (1 - p_t)^gamma — near 0 for easy correct preds, near 1 for hard ones
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        return (focal_weight * bce).mean()


# =============================================================================
# AUDIO AUGMENTATION
# =============================================================================
def augment_audio(y, sr):
    """Applies random acoustic perturbations to multiply dataset size and robustness."""
    augmented_versions = [y]  # Original always included

    # Pitch shift (simulate different voices)
    augmented_versions.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2.0))
    augmented_versions.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2.0))

    # Time stretch (simulate fast/slow speech)
    augmented_versions.append(librosa.effects.time_stretch(y, rate=1.2))
    augmented_versions.append(librosa.effects.time_stretch(y, rate=0.8))

    # Background noise (simulate bad mic / cellular network)
    noise = np.random.normal(0, 0.005, y.shape)
    augmented_versions.append(y + noise)

    # High-frequency muffling (simulate cellular codec compression)
    augmented_versions.append(librosa.effects.preemphasis(y, coef=0.5))

    return augmented_versions


# =============================================================================
# FEATURE EXTRACTION (uses upgraded extract_features — PCEN + dual channel)
# =============================================================================
def load_and_preprocess(filepath):
    """
    Loads an audio file, applies 7 augmentation variants, and converts each
    1-second chunk to a (2, 40, 400) dual-channel feature tensor.
    """
    try:
        # Transcode .webm browser recordings to wav on the fly
        if filepath.endswith('.webm'):
            import imageio_ffmpeg
            import subprocess
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            wav_path = filepath + ".wav"
            subprocess.run(
                [ffmpeg_exe, '-y', '-i', filepath, '-ar', '16000', '-ac', '1', wav_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            process_path = wav_path
        else:
            process_path = filepath

        y, sr = librosa.load(process_path, sr=16000)
        y = librosa.util.normalize(y)

        if filepath.endswith('.webm') and os.path.exists(process_path):
            os.remove(process_path)

        chunk_samples = 16000
        features_list = []

        for aug_y in augment_audio(y, sr):
            for i in range(0, len(aug_y), chunk_samples):
                chunk = aug_y[i:i + chunk_samples]
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

                feat = extract_dual_channel_from_waveform(chunk, sample_rate=16000, max_pad_len=400)
                if feat is not None:
                    features_list.append(feat)
                    if MAX_SLICES_PER_FILE > 0 and len(features_list) >= MAX_SLICES_PER_FILE:
                        break
            if MAX_SLICES_PER_FILE > 0 and len(features_list) >= MAX_SLICES_PER_FILE:
                break

        if not features_list:
            return None

        # Shape: (N_chunks, 2, 40, 400)
        return torch.tensor(np.array(features_list), dtype=torch.float32).to(DEVICE)

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


# =============================================================================
# DATASET BUILDER
# =============================================================================
def build_dataset_tensors():
    all_features = []
    all_labels = []
    human_count, ai_count = 0, 0

    categories = {"human": 0.0, "ai": 1.0}
    rng = random.Random(RANDOM_SEED)

    for category, label_val in categories.items():
        folder_path = os.path.join(DATASET_DIR, category)
        if not os.path.exists(folder_path):
            continue

        files = [
            file for file in os.listdir(folder_path)
            if file.endswith(('.wav', '.webm', '.mp3', '.ogg', '.flac'))
        ]

        limit = MAX_HUMAN_FILES if category == "human" else MAX_AI_FILES
        if limit > 0 and len(files) > limit:
            rng.shuffle(files)
            files = files[:limit]

        print(f"[*] Scanning {len(files)} files in Knowledge Base: /{category}/ ...")

        for file in files:
            file_path = os.path.join(folder_path, file)
            tensors = load_and_preprocess(file_path)

            if tensors is not None:
                all_features.append(tensors)
                labels = torch.full((tensors.shape[0], 1), label_val, dtype=torch.float32).to(DEVICE)
                all_labels.append(labels)

                if category == "human":
                    human_count += tensors.shape[0]
                else:
                    ai_count += tensors.shape[0]

    if not all_features:
        return None, None, 0, 0

    X = torch.cat(all_features, dim=0)
    Y = torch.cat(all_labels, dim=0)

    # Shuffle
    indices = torch.randperm(X.shape[0])
    return X[indices], Y[indices], human_count, ai_count


# =============================================================================
# TRAINING LOOP
# =============================================================================
def train_continuous_learning():
    print("=" * 60)
    print(
        f" Config: epochs={TRAIN_EPOCHS}, batch={TRAIN_BATCH_SIZE}, "
        f"max_ai_files={MAX_AI_FILES or 'all'}, max_human_files={MAX_HUMAN_FILES or 'all'}, "
        f"max_slices_per_file={MAX_SLICES_PER_FILE or 'all'}"
    )
    print(" VACHA-SHIELD: ADVANCED DATA AUGMENTATION TRAINER")
    print(" Upgrades: PCEN · Delta Channel · Focal Loss")
    print("=" * 60)

    if not os.path.exists(DATASET_DIR):
        print("\n[!] The continuous learning dataset folder is empty.")
        print("    Use the Web UI 'Feedback' buttons to submit at least 1 Human and 1 AI audio.")
        return

    X, Y, h_count, a_count = build_dataset_tensors()

    if X is None or h_count == 0 or a_count == 0:
        print("\n[!] CRITICAL ERROR: Imbalanced Knowledge Base!")
        print(f"    Human slices: {h_count} | AI slices: {a_count}")
        print("    Submit BOTH human and AI clips via the Web UI and retry.\n")
        return

    print(f"\n[+] DATA AUGMENTATION COMPLETE: {X.shape[0]} unique training matrices generated")
    print(f"    Human Training Slices:    {h_count}")
    print(f"    Deepfake Training Slices: {a_count}")
    print(f"    Feature shape per sample: {tuple(X.shape[1:])}")  # Should be (2, 40, 400)

    # Load model (architecture now expects 2-channel input)
    model = AudioCNN(num_classes=1).to(DEVICE)
    if os.path.exists("model.pth"):
        try:
            model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
            print("\n[*] Loaded existing base model. Fine-tuning with upgraded pipeline...")
        except RuntimeError:
            print("\n[!] Existing model.pth has incompatible architecture (old 1-channel weights).")
            print("[*] Training fresh 2-channel model from scratch.")
    else:
        print("\n[*] No existing model found — training from scratch.")

    # Calculate dynamic alpha to balance classes based on training data availability
    total_count = h_count + a_count
    # Alpha is the weight for the positive (AI) class. 
    # If AI is the majority, alpha should be low so Human predictions get higher gradient weight.
    dynamic_alpha = h_count / total_count
    
    # UPGRADE 2: Focal Loss replaces BCEWithLogitsLoss
    # alpha=dynamic_alpha automatically balances the Human vs AI dataset ratio
    # gamma=2.0:  Standard focusing parameter — ignores easy predictions, hammers hard ones
    criterion = FocalLoss(alpha=dynamic_alpha, gamma=2.0)
    print(f"[*] Loss function: FocalLoss(alpha={dynamic_alpha:.4f}, gamma=2.0)")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = max(1, TRAIN_EPOCHS)
    batch_size = max(1, TRAIN_BATCH_SIZE)

    # === BALANCE THE DATASET ===
    # With 8762 AI vs 1087 human samples, the model learns to just always predict AI.
    # We fix this by oversampling the minority human class to create a 50/50 mix.
    human_mask = (Y == 0).squeeze()
    ai_mask = (Y == 1).squeeze()
    X_human, Y_human = X[human_mask], Y[human_mask]
    X_ai, Y_ai = X[ai_mask], Y[ai_mask]
    target_size = max(len(X_human), len(X_ai))
    if len(X_human) < target_size:
        repeat = (target_size // len(X_human)) + 1
        X_human = X_human.repeat(repeat, 1, 1, 1)[:target_size]
        Y_human = Y_human.repeat(repeat, 1)[:target_size]
    if len(X_ai) < target_size:
        repeat = (target_size // len(X_ai)) + 1
        X_ai = X_ai.repeat(repeat, 1, 1, 1)[:target_size]
        Y_ai = Y_ai.repeat(repeat, 1)[:target_size]
    X = torch.cat([X_human, X_ai], dim=0)
    Y = torch.cat([Y_human, Y_ai], dim=0)
    indices = torch.randperm(X.shape[0])
    X, Y = X[indices], Y[indices]
    print(f"[*] Balanced dataset: {target_size} human + {target_size} AI = {X.shape[0]} total samples")

    print(f"[*] Training for {epochs} epochs on {DEVICE}...\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i in range(0, len(X), batch_size):
            batch_X = X[i:i + batch_size]
            batch_Y = Y[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(X)
        if (epoch + 1) % 5 == 0:
            print(f"  -> Epoch {epoch + 1}/{epochs} | Focal Loss: {epoch_loss:.4f}")

    # Save with backup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"model_backup_{timestamp}.pth"

    if os.path.exists("model.pth"):
        os.rename("model.pth", backup_name)
        print(f"\n[*] Backed up old weights to '{backup_name}'")

    torch.save(model.state_dict(), "model.pth")
    print("\n[+] TRAINING COMPLETE!")
    print("    The model now uses PCEN + Delta features with Focal Loss weighting.")
    print("    Restart the Flask app (app.py) to load the upgraded neural network.")


if __name__ == "__main__":
    train_continuous_learning()
