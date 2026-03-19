import librosa
import numpy as np
import warnings

# Suppress librosa warnings about PySoundFile falling back to audioread
warnings.filterwarnings("ignore", category=UserWarning)


def extract_dual_channel_from_waveform(
    audio: np.ndarray,
    sample_rate: int = 16000,
    max_pad_len: int = 400,
) -> np.ndarray | None:
    """
    Extracts Vacha-Shield production features from an in-memory waveform.
    Returns shape (2, 40, 400):
      - channel 0: PCEN Mel spectrogram
      - channel 1: Delta(PCEN)
    """
    try:
        if audio is None or len(audio) == 0:
            return None

        audio = np.asarray(audio, dtype=np.float32)

        # 10ms frame hop @ 16kHz, tuned for speech anti-spoofing.
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=40,
            n_fft=1024,
            hop_length=160,
            fmax=8000,
        )
        pcen = librosa.pcen(
            mel * (2**31),
            sr=sample_rate,
            hop_length=160,
            gain=0.8,
            bias=10,
            power=0.25,
            time_constant=0.4,
            eps=1e-6,
        ).astype(np.float32)

        delta = librosa.feature.delta(pcen, width=9).astype(np.float32)
        features = np.stack([pcen, delta], axis=0)

        # Fixed temporal size for model compatibility.
        time_frames = features.shape[2]
        if time_frames < max_pad_len:
            features = np.pad(
                features,
                pad_width=((0, 0), (0, 0), (0, max_pad_len - time_frames)),
                mode="constant",
            )
        else:
            features = features[:, :, :max_pad_len]

        return features.astype(np.float32)
    except Exception as e:
        print(f"Error extracting dual-channel features from waveform: {e}")
        return None


def extract_features(file_path, max_pad_len=400, feature_type="spectrogram", target_sr=16000):
    """
    Extract acoustic features from a file.
    Returns (2, 40, 400) for spectrogram and (1, 40, 400) for MFCC.
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=target_sr, res_type="kaiser_fast")

        if feature_type == "mfcc":
            features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            pad_width = max_pad_len - features.shape[1]
            if pad_width > 0:
                features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode="constant")
            else:
                features = features[:, :max_pad_len]
            return features[np.newaxis, :, :].astype(np.float32)

        if feature_type == "spectrogram":
            return extract_dual_channel_from_waveform(
                audio=audio,
                sample_rate=sample_rate,
                max_pad_len=max_pad_len,
            )

        raise ValueError("feature_type must be 'mfcc' or 'spectrogram'")
    except Exception as e:
        print(f"Error extracting {feature_type} from {file_path}: {e}")
        return None


if __name__ == "__main__":
    print("Feature Extraction Engine Ready.")
    print("Spectrogram output shape: (2, 40, 400)")
    print("MFCC output shape:        (1, 40, 400)")
