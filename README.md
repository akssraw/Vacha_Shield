# Vacha-Shield

Vacha-Shield is an AI deepfake voice detection system for web and mobile-assisted call monitoring.

## Quick Start

1. Install dependencies:
   `pip install -r requirements.txt`
2. Run backend:
   `python app.py`
3. Open:
   `http://127.0.0.1:5000`

## Core Files

- `app.py` - Flask API + UI serving
- `deepfake_detector.py` - Hybrid inference and scoring logic
- `feature_extraction.py` - PCEN + Delta feature pipeline
- `model.py` - AudioCNN architecture
- `train_knowledge_base.py` - Continuous learning retraining

## Architecture

See `docs/architecture.png`.

## Notes

- Include `model.pth` and `model_calibration.json` for inference.
- Install `pip install -r requirements-optional.txt` only if you need live microphone features like `call_monitor.py`.
- Do not commit logs, tunnel binaries, or backup checkpoints.

## Hackathon Cloud Demo

Fastest path: deploy this repo to Railway or Render using the included `Dockerfile`.

1. Push the repo to GitHub with `model.pth`, `model_calibration.json`, and the built `lovable-project-*/dist` folder included.
2. Create a new Railway or Render web service from that repo.
3. No custom start command is needed because the container boots with Gunicorn.
4. After deploy, open `/health` to confirm the model loaded.

Optional environment variables:
- `GROQ_API_KEY`
- `GROQ_MODEL`
- `SEMANTIC_OVERRIDE_ENABLED`
- `DEFAULT_ANALYSIS_PROFILE`
- `ALERT_MIN_THRESHOLD`


## Curated Training

- List approved data sources: `python sync_approved_sources.py --list`
- Sync automatic human corpora into the registry: `python sync_approved_sources.py --sync human_yesno human_librispeech_dev_clean --limit 50`
- Register manually downloaded Hindi human corpora: `python sync_approved_sources.py --register human_common_voice_hi --from-dir "C:\path\to\commonvoice_hi" --limit 300`
- Register manually downloaded AI datasets like WaveFake or ASVspoof: `python sync_approved_sources.py --register ai_wavefake --from-dir "C:\path\to\dataset" --limit 200`
- Train with approved sources plus generated English/Hindi/Hinglish clone audio: `python train_internet_model.py --approved_human_sources all --approved_ai_sources all`

See `docs/curated_sources.md` for the guarded data-ingestion flow.
