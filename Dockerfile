FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV LOVABLE_DIST_DIR=lovable-dist
ENV HOME=/home/appuser
ENV PATH=/home/appuser/.local/bin:$PATH

RUN useradd -m -u 1000 appuser

WORKDIR $HOME/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Core Python files
COPY --chown=appuser:appuser app.py deepfake_detector.py feature_extraction.py model.py ./
COPY --chown=appuser:appuser approved_sources.py dataset_loader.py ./

# Model weights
COPY --chown=appuser:appuser model.pth model_calibration.json ./

# Create necessary folders
RUN mkdir -p ./templates ./static ./lovable-dist

# Copy static files
COPY --chown=appuser:appuser style.css ./static/
COPY --chown=appuser:appuser script.js ./static/
COPY --chown=appuser:appuser mobile.css ./static/
COPY --chown=appuser:appuser mobile.js ./static/

# Copy template files
COPY --chown=appuser:appuser index.html ./templates/
COPY --chown=appuser:appuser mobile.html ./templates/

RUN mkdir -p temp_uploads flagged_calls continuous_learning_dataset \
    && chown -R appuser:appuser temp_uploads flagged_calls continuous_learning_dataset

USER appuser
EXPOSE 7860

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 8 --timeout 180 app:app"]
