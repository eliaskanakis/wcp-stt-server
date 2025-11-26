FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    TRANSFORMERS_OFFLINE=1 \
    VOSK_MODEL_PATH=/models/vosk \
    SUMMARIZER_MODEL=/models/distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff \
    SUMMARY_INTERVAL_CHARS=300 \
    STT_SAMPLE_RATE=16000 \
    PORT=8080

# System deps (ffmpeg for opus -> PCM decode)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy baked models into the image.
# Expect:
#  - assets/vosk-model-small-en-us-0.15 (Vosk model dir)
#  - assets/distilbart-cnn-12-6 (HF summarizer model cache dir)
COPY assets/vosk-model-small-en-us-0.15 /models/vosk
COPY assets/distilbart-cnn-12-6 /models/distilbart-cnn-12-6

# Copy application code
COPY . .

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
