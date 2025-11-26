# WCP STT Server

Lightweight FastAPI WebSocket service for streaming speech-to-text with Vosk and periodic summaries (HF summarizer). Pairs with the WCP frontend and chat server.

- Frontend repo: https://github.com/eliaskanakis/wcp-frontend  
- Chat server repo: https://github.com/eliaskanakis/wcp-chat-server  
- Live frontend: https://wrh-coord-platform.web.app/  
- Demo video: https://youtu.be/VIDEO_PLACEHOLDER

## Run locally
1) Install Python 3.11, create and activate `.venv`.
2) `pip install -r requirements.txt`
3) Set env vars (example):
   - `VOSK_MODEL_PATH=/path/to/vosk-model-small-en-us-0.15`
   - `SUMMARIZER_MODEL=sshleifer/distilbart-cnn-12-6` (or a local snapshot path)
   - `FFMPEG_BIN=ffmpeg` if not on PATH
4) `python -m uvicorn main:app --reload --port 8080`

WebSocket: `ws://localhost:8080/ws-stt`  
Messages: `stt-start`, `stt-audio-chunk` (base64 webm/opus), `stt-stop`. Emits partials/finals and `stt-summary` every `SUMMARY_INTERVAL_CHARS` (default 300).

## Deploy (Cloud Run quick start)
Build with baked models (see Dockerfile), then:
```
gcloud run deploy stt-server \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --env-vars-file .env-yaml \
  --memory=2Gi --cpu=2 --concurrency=1 --min-instances=0
```
