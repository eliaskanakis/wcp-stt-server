import os
import json
import base64
import tempfile
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
import vosk
from transformers import pipeline

DEFAULT_SAMPLE_RATE = int(os.environ.get("STT_SAMPLE_RATE", "16000"))
DEFAULT_CHANNELS = int(os.environ.get("STT_CHANNELS", "1"))
SUMMARY_INTERVAL_CHARS = int(os.environ.get("SUMMARY_INTERVAL_CHARS", "300"))
VOSK_MODEL_PATH = os.environ.get("VOSK_MODEL_PATH")
SUMMARIZER_MODEL = os.environ.get("SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")
vosk.SetLogLevel(int(os.environ.get("VOSK_LOG_LEVEL", "-1")))

app = FastAPI()

# In-memory state for demo (not persistent, just per-process)
# callId -> simple counters / buffers
active_calls: Dict[str, Dict[str, Any]] = {}
_vosk_model: Optional[vosk.Model] = None
_summarizer: Optional[Callable] = None


def cleanup_call(call_id: str):
    """Close any open file handle and drop the call state."""
    call_state = active_calls.pop(call_id, None)
    if not call_state:
        return None

    file_obj = call_state.get("file")
    if file_obj and not file_obj.closed:
        try:
            file_obj.flush()
        except Exception:
            pass
        try:
            file_obj.close()
        except Exception:
            pass

    return call_state


def cleanup_calls_for_client(ws_client_id: str):
    """Close files for all calls initiated by this client."""
    to_cleanup = [
        call_id for call_id, state in active_calls.items()
        if state.get("wsClientId") == ws_client_id
    ]
    for call_id in to_cleanup:
        cleanup_call(call_id)


def get_vosk_model() -> vosk.Model:
    """Lazy-load the Vosk model from VOSK_MODEL_PATH."""
    global _vosk_model
    if _vosk_model is not None:
        return _vosk_model

    if not VOSK_MODEL_PATH:
        raise RuntimeError("VOSK_MODEL_PATH is not set")

    model_path = Path(VOSK_MODEL_PATH)
    if not model_path.exists():
        raise RuntimeError(f"VOSK_MODEL_PATH does not exist: {model_path}")

    _vosk_model = vosk.Model(str(model_path))
    return _vosk_model


def make_recognizer(sample_rate: int) -> vosk.KaldiRecognizer:
    model = get_vosk_model()
    return vosk.KaldiRecognizer(model, sample_rate)


def get_summarizer():
    """Lazy-load the summarization pipeline."""
    global _summarizer
    if _summarizer is not None:
        return _summarizer
    _summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, device=-1)
    return _summarizer


async def generate_summary(text: str) -> str:
    """Run summarization in a thread to avoid blocking the event loop."""
    if not text:
        return ""
    summarizer = get_summarizer()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: summarizer(
            text,
            max_length=120,
            min_length=30,
            do_sample=False,
        )[0]["summary_text"].strip()
    )
    return result


def decode_webm_file_to_pcm(webm_path: Path, sample_rate: int, channels: int = 1) -> bytes:
    """Decode a whole webm/opus file to raw PCM (s16le) using ffmpeg."""
    ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-i",
        str(webm_path),
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {result.stderr.decode(errors='ignore').strip()}")
    if not result.stdout:
        raise RuntimeError("ffmpeg decode produced no audio")
    return result.stdout


def decode_webm_chunk_to_pcm(webm_bytes: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Decode a single webm/opus chunk to raw PCM (s16le) using ffmpeg."""
    ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-f",
        "webm",
        "-analyzeduration",
        "0",
        "-probesize",
        "32k",
        "-i",
        "pipe:0",
        "-map",
        "0:a:0",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    result = subprocess.run(cmd, input=webm_bytes, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {result.stderr.decode(errors='ignore').strip()}")
    if not result.stdout:
        raise RuntimeError("ffmpeg decode produced no audio")
    return result.stdout


def try_decode_audio(call_state: Dict[str, Any], audio_bytes: bytes, sample_rate: int) -> bytes:
    """
    Attempt to decode a chunk using the first chunk as a header; if that fails,
    fall back to decoding the whole temp file.
    """
    header_bytes = call_state.get("webm_header")
    if header_bytes is None:
        call_state["webm_header"] = audio_bytes
        header_bytes = audio_bytes

    # First attempt: header + current chunk (or just header for the first chunk)
    to_decode = header_bytes if call_state.get("chunks", 0) == 1 else header_bytes + audio_bytes
    try:
        return decode_webm_chunk_to_pcm(to_decode, sample_rate, channels=call_state.get("channels", 1))
    except Exception as err:
        # Fallback: try decoding the accumulated temp file
        temp_path = call_state.get("temp_path")
        if temp_path:
            try:
                return decode_webm_file_to_pcm(Path(temp_path), sample_rate, channels=call_state.get("channels", 1))
            except Exception as err2:
                raise RuntimeError(
                    f"chunk decode failed: {err}; file decode failed: {err2}"
                )
        raise


@app.get("/", response_class=PlainTextResponse)
async def root():
    return "STT WebSocket service is running"


@app.websocket("/ws-stt")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint for STT.
    Expects JSON text messages with:
      - stt-start
      - stt-audio-chunk (base64-encoded audio bytes)
      - stt-stop

    Audio is expected as webm/opus chunks (MediaRecorder). Each chunk is decoded
    with ffmpeg and fed to a Vosk recognizer for partial/final transcripts.
    """
    await ws.accept()
    client_id = f"{id(ws)}"
    print(f"[WS-STT] Client connected: {client_id}")

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[WS-STT] Invalid JSON from client {client_id}") 
                await ws.send_text(json.dumps({
                    "type": "stt-error",
                    "error": "Invalid JSON"
                }))
                continue

            msg_type = msg.get("type")

            # --- Handle stt-start ---
            if msg_type == "stt-start":
                call_id = msg.get("callId")
                channel_id = msg.get("channelId")
                user_id = msg.get("userId")
                language = msg.get("language", "en")
                sample_rate = int(msg.get("sampleRate", DEFAULT_SAMPLE_RATE))
                channels = int(msg.get("channels", DEFAULT_CHANNELS))

                if not call_id:
                    await ws.send_text(json.dumps({
                        "type": "stt-error",
                        "error": "Missing callId in stt-start"
                    }))
                    continue

                try:
                    recognizer = make_recognizer(sample_rate)
                except Exception as e:
                    print(f"[WS-STT] Error loading STT model: {e}")
                    await ws.send_text(json.dumps({
                        "type": "stt-error",
                        "error": f"Unable to load STT model: {e}"
                    }))
                    continue

                temp_file = tempfile.NamedTemporaryFile(
                    prefix=f"stt_{call_id}_",
                    suffix=".webm",
                    delete=False
                )

                active_calls[call_id] = {
                    "channelId": channel_id,
                    "userId": user_id,
                    "language": language,
                    "chunks": 0,
                    "bytes_written": 0,
                    "file": temp_file,
                    "temp_path": temp_file.name,
                    "wsClientId": client_id,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "recognizer": recognizer,
                    "transcript": "",
                    "next_summary_len": SUMMARY_INTERVAL_CHARS,
                }

                print(
                    f"[WS-STT] stt-start callId={call_id}, user={user_id}, channel={channel_id}, "
                    f"temp_file={temp_file.name}, format={channels}ch/{sample_rate}Hz (webm/opus)"
                )
                await ws.send_text(json.dumps({
                    "type": "stt-started",
                    "callId": call_id,
                }))
                continue

            # --- Handle stt-audio-chunk ---
            if msg_type == "stt-audio-chunk":
                call_id = msg.get("callId")
                seq = msg.get("seq")
                data_b64 = msg.get("data")

                if not call_id or call_id not in active_calls:
                    await ws.send_text(json.dumps({
                        "type": "stt-error",
                        "error": "Unknown or missing callId for stt-audio-chunk"
                    }))
                    continue

                call_state = active_calls[call_id]
                file_obj = call_state.get("file")
                if not file_obj or file_obj.closed:
                    await ws.send_text(json.dumps({
                        "type": "stt-error",
                        "error": "Audio file not available for writing"
                    }))
                    continue

                try:
                    audio_bytes = base64.b64decode(data_b64, validate=True)
                except Exception:
                    print(f"[WS-STT] Invalid base64 in stt-audio-chunk callId={call_id}")   
                    await ws.send_text(json.dumps({
                        "type": "stt-error",
                        "error": "Invalid base64 in stt-audio-chunk"
                    }))
                    continue

                file_obj.write(audio_bytes)
                file_obj.flush()
                call_state["bytes_written"] += len(audio_bytes)

                call_state["chunks"] += 1
                chunks = call_state["chunks"]
                sample_rate = call_state.get("sample_rate", DEFAULT_SAMPLE_RATE)

                recognizer = call_state.get("recognizer")
                if not recognizer:
                    await ws.send_text(json.dumps({
                        "type": "stt-error",
                        "error": "Recognizer not initialized"
                    }))
                    continue

                try:
                    pcm_audio = try_decode_audio(call_state, audio_bytes, sample_rate)
                except Exception as err:
                    print(f"[WS-STT] Decode error in stt-audio-chunk callId={call_id}, seq={seq}, chunk={chunks}: {err!r}")
                    await ws.send_text(json.dumps({
                        "type": "stt-error",
                        "error": f"Decode error: {err}"
                    }))
                    continue

                if recognizer.AcceptWaveform(pcm_audio):
                    result_json = json.loads(recognizer.Result() or "{}")
                    final_text = result_json.get("text", "")
                    if final_text:
                        transcript = call_state.get("transcript", "")
                        updated_transcript = (transcript + " " + final_text).strip()
                        call_state["transcript"] = updated_transcript
                        call_state["next_summary_len"] = call_state.get("next_summary_len", SUMMARY_INTERVAL_CHARS)
                        await ws.send_text(json.dumps({
                            "type": "stt-final",
                            "callId": call_id,
                            "text": final_text,
                            "isFinal": True,
                            "result": result_json,
                            "seq": seq,
                            "chunk": chunks,
                        }))

                        # Emit periodic summaries when transcript crosses threshold
                        next_len = call_state.get("next_summary_len", SUMMARY_INTERVAL_CHARS)
                        if len(updated_transcript) >= next_len:
                            summary_text = await generate_summary(updated_transcript)
                            await ws.send_text(json.dumps({
                                "type": "stt-summary",
                                "callId": call_id,
                                "text": summary_text,
                                "isFinal": False,
                                "seq": seq,
                                "chunk": chunks,
                                "sourceLength": len(updated_transcript),
                            }))
                            call_state["next_summary_len"] = next_len + SUMMARY_INTERVAL_CHARS
                else:
                    partial_json = json.loads(recognizer.PartialResult() or "{}")
                    partial_text = partial_json.get("partial", "")
                    if partial_text:
                        await ws.send_text(json.dumps({
                            "type": "stt-partial",
                            "callId": call_id,
                            "text": partial_text,
                            "isFinal": False,
                            "result": partial_json,
                            "seq": seq,
                            "chunk": chunks,
                        }))
                continue

            # --- Handle stt-stop ---
            if msg_type == "stt-stop":
                call_id = msg.get("callId")
                if not call_id or call_id not in active_calls:
                    await ws.send_text(json.dumps({
                        "type": "stt-error",
                        "error": "Unknown or missing callId for stt-stop"
                    }))
                    continue

                final_state = cleanup_call(call_id)
                bytes_written = (final_state or {}).get("bytes_written", 0)

                valid_audio = bytes_written > 0

                if final_state:
                    print(
                        f"[WS-STT] stt-stop callId={call_id}, bytes={bytes_written}"
                    )

                response = {
                    "type": "stt-stopped",
                    "callId": call_id,
                    "validAudio": valid_audio,
                }
                if not valid_audio:
                    response["error"] = "No audio received; nothing saved."

                await ws.send_text(json.dumps(response))
                continue

            # --- Unknown message type ---
            await ws.send_text(json.dumps({
                "type": "stt-error",
                "error": f"Unknown message type: {msg_type}"
            }))

    except WebSocketDisconnect:
        print(f"[WS-STT] Client disconnected: {client_id}")
        cleanup_calls_for_client(client_id)
    except Exception as e:
        print(f"[WS-STT] Error in websocket_endpoint: {e}")
        cleanup_calls_for_client(client_id)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
