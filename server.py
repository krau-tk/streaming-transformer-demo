import logging
from datetime import date as _date
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


import torchaudio

from model_loader import ASREngine, EXP_NAME
from pseudo_streaming_session import (
    PseudoStreamingASRSession,
    recognize_pseudo_streaming_waveform,
)
from online_session import (
    DEFAULT_DECODER_STEP_CHUNKS,
    SAMPLES_PER_CHUNK,
    SegmentedOnlineASRSession,
    recognize_online_streaming_waveform,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

engine: ASREngine = None
STATIC_DIR = Path(__file__).parent / "static"
PSEUDO_NUM_DECODE_CHUNKS = 4
PSEUDO_DECODE_INTERVAL_SAMPLES = PSEUDO_NUM_DECODE_CHUNKS * SAMPLES_PER_CHUNK


LOG_DIR = Path(__file__).parent / "log"

_SESSION_LOGGERS = ["online_session", "pseudo_streaming_session", "__main__"]


def _make_log_handler(mode: str) -> logging.FileHandler:
    LOG_DIR.mkdir(exist_ok=True)
    filename = f"{EXP_NAME}_{mode}_{_date.today().strftime('%Y%m%d')}.txt"
    handler = logging.FileHandler(LOG_DIR / filename)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    handler.setLevel(logging.INFO)
    return handler


def _attach_handler(handler: logging.FileHandler):
    for name in _SESSION_LOGGERS:
        logging.getLogger(name).addHandler(handler)


def _detach_handler(handler: logging.FileHandler):
    for name in _SESSION_LOGGERS:
        logging.getLogger(name).removeHandler(handler)
    handler.close()


def normalize_mode(mode: str) -> str:
    mode = (mode or "pseudo").strip().lower()
    if mode not in {"pseudo", "online"}:
        raise ValueError(f"Unsupported mode: {mode}")
    return mode


@asynccontextmanager
async def lifespan(app):
    global engine
    log.info("Loading ASR model...")
    engine = ASREngine(device="cuda:0")
    log.info("ASR model ready.")
    yield


app = FastAPI(title="Streaming ASR Demo", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.websocket("/ws/asr")
async def websocket_asr(
    ws: WebSocket,
    mode: str = Query("pseudo"),
    decoder_step_chunks: int = Query(
        DEFAULT_DECODER_STEP_CHUNKS,
        ge=1,
        le=16,
    ),
):
    await ws.accept()
    try:
        mode = normalize_mode(mode)
    except ValueError as e:
        await ws.send_json({"error": str(e), "is_final": True})
        await ws.close(code=1008)
        return

    if mode == "online":
        session = SegmentedOnlineASRSession(
            model=engine.model,
            sp=engine.sp,
            params=engine.params,
            device=engine.device,
            decoder_step_chunks=decoder_step_chunks,
        )
    else:
        session = PseudoStreamingASRSession(
            model=engine.model,
            sp=engine.sp,
            params=engine.params,
            device=engine.device,
            decode_interval_samples=PSEUDO_DECODE_INTERVAL_SAMPLES,
            min_decode_samples=PSEUDO_DECODE_INTERVAL_SAMPLES,
            soft_segment_samples=PSEUDO_DECODE_INTERVAL_SAMPLES * 8,
            hard_segment_samples=PSEUDO_DECODE_INTERVAL_SAMPLES * 14,
            num_decode_chunks=PSEUDO_NUM_DECODE_CHUNKS,
        )
    log.info("New %s ASR session started", mode)

    fh = _make_log_handler(mode)
    _attach_handler(fh)
    try:
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                try:
                    result = session.feed_audio(message["bytes"])
                except Exception as e:
                    log.error("feed_audio error: %s", e, exc_info=True)
                    result = None
                if result:
                    await ws.send_json(result)

            elif "text" in message and message["text"] == "finalize":
                result = session.finalize()
                await ws.send_json(result)
                break

    except WebSocketDisconnect:
        log.info("ASR session disconnected")
    except RuntimeError as e:
        if "disconnect" in str(e).lower():
            log.info("ASR session disconnected (runtime)")
        else:
            log.error("ASR session error: %s", e)
    except Exception as e:
        log.error("ASR session error: %s", e, exc_info=True)
    finally:
        _detach_handler(fh)


@app.post("/api/recognize")
async def recognize_file(
    file: UploadFile = File(...),
    mode: str = Query("pseudo"),
    num_decode_chunks: int = Query(4, ge=1, le=16),
):
    """Offline recognition through the selected streaming path."""
    import io

    try:
        mode = normalize_mode(mode)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    log.info(
        "Offline recognize(%s): %s, num_decode_chunks=%d",
        mode,
        file.filename,
        num_decode_chunks,
    )
    content = await file.read()

    # Load audio
    buf = io.BytesIO(content)
    try:
        waveform, sr = torchaudio.load(buf)
    except Exception as e:
        return JSONResponse({"error": f"Cannot load audio: {e}"}, status_code=400)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio_duration = waveform.shape[1] / 16000
    log.info("  Audio: %.2fs, %d samples", audio_duration, waveform.shape[1])

    fh = _make_log_handler(mode)
    _attach_handler(fh)
    try:
        if mode == "online":
            result = recognize_online_streaming_waveform(
                model=engine.model,
                sp=engine.sp,
                params=engine.params,
                device=engine.device,
                waveform=waveform,
                decoder_step_chunks=num_decode_chunks,
            )
        else:
            result = recognize_pseudo_streaming_waveform(
                model=engine.model,
                sp=engine.sp,
                params=engine.params,
                device=engine.device,
                waveform=waveform,
                decode_interval_samples=PSEUDO_DECODE_INTERVAL_SAMPLES,
                min_decode_samples=PSEUDO_DECODE_INTERVAL_SAMPLES,
                soft_segment_samples=PSEUDO_DECODE_INTERVAL_SAMPLES * 8,
                hard_segment_samples=PSEUDO_DECODE_INTERVAL_SAMPLES * 14,
                num_decode_chunks=num_decode_chunks,
            )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    finally:
        _detach_handler(fh)

    log.info(
        "  Result: %s (mode=%s, decode_calls=%d, elapsed=%.2fs)",
        result["text"],
        mode,
        result.get("decode_calls", 0),
        result["elapsed"],
    )

    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
