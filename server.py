import logging
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


import torchaudio

from full_encoder_decode import recognize_full_encoder_waveform
from model_loader import ASREngine
from pseudo_streaming_session import PseudoStreamingASRSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

engine: ASREngine = None
STATIC_DIR = Path(__file__).parent / "static"


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
async def websocket_asr(ws: WebSocket):
    await ws.accept()
    session = PseudoStreamingASRSession(
        model=engine.model,
        sp=engine.sp,
        params=engine.params,
        device=engine.device,
        decode_interval_samples=16000,
        min_decode_samples=16000,
        soft_segment_samples=10 * 16000,
        hard_segment_samples=20 * 16000,
        num_decode_chunks=4,
    )
    log.info("New pseudo-streaming ASR session started")

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


@app.post("/api/recognize")
async def recognize_file(
    file: UploadFile = File(...),
    num_decode_chunks: int = Query(1, ge=1, le=16),
):
    """Offline recognition using the reference full-encoder decode path."""
    import io

    log.info(
        "Offline recognize(full-encoder): %s, num_decode_chunks=%d",
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

    try:
        result = recognize_full_encoder_waveform(
            model=engine.model,
            sp=engine.sp,
            params=engine.params,
            device=engine.device,
            waveform=waveform,
            num_decode_chunks=num_decode_chunks,
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    log.info(
        "  Result: %s (encoder_frames=%d, chunks=%d, elapsed=%.2fs)",
        result["text"],
        result["encoder_frames"],
        result["decode_chunks"],
        result["elapsed"],
    )

    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
