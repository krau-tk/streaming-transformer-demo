#!/usr/bin/env python3
"""Offline streaming ASR test: read an audio file, slice it into chunks,
and feed through the same streaming encoder + decoder pipeline.

This simulates real-time streaming on a known audio file to verify the
pipeline produces correct output.

Usage:
    python test_offline.py /path/to/audio.wav
    python test_offline.py /path/to/audio.wav --mode full-encoder

Modes:
    streaming    - True streaming: encoder_embed.streaming_forward + encoder.streaming_forward
    full-encoder - Pseudo streaming: full encoder on entire audio, then chunked decoder
                   (same as decode_stream_attention_kv_cache.py - known to work)
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJ_DIR = Path("/nfs/bichunhao/uag-zipformer-transformer-streaming")
sys.path.insert(0, str(PROJ_DIR / "zipformer"))
sys.path.insert(0, str(Path("/nfs/asr/icefall")))

from model_loader import ASREngine  # noqa: E402
from online_session import OnlineASRSession, SAMPLES_PER_CHUNK, FBANK_CHUNK_SIZE  # noqa: E402
from decode_stream_attention_kv_cache import (  # noqa: E402
    compute_chunk_boundaries,
    simulated_streaming_decode_kv,
)
from icefall.utils import AttributeDict  # noqa: E402

import math
LOG_EPS = math.log(1e-10)


def test_streaming_mode(engine: ASREngine, waveform: torch.Tensor):
    """Test using the true streaming pipeline (same as online demo)."""
    log.info("=== STREAMING MODE ===")
    session = OnlineASRSession(
        model=engine.model,
        sp=engine.sp,
        params=engine.params,
        device=engine.device,
    )

    # Slice waveform into chunks (simulating WebSocket frames)
    import numpy as np
    samples = waveform.squeeze().numpy()
    chunk_size_samples = SAMPLES_PER_CHUNK
    num_chunks = len(samples) // chunk_size_samples

    log.info("Audio: %.2fs, %d samples, will send %d chunks of %d samples",
             len(samples) / 16000, len(samples), num_chunks, chunk_size_samples)

    start_time = time.time()
    for i in range(num_chunks):
        chunk = samples[i * chunk_size_samples:(i + 1) * chunk_size_samples]
        pcm_bytes = (chunk * 32768).astype(np.int16).tobytes()
        result = session.feed_audio(pcm_bytes)
        if result:
            log.info("  Chunk %d: %s", i + 1, result["text"])

    # Send remaining
    remaining = samples[num_chunks * chunk_size_samples:]
    if len(remaining) > 0:
        pcm_bytes = (remaining * 32768).astype(np.int16).tobytes()
        result = session.feed_audio(pcm_bytes)
        if result:
            log.info("  Remaining: %s", result["text"])

    # Finalize
    final = session.finalize()
    elapsed = time.time() - start_time
    log.info("  FINAL: %s", final["text"])
    log.info("  Time: %.2fs, RTF: %.3f", elapsed, elapsed / (len(samples) / 16000))
    return final["text"]


def test_full_encoder_mode(engine: ASREngine, waveform: torch.Tensor):
    """Test using full encoder + chunked decoder (known-good approach)."""
    log.info("=== FULL-ENCODER MODE (reference) ===")
    model = engine.model
    sp = engine.sp
    params = engine.params
    device = engine.device

    # Compute fbank on entire audio
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        sample_frequency=16000,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
    )
    log.info("Fbank: %s", fbank.shape)

    feature = fbank.unsqueeze(0).to(device)  # (1, T, 80)
    feature_lens = torch.tensor([fbank.shape[0]], dtype=torch.long, device=device)

    # Causal padding (same as decode script)
    pad_len = 30
    feature_lens = feature_lens + pad_len
    feature = torch.nn.functional.pad(feature, (0, 0, 0, pad_len), value=LOG_EPS)

    start_time = time.time()

    with torch.inference_mode():
        encoder_out, encoder_out_lens = model.forward_encoder(feature, feature_lens)

    log.info("Encoder out: %s, lens=%s", encoder_out.shape, encoder_out_lens)

    # Compute chunk boundaries
    audio_duration = waveform.shape[1] / 16000
    chunk_boundaries = compute_chunk_boundaries(
        audio_duration=audio_duration,
        chunk_size_frames=32,
        frame_shift=0.01,
        start_current_time=0.025,
        downsample_factor=4,
        frame_offset=7,
        num_chunks_per_segment=1,
    )
    log.info("Chunk boundaries: %d chunks", len(chunk_boundaries))

    # Build params for decode
    decode_params = AttributeDict({
        "sos_id": params.sos_id,
        "eos_id": params.eos_id,
        "wait_id": params.wait_id,
        "max_token_len": 200,
    })

    with torch.inference_mode():
        hyp_tokens = simulated_streaming_decode_kv(
            model=model,
            full_encoder_out=encoder_out[:, :encoder_out_lens[0], :],
            chunk_boundaries=chunk_boundaries,
            params=decode_params,
        )

    elapsed = time.time() - start_time
    text = sp.decode(hyp_tokens)
    log.info("  RESULT: %s", text)
    log.info("  Time: %.2fs, RTF: %.3f", elapsed, elapsed / audio_duration)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", help="Path to wav/flac audio file (16kHz mono)")
    parser.add_argument("--mode", choices=["streaming", "full-encoder", "both"],
                        default="both", help="Which mode to test")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Load audio
    waveform, sr = torchaudio.load(args.audio_file)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    log.info("Loaded %s: %.2fs, sr=%d", args.audio_file, waveform.shape[1] / sr, sr)

    # Load model
    engine = ASREngine(device=args.device)

    if args.mode in ("full-encoder", "both"):
        text_full = test_full_encoder_mode(engine, waveform)

    if args.mode in ("streaming", "both"):
        text_stream = test_streaming_mode(engine, waveform)

    if args.mode == "both":
        log.info("")
        log.info("=== COMPARISON ===")
        log.info("  Full-encoder: %s", text_full)
        log.info("  Streaming:    %s", text_stream)
        if text_full == text_stream:
            log.info("  ✓ Results match!")
        else:
            log.info("  ✗ Results differ — streaming encoder may have issues")


if __name__ == "__main__":
    main()
