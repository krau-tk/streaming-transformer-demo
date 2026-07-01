import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio

PROJ_DIR = Path("/nfs/bichunhao/uag-zipformer-transformer-streaming")
ICEFALL_ROOT = Path("/nfs/asr/icefall")

sys.path.insert(0, str(PROJ_DIR / "zipformer"))
sys.path.insert(0, str(ICEFALL_ROOT))

from decode_stream_attention import compute_chunk_boundaries, simulated_streaming_decode  # noqa: E402
from decode_stream_attention_dataset import _pad_to_stride  # noqa: E402

LOG_EPS = math.log(1e-10)
SAMPLE_RATE = 16000


@torch.inference_mode()
def recognize_full_encoder_waveform(
    *,
    model,
    sp,
    params,
    device: torch.device,
    waveform: torch.Tensor,
    num_decode_chunks: int = 1,
) -> Dict:
    """Recognize a complete mono 16 kHz waveform using the reference decode path.

    This matches decode_stream_attention.py: full-utterance encoder first, then
    simulated streaming attention decoding over sliced encoder output.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2 or waveform.size(0) != 1:
        raise ValueError(f"Expected mono waveform with shape (1, T), got {tuple(waveform.shape)}")

    waveform = waveform.detach().cpu().float()
    audio_duration = waveform.shape[1] / SAMPLE_RATE

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=80,
        sample_frequency=SAMPLE_RATE,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
    )
    if fbank.numel() == 0:
        raise ValueError("Audio is too short to extract fbank features")

    feature = fbank.unsqueeze(0)
    feature_lens = torch.tensor([fbank.shape[0]], dtype=torch.long)

    feature, feature_lens = _pad_to_stride(
        feature,
        feature_lens,
        pad_stride=getattr(params, "pad_stride", 32),
    )

    feature = feature.to(device)
    feature_lens = feature_lens.to(device)

    if params.causal:
        pad_len = 30
        feature_lens = feature_lens + pad_len
        feature = torch.nn.functional.pad(
            feature, (0, 0, 0, pad_len), value=LOG_EPS
        )

    start_time = time.time()

    encoder_out, encoder_out_lens = model.forward_encoder(feature, feature_lens)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    utt_enc_out = encoder_out[:, : int(encoder_out_lens[0].item()), :]
    chunk_boundaries = compute_chunk_boundaries(
        audio_duration=audio_duration,
        chunk_size_frames=32,
        frame_shift=0.01,
        start_current_time=0.025,
        downsample_factor=4,
        frame_offset=7,
        num_chunks_per_segment=num_decode_chunks,
    )

    if chunk_boundaries:
        hyp_tokens, per_chunk_info = simulated_streaming_decode(
            model=model,
            full_encoder_out=utt_enc_out,
            chunk_boundaries=chunk_boundaries,
            params=params,
            decoding_method="greedy_search",
            beam_size=4,
            verbose_beam=False,
            is_rank0=False,
        )
    else:
        hyp_tokens = []
        per_chunk_info = []

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.time() - start_time
    text = sp.decode(hyp_tokens)

    partial_results: List[str] = []
    for _, _, token_ids in per_chunk_info:
        clean_ids = [
            t for t in token_ids
            if t not in (params.wait_id, params.eos_id)
        ]
        if clean_ids:
            partial_results.append(sp.decode(clean_ids))

    return {
        "text": text,
        "duration": round(audio_duration, 2),
        "partial_results": partial_results,
        "mode": "full_encoder_simulated_streaming",
        "num_decode_chunks": num_decode_chunks,
        "decode_chunks": len(chunk_boundaries),
        "feature_frames": int(fbank.shape[0]),
        "encoder_frames": int(utt_enc_out.size(1)),
        "elapsed": round(elapsed, 3),
    }
