import logging
import time
from typing import Dict, Optional

import numpy as np
import torch

from full_encoder_decode import SAMPLE_RATE, recognize_full_encoder_waveform

log = logging.getLogger(__name__)

DEFAULT_DECODE_INTERVAL_SAMPLES = SAMPLE_RATE
DEFAULT_MIN_DECODE_SAMPLES = SAMPLE_RATE
DEFAULT_SOFT_SEGMENT_SAMPLES = 8 * SAMPLE_RATE
DEFAULT_HARD_SEGMENT_SAMPLES = 14 * SAMPLE_RATE
DEFAULT_COMMIT_OVERLAP_SAMPLES = SAMPLE_RATE
DEFAULT_NUM_DECODE_CHUNKS = 4
SENTENCE_END_CHARS = set("。！？!?.")


class PseudoStreamingASRSession:
    """Realtime-ish session that re-runs the reference full-encoder decode.

    This is intentionally pseudo-streaming: incoming audio is accumulated, and
    every decode interval the complete prefix is encoded with model.forward_encoder.
    The path matches the offline upload recognizer much more closely than the
    true streaming encoder session.
    """

    def __init__(
        self,
        *,
        model,
        sp,
        params,
        device: torch.device,
        decode_interval_samples: int = DEFAULT_DECODE_INTERVAL_SAMPLES,
        min_decode_samples: int = DEFAULT_MIN_DECODE_SAMPLES,
        soft_segment_samples: int = DEFAULT_SOFT_SEGMENT_SAMPLES,
        hard_segment_samples: int = DEFAULT_HARD_SEGMENT_SAMPLES,
        commit_overlap_samples: int = DEFAULT_COMMIT_OVERLAP_SAMPLES,
        num_decode_chunks: int = DEFAULT_NUM_DECODE_CHUNKS,
    ):
        self.model = model
        self.sp = sp
        self.params = params
        self.device = device
        self.decode_interval_samples = decode_interval_samples
        self.min_decode_samples = min_decode_samples
        self.soft_segment_samples = soft_segment_samples
        self.hard_segment_samples = hard_segment_samples
        self.commit_overlap_samples = commit_overlap_samples
        self.num_decode_chunks = num_decode_chunks

        self._segment_samples = torch.empty(0, dtype=torch.float32)
        self._last_decode_samples = 0
        self._committed_text = ""
        self._current_text = ""

    def feed_audio(self, pcm_int16_bytes: bytes) -> Optional[Dict]:
        samples = torch.from_numpy(
            np.frombuffer(pcm_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        return self.feed_samples(samples)

    def feed_samples(self, samples: torch.Tensor) -> Optional[Dict]:
        samples = samples.detach().cpu().float().reshape(-1)
        if samples.numel() == 0:
            return None

        self._segment_samples = torch.cat([self._segment_samples, samples])
        total_samples = self._segment_samples.numel()

        log.debug(
            "pseudo feed_audio: received=%d samples, total=%.2fs",
            samples.numel(),
            total_samples / SAMPLE_RATE,
        )

        if total_samples < self.min_decode_samples:
            return None
        if total_samples - self._last_decode_samples < self.decode_interval_samples:
            return None

        self._last_decode_samples = total_samples
        return self._decode(is_final=False, allow_commit=True)

    def finalize(self) -> Dict:
        if self._segment_samples.numel() > 0:
            result = self._decode(is_final=True, allow_commit=False)
            self._commit_current_segment(preserve_overlap=False)
            result["text"] = self._committed_text
            return result
        return {"text": self._committed_text, "is_final": True}

    def _decode(self, *, is_final: bool, allow_commit: bool) -> Dict:
        current_segment_samples = self._segment_samples.numel()
        try:
            result = recognize_full_encoder_waveform(
                model=self.model,
                sp=self.sp,
                params=self.params,
                device=self.device,
                waveform=self._segment_samples.unsqueeze(0),
                num_decode_chunks=self.num_decode_chunks,
            )
        except ValueError as e:
            log.warning("pseudo decode skipped: %s", e)
            return {"text": self._combined_text(), "is_final": is_final}

        self._current_text = result["text"]
        committed_segment = False

        if allow_commit and self._should_commit_current_segment():
            self._commit_current_segment(preserve_overlap=True)
            committed_segment = True

        return {
            "text": self._combined_text(),
            "is_final": is_final,
            "mode": "pseudo_streaming_full_encoder",
            "duration": result["duration"],
            "elapsed": result["elapsed"],
            "encoder_frames": result["encoder_frames"],
            "num_decode_chunks": result["num_decode_chunks"],
            "decode_chunks": result["decode_chunks"],
            "segment_committed": committed_segment,
            "current_segment_duration": round(current_segment_samples / SAMPLE_RATE, 2),
            "commit_overlap_duration": round(
                self.commit_overlap_samples / SAMPLE_RATE, 2
            ) if committed_segment else 0.0,
        }

    def _should_commit_current_segment(self) -> bool:
        segment_samples = self._segment_samples.numel()
        if segment_samples >= self.hard_segment_samples:
            return True
        if segment_samples < self.soft_segment_samples:
            return False
        return self._ends_with_sentence_boundary(self._current_text)

    def _commit_current_segment(self, *, preserve_overlap: bool) -> None:
        if self._current_text:
            self._committed_text += self._dedupe_against_committed(self._current_text)

        if (
            preserve_overlap
            and self.commit_overlap_samples > 0
            and self._segment_samples.numel() > self.commit_overlap_samples
        ):
            self._segment_samples = self._segment_samples[
                -self.commit_overlap_samples:
            ].clone()
        else:
            self._segment_samples = torch.empty(0, dtype=torch.float32)

        self._last_decode_samples = 0
        self._current_text = ""

    def _combined_text(self) -> str:
        return self._committed_text + self._dedupe_against_committed(self._current_text)

    def _dedupe_against_committed(self, text: str) -> str:
        if not self._committed_text or not text:
            return text

        max_overlap = min(len(self._committed_text), len(text), 80)
        for num_chars in range(max_overlap, 0, -1):
            if self._committed_text[-num_chars:] == text[:num_chars]:
                return text[num_chars:]
        return text

    @staticmethod
    def _ends_with_sentence_boundary(text: str) -> bool:
        stripped = text.rstrip()
        if not stripped:
            return False
        return stripped[-1] in SENTENCE_END_CHARS


def recognize_pseudo_streaming_waveform(
    *,
    model,
    sp,
    params,
    device: torch.device,
    waveform: torch.Tensor,
    feed_chunk_samples: int = SAMPLE_RATE,
    decode_interval_samples: int = DEFAULT_DECODE_INTERVAL_SAMPLES,
    min_decode_samples: int = DEFAULT_MIN_DECODE_SAMPLES,
    soft_segment_samples: int = DEFAULT_SOFT_SEGMENT_SAMPLES,
    hard_segment_samples: int = DEFAULT_HARD_SEGMENT_SAMPLES,
    commit_overlap_samples: int = DEFAULT_COMMIT_OVERLAP_SAMPLES,
    num_decode_chunks: int = DEFAULT_NUM_DECODE_CHUNKS,
) -> Dict:
    """Recognize an uploaded file through the same segmented pseudo-stream path."""
    if waveform.dim() == 2:
        if waveform.size(0) != 1:
            raise ValueError(
                f"Expected mono waveform with shape (1, T), got {tuple(waveform.shape)}"
            )
        samples = waveform[0]
    elif waveform.dim() == 1:
        samples = waveform
    else:
        raise ValueError(f"Expected waveform with shape (T,) or (1, T), got {tuple(waveform.shape)}")

    samples = samples.detach().cpu().float().reshape(-1)
    duration = samples.numel() / SAMPLE_RATE

    session = PseudoStreamingASRSession(
        model=model,
        sp=sp,
        params=params,
        device=device,
        decode_interval_samples=decode_interval_samples,
        min_decode_samples=min_decode_samples,
        soft_segment_samples=soft_segment_samples,
        hard_segment_samples=hard_segment_samples,
        commit_overlap_samples=commit_overlap_samples,
        num_decode_chunks=num_decode_chunks,
    )

    start_time = time.time()
    partial_results = []
    decode_calls = 0
    committed_segments = 0

    for start in range(0, samples.numel(), feed_chunk_samples):
        chunk = samples[start:start + feed_chunk_samples]
        result = session.feed_samples(chunk)
        if result is None:
            continue
        decode_calls += 1
        partial_results.append(result["text"])
        if result.get("segment_committed"):
            committed_segments += 1

    final = session.finalize()
    elapsed = time.time() - start_time

    final.update(
        {
            "text": final.get("text", ""),
            "duration": round(duration, 2),
            "partial_results": partial_results,
            "mode": "offline_pseudo_streaming_full_encoder",
            "num_decode_chunks": num_decode_chunks,
            "decode_calls": decode_calls,
            "committed_segments": committed_segments,
            "elapsed": round(elapsed, 3),
            "is_final": True,
        }
    )
    return final
