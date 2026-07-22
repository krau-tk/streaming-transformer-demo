import logging
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from audio_utils import FbankExtractor

import sys
from pathlib import Path

PROJ_DIR = Path("/nfs_tmk/asr/bichunhao/uag-zipformer-transformer-streaming")
sys.path.insert(0, str(PROJ_DIR / "zipformer"))
sys.path.insert(0, str(Path("/nfs/asr/icefall")))

from attention_decoder_stream import KVCache  # noqa: E402
from icefall.utils import make_pad_mask  # noqa: E402

log = logging.getLogger(__name__)
SENTENCE_END_CHARS = set("。！？!?.")

# Chunk sizing:
# chunk_size = 16 (encoder-level frames after Conv2dSubsampling)
# pad_length = 7 + 2*3 = 13  (Conv2d kernel + ConvNeXt padding)
# fbank_chunk_size = chunk_size * 2 + pad_length = 45 fbank frames per chunk
# fbank_chunk_shift = chunk_size * 2 = 32 fbank frames per step
# At 10ms frame shift: 45 frames = 450ms of audio
# Each new encoder call advances by 32 frames = 320ms, keeping 13 frames overlap
# After Zipformer2 output_downsampling_factor=2: 16 → 8 encoder output frames per chunk
CHUNK_SIZE = 16
PAD_LENGTH = 7 + 2 * 3  # 13
FBANK_CHUNK_SHIFT = CHUNK_SIZE * 2  # 32
FBANK_CHUNK_SIZE = CHUNK_SIZE * 2 + PAD_LENGTH  # 45
FRAME_SHIFT_MS = 10.0
SAMPLE_RATE = 16000
SAMPLES_PER_CHUNK = int(FBANK_CHUNK_SHIFT * FRAME_SHIFT_MS / 1000 * SAMPLE_RATE)  # 5120
DECODER_RESET_WARMUP_CHUNKS = 0
DEFAULT_DECODER_STEP_CHUNKS = 4
DEFAULT_ONLINE_SOFT_SEGMENT_SAMPLES = int(5.12 * SAMPLE_RATE)
DEFAULT_ONLINE_HARD_SEGMENT_SAMPLES = int(8.96 * SAMPLE_RATE)
DEFAULT_ONLINE_COMMIT_OVERLAP_SAMPLES = int(1.28 * SAMPLE_RATE)


class OnlineASRSession:
    """Stateful per-connection streaming ASR session.

    Implements true streaming:
      - encoder_embed.streaming_forward (Conv2dSubsampling with cache)
      - encoder.streaming_forward (Zipformer2 with cached states)
      - attention decoder forward_step with KV cache
    """

    def __init__(
        self,
        model: nn.Module,
        sp,
        params,
        device: torch.device,
        decoder_step_chunks: int = DEFAULT_DECODER_STEP_CHUNKS,
    ):
        self.model = model
        self.sp = sp
        self.params = params
        self.device = device
        self.decoder_step_chunks = max(1, int(decoder_step_chunks))

        self.fbank = FbankExtractor(sample_rate=SAMPLE_RATE, num_mel_bins=80)
        self._fbank_buffer = torch.empty(0, 80)

        # Encoder states
        self._embed_cache = model.encoder_embed.get_init_states(
            batch_size=1, device=device
        )
        self._encoder_states = model.encoder.get_init_states(
            batch_size=1, device=device
        )
        self._processed_lens = torch.zeros(1, dtype=torch.int64, device=device)

        # Decoder states
        decoder = model.attention_decoder.decoder
        self._kv_cache = KVCache(num_layers=decoder.num_layers)
        layer0_attn = decoder.layers[0].self_attn
        num_heads = layer0_attn.num_heads
        head_dim = layer0_attn.head_dim
        self._kv_cache.init_buffers(
            batch=1,
            num_heads=num_heads,
            head_dim=head_dim,
            max_self_len=512,
            max_cross_len=max(
                8,
                int(getattr(decoder, "cross_attention_window_frames", 0)) or 2048,
            ),
            device=device,
            dtype=torch.float32,
        )

        self._decoder_tokens: List[int] = [params.sos_id]
        self._enc_wait_positions: List[int] = []
        self._total_enc_frames: int = 0
        self._decoder_warmup_chunks: int = 0
        self._decoder_memory_buffer: List[torch.Tensor] = []
        self._all_text: str = ""
        self._chunk_count: int = 0

        # Audio sample buffer
        self._audio_buffer = torch.empty(0, dtype=torch.float32)
        log.info(
            "OnlineASRSession: chunk_shift=%d fbank_frames, decoder_step_chunks=%d, "
            "exclude_wait_from_self_cache=%s, cross_attention_window_frames=%d, "
            "wait_alternative_beam_size=%d",
            FBANK_CHUNK_SHIFT,
            self.decoder_step_chunks,
            getattr(decoder, "exclude_wait_from_self_cache", False),
            getattr(decoder, "cross_attention_window_frames", 0),
            getattr(params, "wait_alternative_beam_size", 1),
        )

    def feed_audio(self, pcm_int16_bytes: bytes) -> Optional[Dict]:
        """Feed raw PCM int16 bytes, return result dict if new text is produced.

        Returns:
            {"text": str, "is_final": False} or None if not enough data yet.
        """
        samples = torch.from_numpy(
            np.frombuffer(pcm_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        self._audio_buffer = torch.cat([self._audio_buffer, samples])

        log.info(
            "feed_audio: received %d bytes (%d samples), buffer=%d, need=%d",
            len(pcm_int16_bytes), len(samples), self._audio_buffer.numel(), SAMPLES_PER_CHUNK
        )

        # Process complete audio steps. The fbank/encoder window itself keeps
        # 13 frames overlap, matching the exported streaming examples.
        new_text_parts = []
        while self._audio_buffer.numel() >= SAMPLES_PER_CHUNK:
            chunk_samples = self._audio_buffer[:SAMPLES_PER_CHUNK]
            self._audio_buffer = self._audio_buffer[SAMPLES_PER_CHUNK:]

            text = self._process_audio_chunk(chunk_samples)
            log.info("  chunk processed, text=%r", text)
            if text:
                new_text_parts.append(text)

        if new_text_parts:
            new_text = "".join(new_text_parts)
            self._all_text += new_text
            return {"text": self._all_text, "is_final": False}
        return None

    def finalize(self) -> Dict:
        """Process remaining audio and run final decode pass."""
        # Process any remaining audio that's enough for partial fbank
        if self._audio_buffer.numel() > 0:
            fbank = self.fbank.extract(self._audio_buffer)
            self._audio_buffer = torch.empty(0, dtype=torch.float32)
            if fbank.numel() > 0:
                self._fbank_buffer = torch.cat([self._fbank_buffer, fbank])

        # Process any complete windows first, then pad the final partial window.
        while self._fbank_buffer.shape[0] >= FBANK_CHUNK_SIZE:
            fbank_chunk = self._fbank_buffer[:FBANK_CHUNK_SIZE]
            self._fbank_buffer = self._fbank_buffer[FBANK_CHUNK_SHIFT:]
            text = self._run_encoder_decoder(fbank_chunk)
            if text:
                self._all_text += text

        if self._fbank_buffer.shape[0] > 0:
            pad_needed = FBANK_CHUNK_SIZE - self._fbank_buffer.shape[0]
            padding = torch.full((pad_needed, 80), -23.0)  # log(1e-10) ≈ -23
            fbank_chunk = torch.cat([self._fbank_buffer, padding])
            self._fbank_buffer = torch.empty(0, 80)
            text = self._run_encoder_decoder(fbank_chunk)
            if text:
                self._all_text += text

        # Reveal any remaining encoder chunks that did not fill a full decoder
        # step.  This keeps uploaded-file tests from dropping the tail.
        text = self._decode_buffered_encoder_memory(force=True)
        if text:
            self._all_text += text

        # With grouped decoder steps, the forced buffered-memory decode above has
        # already exposed every available encoder frame.  Running another pass
        # without new memory advances the wait schedule only and can repeat the
        # beginning of a short utterance.  Keep the legacy extra pass solely for
        # the one-chunk mode.
        if (
            self.decoder_step_chunks == 1
            and self._decoder_tokens
            and self._decoder_tokens[-1] == self.params.wait_id
        ):
            final_text = self._final_decode_pass()
            if final_text:
                self._all_text += final_text

        return {"text": self._all_text, "is_final": True}

    def _process_audio_chunk(self, samples: torch.Tensor) -> str:
        """Process one chunk's worth of audio samples through the full pipeline."""
        fbank = self.fbank.extract(samples)
        if fbank.numel() == 0:
            log.warning("  _process_audio_chunk: fbank empty")
            return ""

        self._fbank_buffer = torch.cat([self._fbank_buffer, fbank])
        log.info("  fbank extracted: %d frames, buffer=%d, need=%d",
                 fbank.shape[0], self._fbank_buffer.shape[0], FBANK_CHUNK_SIZE)

        # Process all complete fbank windows. Each encoder call sees 45 frames
        # but we advance by only 32 frames, preserving the 13-frame lookahead
        # required by Conv2dSubsampling.streaming_forward.
        text_parts = []
        while self._fbank_buffer.shape[0] >= FBANK_CHUNK_SIZE:
            fbank_chunk = self._fbank_buffer[:FBANK_CHUNK_SIZE]
            self._fbank_buffer = self._fbank_buffer[FBANK_CHUNK_SHIFT:]

            text = self._run_encoder_decoder(fbank_chunk)
            if text:
                text_parts.append(text)

        return "".join(text_parts)

    @torch.inference_mode()
    def _run_encoder_decoder(self, fbank_chunk: torch.Tensor) -> str:
        """Run one fbank chunk through encoder streaming + decoder with KV cache.

        Args:
            fbank_chunk: (FBANK_CHUNK_SIZE, 80) tensor
        """
        device = self.device
        self._chunk_count += 1
        log.info("  _run_encoder_decoder: chunk #%d, fbank shape=%s",
                 self._chunk_count, fbank_chunk.shape)

        # --- Encoder Embed (Conv2dSubsampling) streaming ---
        x = fbank_chunk.unsqueeze(0).to(device)  # (1, 45, 80)
        x_lens = torch.tensor([FBANK_CHUNK_SIZE], dtype=torch.long, device=device)

        x, x_lens, self._embed_cache = self.model.encoder_embed.streaming_forward(
            x=x,
            x_lens=x_lens,
            cached_left_pad=self._embed_cache,
        )
        # x: (1, chunk_size=16, embed_dim)

        # --- Build padding mask with processed_lens ---
        left_context_len = self.model.encoder.left_context_frames[0]
        processed_mask = torch.arange(left_context_len, device=device).expand(
            1, left_context_len
        )
        processed_mask = (self._processed_lens.unsqueeze(1) <= processed_mask).flip(1)

        src_key_padding_mask = torch.zeros(1, x.size(1), dtype=torch.bool, device=device)
        src_key_padding_mask = torch.cat([processed_mask, src_key_padding_mask], dim=1)

        self._processed_lens = self._processed_lens + x_lens

        # --- Encoder (Zipformer2) streaming ---
        x = x.permute(1, 0, 2)  # (chunk_size, 1, C)

        encoder_out, encoder_out_lens, self._encoder_states = (
            self.model.encoder.streaming_forward(
                x=x,
                x_lens=x_lens,
                states=self._encoder_states,
                src_key_padding_mask=src_key_padding_mask,
            )
        )
        encoder_out = encoder_out.permute(1, 0, 2)  # (1, T_out, C)
        chunk_enc_frames = encoder_out.shape[1]

        log.info(
            "  encoder done: out_frames=%d, buffered_chunks=%d/%d, revealed_enc_frames=%d",
            chunk_enc_frames,
            len(self._decoder_memory_buffer) + 1,
            self.decoder_step_chunks,
            self._total_enc_frames,
        )

        self._decoder_memory_buffer.append(encoder_out)

        if self._decoder_warmup_chunks > 0:
            self._decoder_warmup_chunks -= 1
            log.info(
                "  decoder warmup: buffered chunk, remaining=%d",
                self._decoder_warmup_chunks,
            )
            if self._decoder_warmup_chunks > 0:
                return ""

        return self._decode_buffered_encoder_memory(force=False)

    def _decode_buffered_encoder_memory(self, *, force: bool) -> str:
        if not self._decoder_memory_buffer:
            return ""
        if not force and len(self._decoder_memory_buffer) < self.decoder_step_chunks:
            return ""

        decoder_memory = torch.cat(self._decoder_memory_buffer, dim=1)
        self._decoder_memory_buffer = []
        memory_frames = decoder_memory.shape[1]

        self._total_enc_frames += memory_frames
        self._enc_wait_positions.append(self._total_enc_frames - 1)

        log.info(
            "  decoder step: reveal_frames=%d, total_revealed=%d, wait_positions=%d",
            memory_frames,
            self._total_enc_frames,
            len(self._enc_wait_positions),
        )

        # --- Attention Decoder with KV Cache ---
        new_tokens, hit_wait, hit_eos = self._streaming_decode(
            new_enc_frames=decoder_memory,
            add_new_memory=True,
        )

        log.info(
            "  decoder done: new_tokens=%s, hit_wait=%s, hit_eos=%s",
            new_tokens,
            hit_wait,
            hit_eos,
        )

        if not hit_wait and not hit_eos:
            log.warning(
                "  decoder runaway: generated %d tokens without wait/eos; "
                "dropping chunk text and resetting decoder state",
                len(new_tokens),
            )
            self._reset_decoder_state(
                reason="decoder runaway",
                warmup_chunks=DECODER_RESET_WARMUP_CHUNKS,
            )
            return ""

        self._decoder_tokens.extend(new_tokens)

        # Decode tokens (strip wait/eos)
        content_tokens = [
            t for t in new_tokens
            if t != self.params.wait_id and t != self.params.eos_id
        ]
        decoded_text = self.sp.decode(content_tokens) if content_tokens else ""

        if hit_eos and not hit_wait:
            # A real, separately trained EOS can end the current decoder segment.
            # Keep acoustic states alive so the next segment does not lose frames.
            self._reset_decoder_state(
                reason="decoder eos",
                warmup_chunks=DECODER_RESET_WARMUP_CHUNKS,
            )
        elif (
            hit_wait
            and getattr(self.params, "reset_decoder_on_sentence_boundary", True)
            and self._ends_with_sentence_boundary(decoded_text)
        ):
            # Training data is effectively sentence-like, so a sentence-final
            # punctuation token followed by <wait> can make the decoder keep
            # emitting <wait>.  Reset only decoder state; encoder state and
            # frontend buffers remain continuous.
            self._reset_decoder_state(
                reason="sentence punctuation",
                warmup_chunks=0,
            )
        return decoded_text

    def _streaming_decode(
        self,
        new_enc_frames: torch.Tensor,
        add_new_memory: bool = True,
    ) -> tuple:
        """Decode until <W>/<EOS>, retaining a close non-wait alternative."""
        device = self.device
        decoder = self.model.attention_decoder.decoder
        wait_id = self.params.wait_id
        eos_id = self.params.eos_id
        max_len = self.params.max_token_len

        memory_lens_t = torch.tensor([self._total_enc_frames], device=device)
        enc_wait = [self._enc_wait_positions]
        current_tokens = list(self._decoder_tokens)
        new_tokens: List[int] = []
        cumulative_score = 0.0
        alternative_beam_size = max(
            1,
            int(getattr(self.params, "wait_alternative_beam_size", 1)),
        )
        alternative_margin = float(
            getattr(self.params, "wait_alternative_logit_margin", 0.0)
        )
        length_penalty = float(
            getattr(self.params, "wait_alternative_length_penalty", 0.6)
        )

        for _ in range(max_len):
            x_ids = torch.tensor(
                [[current_tokens[-1]]], dtype=torch.long, device=device
            )
            num_prev = sum(1 for t in current_tokens[:-1] if t == wait_id)

            logits = decoder.forward_step(
                x_ids=x_ids,
                cache=self._kv_cache,
                memory=new_enc_frames,
                memory_lens=memory_lens_t,
                enc_wait_positions=enc_wait,
                wait_id=wait_id,
                new_memory=add_new_memory,
                num_prev_wait=num_prev,
            )

            add_new_memory = False

            log_probs = torch.log_softmax(logits[0].float(), dim=-1)
            next_token = int(log_probs.argmax().item())

            if next_token == wait_id and alternative_beam_size > 1:
                non_wait_scores = log_probs.clone()
                non_wait_scores[wait_id] = float("-inf")
                non_wait_scores[eos_id] = float("-inf")
                alternative_token = int(non_wait_scores.argmax().item())
                margin = float((log_probs[wait_id] - non_wait_scores[alternative_token]).item())

                if margin <= alternative_margin:
                    wait_tokens = new_tokens + [wait_id]
                    wait_score = cumulative_score + float(log_probs[wait_id].item())
                    alternative_cache = self._kv_cache.clone()
                    (
                        alternative_tokens,
                        alternative_hit_wait,
                        alternative_hit_eos,
                        alternative_score,
                    ) = self._rollout_non_wait_alternative(
                        cache=alternative_cache,
                        current_tokens=current_tokens,
                        new_tokens=new_tokens,
                        first_token=alternative_token,
                        first_token_score=float(non_wait_scores[alternative_token].item()),
                        cumulative_score=cumulative_score,
                        new_enc_frames=new_enc_frames,
                        memory_lens_t=memory_lens_t,
                        enc_wait=enc_wait,
                        max_new_tokens=min(
                            max_len - len(new_tokens),
                            max(
                                1,
                                int(
                                    getattr(
                                        self.params,
                                        "wait_alternative_max_tokens",
                                        8,
                                    )
                                ),
                            ),
                        ),
                    )

                    wait_rank = wait_score / (max(1, len(wait_tokens)) ** length_penalty)
                    alternative_rank = alternative_score / (
                        max(1, len(alternative_tokens)) ** length_penalty
                    )
                    if alternative_hit_wait and alternative_rank > wait_rank:
                        log.info(
                            "  wait alternative selected: margin=%.3f wait_rank=%.3f "
                            "alternative_rank=%.3f tokens=%s",
                            margin,
                            wait_rank,
                            alternative_rank,
                            alternative_tokens,
                        )
                        self._kv_cache = alternative_cache
                        return alternative_tokens, True, False

                    log.info(
                        "  wait alternative retained then rejected: margin=%.3f "
                        "wait_rank=%.3f alternative_rank=%.3f hit_wait=%s hit_eos=%s",
                        margin,
                        wait_rank,
                        alternative_rank,
                        alternative_hit_wait,
                        alternative_hit_eos,
                    )

            current_tokens.append(next_token)
            new_tokens.append(next_token)
            cumulative_score += float(log_probs[next_token].item())

            if next_token == wait_id:
                return new_tokens, True, False
            if next_token == eos_id:
                return new_tokens, False, True

        return new_tokens, False, False

    def _rollout_non_wait_alternative(
        self,
        *,
        cache: KVCache,
        current_tokens: List[int],
        new_tokens: List[int],
        first_token: int,
        first_token_score: float,
        cumulative_score: float,
        new_enc_frames: torch.Tensor,
        memory_lens_t: torch.Tensor,
        enc_wait: List[List[int]],
        max_new_tokens: int,
    ) -> tuple:
        """Greedily complete one close non-wait branch without committing it."""
        decoder = self.model.attention_decoder.decoder
        wait_id = self.params.wait_id
        eos_id = self.params.eos_id
        branch_current = list(current_tokens) + [first_token]
        branch_new = list(new_tokens) + [first_token]
        branch_score = cumulative_score + first_token_score

        for _ in range(max(0, max_new_tokens - 1)):
            x_ids = torch.tensor(
                [[branch_current[-1]]],
                dtype=torch.long,
                device=self.device,
            )
            num_prev = sum(1 for token in branch_current[:-1] if token == wait_id)
            logits = decoder.forward_step(
                x_ids=x_ids,
                cache=cache,
                memory=new_enc_frames,
                memory_lens=memory_lens_t,
                enc_wait_positions=enc_wait,
                wait_id=wait_id,
                new_memory=False,
                num_prev_wait=num_prev,
            )
            log_probs = torch.log_softmax(logits[0].float(), dim=-1)
            token = int(log_probs.argmax().item())
            branch_current.append(token)
            branch_new.append(token)
            branch_score += float(log_probs[token].item())

            if token == wait_id:
                return branch_new, True, False, branch_score
            if token == eos_id:
                return branch_new, False, True, branch_score

        return branch_new, False, False, branch_score

    def _reset_stream_state(self, reason: str) -> None:
        """Fully reset acoustic and decoder state.

        Use this only when the current stream/session is being discarded. Keeping
        frontend buffers while resetting encoder state misaligns the 13-frame
        lookahead window used by encoder_embed.streaming_forward.
        """
        log.info(
            "  reset stream state (%s): audio_buffer=%d samples, fbank_buffer=%d frames",
            reason,
            self._audio_buffer.numel(),
            self._fbank_buffer.shape[0],
        )

        # Audio / feature front-end state.
        self._audio_buffer = torch.empty(0, dtype=torch.float32)
        self._fbank_buffer = torch.empty(0, 80)
        self.fbank.reset()

        # Encoder states.
        self._embed_cache = self.model.encoder_embed.get_init_states(
            batch_size=1, device=self.device
        )
        self._encoder_states = self.model.encoder.get_init_states(
            batch_size=1, device=self.device
        )
        self._processed_lens = torch.zeros(1, dtype=torch.int64, device=self.device)

        # Decoder states.
        self._reset_decoder_state(reason=reason)

    def _reset_decoder_state(self, reason: str, warmup_chunks: int = 0) -> None:
        """Reset attention-decoder segmentation without dropping acoustic context."""
        log.info("  reset decoder state (%s), warmup_chunks=%d", reason, warmup_chunks)
        self._decoder_tokens = [self.params.sos_id]
        self._enc_wait_positions = []
        self._total_enc_frames = 0
        self._decoder_warmup_chunks = warmup_chunks
        self._decoder_memory_buffer = []
        self._kv_cache.reset()

    @staticmethod
    def _ends_with_sentence_boundary(text: str) -> bool:
        stripped = text.rstrip()
        return bool(stripped) and stripped[-1] in SENTENCE_END_CHARS

    @torch.inference_mode()
    def _final_decode_pass(self) -> str:
        """Run one more decode pass without new memory to get remaining tokens."""
        device = self.device
        decoder = self.model.attention_decoder.decoder
        wait_id = self.params.wait_id
        eos_id = self.params.eos_id
        max_len = self.params.max_token_len

        memory_lens_t = torch.tensor([self._total_enc_frames], device=device)
        enc_wait = [self._enc_wait_positions]
        current_tokens = list(self._decoder_tokens)
        new_tokens: List[int] = []

        # Use a dummy memory (won't be projected since add_new_memory=False)
        dummy = torch.zeros(1, 1, self.model.encoder_dim, device=device)

        for _ in range(max_len):
            x_ids = torch.tensor(
                [[current_tokens[-1]]], dtype=torch.long, device=device
            )
            num_prev = sum(1 for t in current_tokens[:-1] if t == wait_id)

            logits = decoder.forward_step(
                x_ids=x_ids,
                cache=self._kv_cache,
                memory=dummy,
                memory_lens=memory_lens_t,
                enc_wait_positions=enc_wait,
                wait_id=wait_id,
                new_memory=False,
                num_prev_wait=num_prev,
            )

            next_token = logits.argmax(dim=-1).item()
            current_tokens.append(next_token)
            new_tokens.append(next_token)

            if next_token == eos_id:
                break
            if next_token == wait_id:
                break

        self._decoder_tokens.extend(new_tokens)
        content_tokens = [t for t in new_tokens if t != wait_id and t != eos_id]
        if content_tokens:
            return self.sp.decode(content_tokens)
        return ""


class SegmentedOnlineASRSession:
    """Bound the lifetime of true-online states for long audio.

    A fresh ``OnlineASRSession`` is created at a stable sentence boundary, or
    at the hard duration limit. A short PCM overlap is replayed into the new
    session so speech crossing the boundary is not dropped. Text decoded from
    that overlap is removed with longest suffix/prefix matching.

    The reset is deliberately complete: frontend, encoder, decoder self-cache,
    and decoder cross-cache all start from the same acoustic position.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        sp,
        params,
        device: torch.device,
        decoder_step_chunks: int = DEFAULT_DECODER_STEP_CHUNKS,
        soft_segment_samples: int = DEFAULT_ONLINE_SOFT_SEGMENT_SAMPLES,
        hard_segment_samples: int = DEFAULT_ONLINE_HARD_SEGMENT_SAMPLES,
        commit_overlap_samples: int = DEFAULT_ONLINE_COMMIT_OVERLAP_SAMPLES,
    ):
        if soft_segment_samples <= 0:
            raise ValueError("soft_segment_samples must be positive")
        if hard_segment_samples < soft_segment_samples:
            raise ValueError("hard_segment_samples must be >= soft_segment_samples")
        if commit_overlap_samples < 0:
            raise ValueError("commit_overlap_samples must be non-negative")
        if commit_overlap_samples >= hard_segment_samples:
            raise ValueError("commit_overlap_samples must be smaller than hard_segment_samples")

        self.model = model
        self.sp = sp
        self.params = params
        self.device = device
        self.decoder_step_chunks = max(1, int(decoder_step_chunks))
        self.soft_segment_samples = int(soft_segment_samples)
        self.hard_segment_samples = int(hard_segment_samples)
        self.commit_overlap_samples = int(commit_overlap_samples)

        self._session = self._new_session()
        self._segment_pcm = np.empty(0, dtype=np.int16)
        self._committed_text = ""
        self._current_text = ""
        self._committed_segments = 0

        log.info(
            "SegmentedOnlineASRSession: soft=%.2fs, hard=%.2fs, overlap=%.2fs",
            self.soft_segment_samples / SAMPLE_RATE,
            self.hard_segment_samples / SAMPLE_RATE,
            self.commit_overlap_samples / SAMPLE_RATE,
        )

    def _new_session(self) -> OnlineASRSession:
        return OnlineASRSession(
            model=self.model,
            sp=self.sp,
            params=self.params,
            device=self.device,
            decoder_step_chunks=self.decoder_step_chunks,
        )

    def feed_audio(self, pcm_int16_bytes: bytes) -> Optional[Dict]:
        if not pcm_int16_bytes:
            return None
        if len(pcm_int16_bytes) % 2 != 0:
            raise ValueError("PCM int16 byte length must be even")

        pcm = np.frombuffer(pcm_int16_bytes, dtype=np.int16).copy()
        self._segment_pcm = np.concatenate([self._segment_pcm, pcm])

        result = self._session.feed_audio(pcm_int16_bytes)
        if result is not None:
            self._current_text = result.get("text", "")

        commit_reason = self._commit_reason()
        if commit_reason is not None:
            final = self._session.finalize()
            self._current_text = final.get("text", self._current_text)
            self._commit_and_rollover(commit_reason)
            return {
                "text": self._combined_text(),
                "is_final": False,
                "segment_committed": True,
                "segment_reason": commit_reason,
                "committed_segments": self._committed_segments,
            }

        if result is None:
            return None
        return {
            "text": self._combined_text(),
            "is_final": False,
            "segment_committed": False,
            "committed_segments": self._committed_segments,
        }

    def finalize(self) -> Dict:
        final = self._session.finalize()
        self._current_text = final.get("text", self._current_text)
        if self._current_text:
            self._committed_text += self._dedupe_against_committed(
                self._current_text
            )
        self._current_text = ""
        self._segment_pcm = np.empty(0, dtype=np.int16)
        return {
            "text": self._committed_text,
            "is_final": True,
            "committed_segments": self._committed_segments,
        }

    def _commit_reason(self) -> Optional[str]:
        segment_samples = self._segment_pcm.size
        if segment_samples >= self.hard_segment_samples:
            return "hard duration"
        if (
            segment_samples >= self.soft_segment_samples
            and self._ends_with_sentence_boundary(self._current_text)
        ):
            return "sentence boundary"
        return None

    def _commit_and_rollover(self, reason: str) -> None:
        segment_samples = self._segment_pcm.size
        if self._current_text:
            self._committed_text += self._dedupe_against_committed(
                self._current_text
            )

        if self.commit_overlap_samples > 0:
            overlap = self._segment_pcm[-self.commit_overlap_samples:].copy()
        else:
            overlap = np.empty(0, dtype=np.int16)

        self._session = self._new_session()
        self._segment_pcm = overlap
        self._current_text = ""
        self._committed_segments += 1

        if overlap.size > 0:
            replay = self._session.feed_audio(overlap.tobytes())
            if replay is not None:
                self._current_text = replay.get("text", "")

        log.info(
            "  online segment committed (%s): duration=%.2fs, overlap=%.2fs, "
            "committed_segments=%d, committed_text=%r, replay_text=%r",
            reason,
            segment_samples / SAMPLE_RATE,
            overlap.size / SAMPLE_RATE,
            self._committed_segments,
            self._committed_text,
            self._current_text,
        )

    def _combined_text(self) -> str:
        return self._committed_text + self._dedupe_against_committed(
            self._current_text
        )

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
        return bool(stripped) and stripped[-1] in SENTENCE_END_CHARS


def recognize_online_streaming_waveform(
    *,
    model,
    sp,
    params,
    device: torch.device,
    waveform: torch.Tensor,
    feed_chunk_samples: int = SAMPLES_PER_CHUNK,
    decoder_step_chunks: int = DEFAULT_DECODER_STEP_CHUNKS,
    segment_long_audio: bool = True,
    soft_segment_samples: int = DEFAULT_ONLINE_SOFT_SEGMENT_SAMPLES,
    hard_segment_samples: int = DEFAULT_ONLINE_HARD_SEGMENT_SAMPLES,
    commit_overlap_samples: int = DEFAULT_ONLINE_COMMIT_OVERLAP_SAMPLES,
) -> Dict:
    """Recognize an uploaded file through the true online streaming path."""
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

    if segment_long_audio:
        session = SegmentedOnlineASRSession(
            model=model,
            sp=sp,
            params=params,
            device=device,
            decoder_step_chunks=decoder_step_chunks,
            soft_segment_samples=soft_segment_samples,
            hard_segment_samples=hard_segment_samples,
            commit_overlap_samples=commit_overlap_samples,
        )
    else:
        session = OnlineASRSession(
            model=model,
            sp=sp,
            params=params,
            device=device,
            decoder_step_chunks=decoder_step_chunks,
        )

    start_time = time.time()
    partial_results = []
    decode_calls = 0

    for start in range(0, samples.numel(), feed_chunk_samples):
        chunk = samples[start:start + feed_chunk_samples]
        chunk_np = chunk.clamp(-1.0, 1.0).numpy()
        pcm = np.where(chunk_np < 0, chunk_np * 0x8000, chunk_np * 0x7FFF)
        result = session.feed_audio(pcm.astype(np.int16).tobytes())
        if result is None:
            continue
        decode_calls += 1
        partial_results.append(result["text"])

    final = session.finalize()
    elapsed = time.time() - start_time

    final.update(
        {
            "text": final.get("text", ""),
            "duration": round(duration, 2),
            "partial_results": partial_results,
            "mode": "offline_online_streaming",
            "feed_chunk_samples": feed_chunk_samples,
            "feed_chunk_duration": round(feed_chunk_samples / SAMPLE_RATE, 3),
            "decoder_step_chunks": session.decoder_step_chunks,
            "decoder_step_duration": round(
                session.decoder_step_chunks * FBANK_CHUNK_SHIFT * FRAME_SHIFT_MS / 1000,
                3,
            ),
            "decode_calls": decode_calls,
            "segment_long_audio": segment_long_audio,
            "committed_segments": final.get("committed_segments", 0),
            "soft_segment_duration": round(soft_segment_samples / SAMPLE_RATE, 3)
            if segment_long_audio else 0.0,
            "hard_segment_duration": round(hard_segment_samples / SAMPLE_RATE, 3)
            if segment_long_audio else 0.0,
            "commit_overlap_duration": round(commit_overlap_samples / SAMPLE_RATE, 3)
            if segment_long_audio else 0.0,
            "elapsed": round(elapsed, 3),
            "is_final": True,
        }
    )
    return final
