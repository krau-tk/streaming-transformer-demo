import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from audio_utils import FbankExtractor

import sys
from pathlib import Path

PROJ_DIR = Path("/nfs/bichunhao/uag-zipformer-transformer-streaming")
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


class OnlineASRSession:
    """Stateful per-connection streaming ASR session.

    Implements true streaming:
      - encoder_embed.streaming_forward (Conv2dSubsampling with cache)
      - encoder.streaming_forward (Zipformer2 with cached states)
      - attention decoder forward_step with KV cache
    """

    def __init__(self, model: nn.Module, sp, params, device: torch.device):
        self.model = model
        self.sp = sp
        self.params = params
        self.device = device

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
            max_cross_len=2048,
            device=device,
            dtype=torch.float32,
        )

        self._decoder_tokens: List[int] = [params.sos_id]
        self._enc_wait_positions: List[int] = []
        self._total_enc_frames: int = 0
        self._all_text: str = ""
        self._chunk_count: int = 0

        # Audio sample buffer
        self._audio_buffer = torch.empty(0, dtype=torch.float32)
        log.info(
            "OnlineASRSession: chunk_shift=%d fbank_frames, full stream reset on sentence punctuation",
            FBANK_CHUNK_SHIFT,
        )

    def feed_audio(self, pcm_int16_bytes: bytes) -> Optional[Dict]:
        """Feed raw PCM int16 bytes, return result dict if new text is produced.

        Returns:
            {"text": str, "is_final": False} or None if not enough data yet.
        """
        import numpy as np
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

        # Final decode pass: try to get remaining tokens after last <W>
        if self._decoder_tokens and self._decoder_tokens[-1] == self.params.wait_id:
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
        self._total_enc_frames += chunk_enc_frames

        log.info("  encoder done: out_frames=%d, total_enc_frames=%d",
                 chunk_enc_frames, self._total_enc_frames)

        # Record wait position: last frame of accumulated encoder output
        self._enc_wait_positions.append(self._total_enc_frames - 1)

        # --- Attention Decoder with KV Cache ---
        new_tokens, hit_wait, hit_eos = self._streaming_decode(
            new_enc_frames=encoder_out,
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
                "dropping chunk text and resetting stream state",
                len(new_tokens),
            )
            self._reset_stream_state(reason="decoder runaway")
            return ""

        self._decoder_tokens.extend(new_tokens)

        # Decode tokens (strip wait/eos)
        content_tokens = [
            t for t in new_tokens
            if t != self.params.wait_id and t != self.params.eos_id
        ]
        decoded_text = self.sp.decode(content_tokens) if content_tokens else ""

        if hit_eos:
            self._reset_stream_state(reason="decoder eos")
        elif self._ends_with_sentence_boundary(decoded_text):
            log.info(
                "  sentence boundary detected after text=%r; resetting stream state",
                decoded_text,
            )
            self._reset_stream_state(
                reason="sentence punctuation",
                reset_frontend=False,
            )
        return decoded_text

    def _streaming_decode(
        self,
        new_enc_frames: torch.Tensor,
        add_new_memory: bool = True,
    ) -> tuple:
        """Run attention decoder with KV cache until <W> or <EOS>."""
        device = self.device
        decoder = self.model.attention_decoder.decoder
        wait_id = self.params.wait_id
        eos_id = self.params.eos_id
        max_len = self.params.max_token_len

        memory_lens_t = torch.tensor([self._total_enc_frames], device=device)
        enc_wait = [self._enc_wait_positions]
        current_tokens = list(self._decoder_tokens)
        new_tokens: List[int] = []

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

            next_token = logits.argmax(dim=-1).item()
            current_tokens.append(next_token)
            new_tokens.append(next_token)

            if next_token == wait_id:
                return new_tokens, True, False
            if next_token == eos_id:
                return new_tokens, False, True

        return new_tokens, False, False

    def _reset_stream_state(self, reason: str, reset_frontend: bool = True) -> None:
        """Start a new segment with fresh model state.

        For punctuation-based segmentation we keep frontend buffers because the
        current 45-frame encoder window contains 13 lookahead frames that the next
        segment still needs. Dropping them loses about 130-320 ms near the boundary.
        """
        log.info(
            "  reset stream state (%s): reset_frontend=%s, audio_buffer=%d samples, fbank_buffer=%d frames",
            reason,
            reset_frontend,
            self._audio_buffer.numel(),
            self._fbank_buffer.shape[0],
        )

        # Audio / feature front-end state.
        if reset_frontend:
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
        self._decoder_tokens = [self.params.sos_id]
        self._enc_wait_positions = []
        self._total_enc_frames = 0
        self._kv_cache.reset()

    @staticmethod
    def _ends_with_sentence_boundary(text: str) -> bool:
        stripped = text.rstrip()
        return bool(stripped) and stripped[-1] in SENTENCE_END_CHARS

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
