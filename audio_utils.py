import torch
import torchaudio


class FbankExtractor:
    """Incremental Fbank feature extractor for streaming audio.

    Match whole-utterance Kaldi Fbank extraction with ``snip_edges=True``.

    After emitting N frames, only N frame shifts are consumed from the
    waveform.  The remaining samples contain the overlap needed by the next
    frame, so arbitrary input chunk boundaries neither duplicate nor shift
    feature frames.
    """

    def __init__(self, sample_rate: int = 16000, num_mel_bins: int = 80):
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.frame_length_ms = 25.0
        self.frame_shift_ms = 10.0
        self.frame_length_samples = int(sample_rate * self.frame_length_ms / 1000)  # 400
        self.frame_shift_samples = int(sample_rate * self.frame_shift_ms / 1000)    # 160
        self._waveform_buffer = torch.zeros(0, dtype=torch.float32)

    def reset(self):
        self._waveform_buffer = torch.zeros(0, dtype=torch.float32)

    def extract(self, pcm_float: torch.Tensor) -> torch.Tensor:
        """Extract fbank features from a PCM chunk.

        Args:
            pcm_float: (num_samples,) float32 tensor, range [-1, 1].

        Returns:
            fbank: (num_frames, num_mel_bins) or empty if not enough samples.
        """
        if pcm_float.ndim != 1:
            raise ValueError(
                f"Expected one-dimensional PCM, got shape {tuple(pcm_float.shape)}"
            )
        if pcm_float.device.type != "cpu":
            raise ValueError("FbankExtractor expects PCM on CPU")

        pcm_float = pcm_float.detach().to(dtype=torch.float32)
        if self._waveform_buffer.numel() > 0:
            waveform = torch.cat([self._waveform_buffer, pcm_float])
        else:
            waveform = pcm_float

        num_samples = waveform.numel()
        if num_samples < self.frame_length_samples:
            self._waveform_buffer = waveform.clone()
            return torch.empty((0, self.num_mel_bins), dtype=torch.float32)

        waveform_2d = waveform.unsqueeze(0)  # (1, num_samples)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform_2d,
            num_mel_bins=self.num_mel_bins,
            sample_frequency=self.sample_rate,
            frame_length=self.frame_length_ms,
            frame_shift=self.frame_shift_ms,
            dither=0.0,
            snip_edges=True,
        )

        # With snip_edges=True, every emitted frame advances the next frame
        # start by exactly one frame shift.  Keep the unconsumed waveform from
        # that next start position; it includes the 15 ms frame overlap.
        num_frames = fbank.shape[0]
        consumed_samples = num_frames * self.frame_shift_samples
        self._waveform_buffer = waveform[consumed_samples:].clone()

        return fbank

    def flush(self) -> torch.Tensor:
        """Emit complete frames and discard the final incomplete frame."""
        fbank = self.extract(torch.zeros(0, dtype=torch.float32))
        self._waveform_buffer = torch.zeros(0, dtype=torch.float32)
        return fbank
