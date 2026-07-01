import torch
import torchaudio


class FbankExtractor:
    """Incremental Fbank feature extractor for streaming audio.

    Handles overlap between chunks (25ms frame length with 10ms shift means
    each frame needs 15ms of context from the previous chunk).
    """

    def __init__(self, sample_rate: int = 16000, num_mel_bins: int = 80):
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.frame_length_ms = 25.0
        self.frame_shift_ms = 10.0
        self.frame_length_samples = int(sample_rate * self.frame_length_ms / 1000)  # 400
        self.frame_shift_samples = int(sample_rate * self.frame_shift_ms / 1000)    # 160
        self.overlap_samples = self.frame_length_samples - self.frame_shift_samples  # 240
        self._tail = torch.zeros(0, dtype=torch.float32)

    def reset(self):
        self._tail = torch.zeros(0, dtype=torch.float32)

    def extract(self, pcm_float: torch.Tensor) -> torch.Tensor:
        """Extract fbank features from a PCM chunk.

        Args:
            pcm_float: (num_samples,) float32 tensor, range [-1, 1].

        Returns:
            fbank: (num_frames, num_mel_bins) or empty if not enough samples.
        """
        if self._tail.numel() > 0:
            waveform = torch.cat([self._tail, pcm_float])
        else:
            waveform = pcm_float

        num_samples = waveform.numel()
        if num_samples < self.frame_length_samples:
            self._tail = waveform
            return torch.empty(0, self.num_mel_bins)

        waveform_2d = waveform.unsqueeze(0)  # (1, num_samples)
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform_2d,
            num_mel_bins=self.num_mel_bins,
            sample_frequency=self.sample_rate,
            frame_length=self.frame_length_ms,
            frame_shift=self.frame_shift_ms,
            dither=0.0,
            snip_edges=False,
        )

        num_frames = fbank.shape[0]
        consumed_samples = (num_frames - 1) * self.frame_shift_samples + self.frame_length_samples
        remaining = num_samples - (num_frames * self.frame_shift_samples)
        self._tail = waveform[-self.overlap_samples:] if remaining > 0 else torch.zeros(0, dtype=torch.float32)

        return fbank

    def flush(self) -> torch.Tensor:
        """Process any remaining audio in the buffer."""
        if self._tail.numel() >= self.frame_length_samples:
            fbank = self.extract(torch.zeros(0, dtype=torch.float32))
            self._tail = torch.zeros(0, dtype=torch.float32)
            return fbank
        return torch.empty(0, self.num_mel_bins)
