import unittest

import torch
import torchaudio

from audio_utils import FbankExtractor


SAMPLE_RATE = 16000
NUM_MEL_BINS = 80


def offline_fbank(waveform: torch.Tensor) -> torch.Tensor:
    return torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        num_mel_bins=NUM_MEL_BINS,
        sample_frequency=SAMPLE_RATE,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
        snip_edges=True,
    )


def streaming_fbank(waveform: torch.Tensor, chunk_sizes) -> torch.Tensor:
    extractor = FbankExtractor(
        sample_rate=SAMPLE_RATE,
        num_mel_bins=NUM_MEL_BINS,
    )
    outputs = []
    offset = 0
    chunk_index = 0
    while offset < waveform.numel():
        chunk_size = chunk_sizes[chunk_index % len(chunk_sizes)]
        chunk = waveform[offset : offset + chunk_size]
        features = extractor.extract(chunk)
        if features.numel() > 0:
            outputs.append(features)
        offset += chunk.numel()
        chunk_index += 1

    final_features = extractor.flush()
    if final_features.numel() > 0:
        outputs.append(final_features)
    if not outputs:
        return torch.empty((0, NUM_MEL_BINS), dtype=torch.float32)
    return torch.cat(outputs, dim=0)


class TestFbankExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = torch.Generator().manual_seed(20260720)
        cls.waveform = torch.randn(16000 * 3 + 731, generator=generator) * 0.05

    def assert_matches_offline(self, chunk_sizes):
        expected = offline_fbank(self.waveform)
        actual = streaming_fbank(self.waveform, chunk_sizes)
        self.assertEqual(actual.shape, expected.shape)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-6)

    def test_matches_offline_for_realtime_chunks(self):
        self.assert_matches_offline([5120])

    def test_matches_offline_for_arbitrary_chunks(self):
        self.assert_matches_offline([137, 4096, 777, 5120, 83])

    def test_reset_discards_previous_waveform(self):
        extractor = FbankExtractor(SAMPLE_RATE, NUM_MEL_BINS)
        extractor.extract(self.waveform[:317])
        extractor.reset()

        actual = extractor.extract(self.waveform[317:5437])
        expected = offline_fbank(self.waveform[317:5437])
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
