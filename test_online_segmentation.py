from types import SimpleNamespace

import numpy as np
import torch

import online_session


class FakeOnlineASRSession:
    instances = []

    def __init__(self, **_kwargs):
        self.index = len(self.instances)
        self.instances.append(self)

    def feed_audio(self, _pcm_bytes):
        if self.index == 0:
            return {"text": "甲乙", "is_final": False}
        return {"text": "乙丙", "is_final": False}

    def finalize(self):
        if self.index == 0:
            return {"text": "甲乙", "is_final": True}
        return {"text": "乙丙", "is_final": True}


def test_hard_rollover_resets_all_state_and_deduplicates_overlap(monkeypatch):
    FakeOnlineASRSession.instances = []
    monkeypatch.setattr(
        online_session,
        "OnlineASRSession",
        FakeOnlineASRSession,
    )

    session = online_session.SegmentedOnlineASRSession(
        model=SimpleNamespace(),
        sp=SimpleNamespace(),
        params=SimpleNamespace(),
        device=torch.device("cpu"),
        decoder_step_chunks=4,
        soft_segment_samples=4,
        hard_segment_samples=8,
        commit_overlap_samples=2,
    )

    first = np.arange(4, dtype=np.int16).tobytes()
    second = np.arange(4, 8, dtype=np.int16).tobytes()

    result = session.feed_audio(first)
    assert result["text"] == "甲乙"
    assert not result["segment_committed"]

    result = session.feed_audio(second)
    assert result["text"] == "甲乙丙"
    assert result["segment_committed"]
    assert result["segment_reason"] == "hard duration"
    assert result["committed_segments"] == 1
    assert len(FakeOnlineASRSession.instances) == 2


class FakeSentenceBoundarySession(FakeOnlineASRSession):
    def feed_audio(self, _pcm_bytes):
        if self.index == 0:
            return {"text": "甲。", "is_final": False}
        return None

    def finalize(self):
        if self.index == 0:
            return {"text": "甲。乙", "is_final": True}
        return {"text": "", "is_final": True}


def test_sentence_boundary_does_not_decode_past_punctuation(monkeypatch):
    FakeSentenceBoundarySession.instances = []
    monkeypatch.setattr(
        online_session,
        "OnlineASRSession",
        FakeSentenceBoundarySession,
    )

    session = online_session.SegmentedOnlineASRSession(
        model=SimpleNamespace(),
        sp=SimpleNamespace(),
        params=SimpleNamespace(),
        device=torch.device("cpu"),
        decoder_step_chunks=4,
        soft_segment_samples=4,
        hard_segment_samples=8,
        commit_overlap_samples=2,
    )

    result = session.feed_audio(np.arange(4, dtype=np.int16).tobytes())

    assert result["text"] == "甲。"
    assert result["segment_committed"]
    assert result["segment_reason"] == "sentence boundary"
