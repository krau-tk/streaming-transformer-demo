from types import SimpleNamespace

import torch

from online_session import OnlineASRSession


class FakeCache:
    def clone(self):
        return FakeCache()


class FakeDecoder:
    def forward_step(self, *, x_ids, **_kwargs):
        token = int(x_ids[0, 0].item())
        logits = torch.full((1, 6), -10.0)
        if token == 1:
            logits[0, 2] = 5.0
            logits[0, 3] = 4.8
        else:
            logits[0, 2] = 5.0
        return logits


def make_session(margin: float) -> OnlineASRSession:
    session = OnlineASRSession.__new__(OnlineASRSession)
    session.device = torch.device("cpu")
    session.model = SimpleNamespace(
        attention_decoder=SimpleNamespace(decoder=FakeDecoder())
    )
    session.params = SimpleNamespace(
        sos_id=1,
        eos_id=0,
        wait_id=2,
        max_token_len=8,
        wait_alternative_beam_size=2,
        wait_alternative_logit_margin=margin,
        wait_alternative_length_penalty=0.6,
        wait_alternative_max_tokens=4,
    )
    session._decoder_tokens = [1]
    session._kv_cache = FakeCache()
    session._total_enc_frames = 8
    session._enc_wait_positions = [7]
    return session


def test_close_non_wait_alternative_is_selected():
    session = make_session(margin=0.21)
    tokens, hit_wait, hit_eos = session._streaming_decode(
        torch.randn(1, 8, 4)
    )
    assert tokens == [3, 2]
    assert hit_wait
    assert not hit_eos


def test_distant_non_wait_alternative_is_rejected():
    session = make_session(margin=0.1)
    tokens, hit_wait, hit_eos = session._streaming_decode(
        torch.randn(1, 8, 4)
    )
    assert tokens == [2]
    assert hit_wait
    assert not hit_eos
