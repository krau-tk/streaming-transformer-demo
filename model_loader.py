import sys
import logging
from pathlib import Path

import sentencepiece as spm
import torch

PROJ_DIR = Path("/nfs/bichunhao/uag-zipformer-transformer-streaming")
ICEFALL_ROOT = Path("/nfs/asr/icefall")

sys.path.insert(0, str(PROJ_DIR / "zipformer"))
sys.path.insert(0, str(ICEFALL_ROOT))

from train_cross_node_stream_transformer import get_model, get_params  # noqa: E402
from icefall.checkpoint import average_checkpoints  # noqa: E402
from icefall.utils import AttributeDict  # noqa: E402

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BPE_MODEL = PROJ_DIR / "data/bpe_zh_5000_20260602/unigram_5000.model"
EXP_DIR = PROJ_DIR / "exp/zh_stream_attn_0.0005_20260630"
EPOCH = 27
AVG = 5


class ASREngine:
    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(BPE_MODEL))

        self.params = self._build_params()
        self.model = self._load_model()
        log.info("ASREngine initialized on %s", self.device)

    def _build_params(self) -> AttributeDict:
        params = get_params()

        params.num_encoder_layers = "2,2,4,5,4,2"
        params.feedforward_dim = "512,768,1536,2048,1536,768"
        params.encoder_dim = "192,256,512,768,512,256"
        params.encoder_unmasked_dim = "192,192,256,320,256,192"
        params.downsampling_factor = "1,2,4,8,4,2"
        params.num_heads = "4,4,4,8,4,4"
        params.query_head_dim = "32"
        params.value_head_dim = "12"
        params.pos_head_dim = "4"
        params.pos_dim = 48
        params.cnn_module_kernel = "31,31,15,15,15,31"
        params.causal = True
        params.chunk_size = "16"
        params.left_context_frames = "128"

        params.attention_decoder_dim = 768
        params.attention_decoder_num_layers = 6
        params.attention_decoder_attention_dim = 768
        params.attention_decoder_num_heads = 8
        params.attention_decoder_feedforward_dim = 3072
        params.attention_decoder_dropout = 0.1

        params.vocab_size = self.sp.get_piece_size()
        params.sos_id = self.sp.piece_to_id("<sos/eos>")
        params.eos_id = params.sos_id
        params.wait_id = self.sp.piece_to_id("<wait>")
        params.blank_id = self.sp.piece_to_id("<blk>")
        params.max_token_len = 200

        assert params.wait_id != 0, "<wait> token not found in BPE model"
        log.info("vocab_size=%d, sos/eos=%d, wait=%d", params.vocab_size, params.sos_id, params.wait_id)

        return params

    def _load_model(self) -> torch.nn.Module:
        log.info("Building model...")
        model = get_model(self.params)

        log.info("Averaging checkpoints epoch %d-%d...", EPOCH - AVG + 1, EPOCH)
        filenames = [EXP_DIR / f"epoch-{e}.pt" for e in range(EPOCH - AVG + 1, EPOCH + 1)]
        for f in filenames:
            assert f.exists(), f"Checkpoint not found: {f}"
        avg_state = average_checkpoints(filenames, device=self.device)
        model.load_state_dict(avg_state, strict=False)

        model.to(self.device)
        model.eval()
        log.info("Model loaded and set to eval mode.")
        return model
