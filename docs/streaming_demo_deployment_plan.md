# 流式语音识别 Demo 部署计划

## 1. 项目概述

将当前的 **Zipformer + Streaming Attention Decoder** 模型部署为一个实时流式语音识别 Demo，支持麦克风输入，逐字输出识别结果。

### 当前模型信息

| 项目 | 值 |
|------|------|
| 模型架构 | Zipformer2 Encoder + Transformer Attention Decoder (带 Wait Token 机制) |
| 模型类 | `AsrStreamTransformerModel` |
| Encoder 层数 | 2,2,4,5,4,2 (6 stage) |
| Encoder 维度 | 192,256,512,768,512,256 |
| Attention Decoder | 6层, dim=768, heads=8, ffn=3072 |
| 词表 | BPE unigram 5000 (`data/bpe_zh_5000_20260602/unigram_5000.model`) |
| 特征 | 80维 Fbank, 帧移 10ms |
| Chunk 大小 | 16 帧 (= 160ms encoder 输入, 下采样4x 后 = 4帧/chunk) |
| 流式策略 | Wait Token `<W>` — decoder 输出 `<W>` 表示等待下一个 chunk |
| Checkpoint | `exp/zh_stream_attn_0.0005_20260630/epoch-27.pt` (avg 5) |
| 支持 KV Cache | 是 (`decode_stream_attention_kv_cache.py`) |

---

## 2. 整体架构

```
┌────────────────┐     WebSocket      ┌──────────────────────────────────────┐
│  浏览器前端     │ ◄──────────────────► │         Python 后端服务               │
│  (HTML/JS)     │   PCM 音频 chunks   │                                      │
│                │   ← 识别文本        │  ┌──────────────┐  ┌──────────────┐  │
│  - 麦克风采集   │                     │  │  Fbank 提取   │→│  Zipformer   │  │
│  - 文本展示     │                     │  │  (torchaudio) │  │  Encoder     │  │
└────────────────┘                     │  └──────────────┘  └──────┬───────┘  │
                                       │                           │          │
                                       │  ┌──────────────────────────────┐   │
                                       │  │  Attention Decoder (KV Cache) │   │
                                       │  │  逐 chunk 自回归解码           │   │
                                       │  └──────────────────────────────┘   │
                                       └──────────────────────────────────────┘
```

---

## 3. 技术方案选择

| 选项 | 方案 | 优点 | 缺点 |
|------|------|------|------|
| **A (推荐)** | FastAPI + WebSocket | 轻量、易部署、原生 async | 单机并发有限 |
| B | gRPC + protobuf | 高性能、跨语言 | 前端需额外 proxy (gRPC-web) |
| C | Gradio | 零前端代码、快速原型 | 定制性差、延迟较高 |

**推荐方案 A**：FastAPI WebSocket，前端纯 HTML/JS，无需构建工具。

---

## 4. 核心模块设计

### 4.1 音频流处理管线

```
浏览器 AudioWorklet (16kHz PCM int16)
    │
    ▼  WebSocket (binary frames, 每 160ms 一帧)
后端接收
    │
    ▼  环形缓冲区 accumulate
    │
    ▼  达到 chunk_size (16 帧 = 160ms) 触发
    │
    ▼  计算 Fbank (80维, torchaudio.compliance.kaldi.fbank)
    │
    ▼  送入 Encoder (causal, chunk-by-chunk)
    │
    ▼  Decoder 自回归生成 tokens 直到 <W> 或 <EOS>
    │
    ▼  SentencePiece decode → 文本
    │
    ▼  WebSocket 发送 partial/final 结果
```

### 4.2 流式 Encoder 推理

当前离线脚本先跑完整个 encoder 再切 chunk。真正的流式部署需要 **chunk-by-chunk encoder**：

- Zipformer2 原生支持 causal 模式 (`--causal 1`)
- 每次仅输入 `chunk_size=16` 帧 + `left_context_frames=256` 帧缓存
- Encoder 输出 4 帧/chunk (下采样因子 4)

关键改造点：
```python
# 伪代码
class StreamingEncoder:
    def __init__(self, model, chunk_size=16, left_context=256):
        self.encoder_embed = model.encoder_embed
        self.encoder = model.encoder
        self.states = None  # encoder 内部 cache

    def process_chunk(self, fbank_chunk):
        """输入 (1, chunk_size, 80), 返回 (1, 4, encoder_dim)"""
        # 拼接 left_context
        # 调用 encoder forward with cached states
        # 返回新输出 + 更新 states
```

### 4.3 流式 Attention Decoder（带 KV Cache）

已有 `decode_stream_attention_kv_cache.py` 实现。核心逻辑：

1. 新 encoder chunk 输出到达 → `new_memory=True` 将新帧投影为 cross-attn K/V
2. Decoder 自回归生成 tokens，复用已缓存的 cross-attn K/V
3. 遇到 `<W>` token → 停止，等待下一个 chunk
4. 遇到 `<EOS>` → 句子结束

### 4.4 后端服务结构

```
demo/
├── server.py              # FastAPI 主服务
├── streaming_inference.py # 流式推理引擎 (封装 encoder + decoder)
├── audio_processor.py     # Fbank 特征提取
├── config.py              # 模型路径、参数配置
├── static/
│   ├── index.html         # 前端页面
│   └── app.js             # AudioWorklet + WebSocket 逻辑
└── requirements.txt       # 依赖
```

---

## 5. 关键实现细节

### 5.1 音频前端 (浏览器)

```javascript
// AudioWorklet 采集 16kHz PCM
const audioCtx = new AudioContext({ sampleRate: 16000 });
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
// AudioWorklet → 每 160ms 发送 2560 samples (int16) via WebSocket
```

### 5.2 Fbank 特征提取

```python
import torchaudio

def compute_fbank(waveform, sample_rate=16000, num_mel_bins=80):
    """在线计算 Fbank, 输入 (1, num_samples), 输出 (num_frames, 80)"""
    features = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        sample_frequency=sample_rate,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
    )
    return features
```

### 5.3 模型加载

```python
from train_cross_node_stream_transformer import get_model, get_params, add_model_arguments

def load_model(checkpoint_path, device="cuda:0"):
    params = get_params()
    # 设置模型参数 (与 decode 脚本一致)
    params.causal = 1
    params.chunk_size = "16"
    params.left_context_frames = "256"
    params.num_encoder_layers = "2,2,4,5,4,2"
    params.encoder_dim = "192,256,512,768,512,256"
    params.feedforward_dim = "512,768,1536,2048,1536,768"
    params.encoder_unmasked_dim = "192,192,256,320,256,192"
    params.attention_decoder_dim = 768
    params.attention_decoder_num_layers = 6
    params.attention_decoder_attention_dim = 768
    params.attention_decoder_num_heads = 8
    params.attention_decoder_feedforward_dim = 3072

    model = get_model(params)
    load_checkpoint(checkpoint_path, model)
    model.to(device).eval()
    return model
```

### 5.4 真流式 vs 伪流式

| | 真流式 | 伪流式 (快速原型) |
|--|--------|-----------------|
| Encoder | chunk-by-chunk, 维护内部 states | 每次重跑整个累积音频 |
| 延迟 | ~200ms (chunk_size + 计算) | 随音频增长线性增加 |
| 复杂度 | 高 (需改造 encoder forward) | 低 (直接复用现有代码) |

**建议分两阶段**：
1. **Phase 1 (1-2天)**: 伪流式 — 每收到新 chunk 重跑 encoder + chunked decoder，快速验证端到端
2. **Phase 2 (3-5天)**: 真流式 — 改造 encoder 为 stateful chunk-by-chunk 推理

---

## 6. 部署步骤

### Phase 1: 伪流式快速原型 (1-2天)

#### Step 1: 环境准备
```bash
conda activate /nfs/asr/envs/k2
pip install fastapi uvicorn websockets
```

#### Step 2: 实现核心推理引擎
- 封装 `StreamingASREngine` 类
- 内部维护: 音频缓冲 → Fbank → Encoder → Chunked Decoder
- 每次新 audio chunk 到达，重跑整段

#### Step 3: 实现 WebSocket 服务
- `/ws/asr` WebSocket endpoint
- 接收 binary PCM frames
- 返回 JSON: `{"text": "...", "is_final": false}`

#### Step 4: 前端页面
- 麦克风采集 (AudioWorklet)
- WebSocket 连接 + 重连
- 实时文本展示 (partial + final)

#### Step 5: 测试验证
- 本地 GPU 上启动: `uvicorn server:app --host 0.0.0.0 --port 8765`
- 浏览器打开测试

### Phase 2: 真流式优化 (3-5天)

#### Step 1: Encoder 状态化改造
- 利用 Zipformer2 的 causal 特性
- 实现 `StreamingEncoderState` 管理 left_context cache
- 每个 chunk 只计算增量输出

#### Step 2: KV Cache Decoder 集成
- 使用已有的 `KVCache` 类 (`attention_decoder_stream.py`)
- 交叉注意力 cache 累积新 encoder 帧
- 自注意力复用历史 token embedding

#### Step 3: 性能优化
- `torch.inference_mode()` + FP16
- 音频帧对齐 (避免边界问题)
- 连接池管理 (多用户并发)

#### Step 4: ONNX/TorchScript 导出 (可选)
- 已有 `export-onnx-streaming.py` 可参考
- 进一步降低推理延迟

---

## 7. 延迟分析

| 环节 | 耗时估计 |
|------|----------|
| 音频传输 (WebSocket) | ~5ms |
| Fbank 计算 (16帧) | ~1ms |
| Encoder (1 chunk, GPU) | ~5-10ms |
| Decoder (自回归, ~5 tokens) | ~15-30ms |
| 网络回传 | ~5ms |
| **端到端 (不含音频积累)** | **~30-50ms** |
| **感知延迟 (含 chunk 积累 160ms)** | **~200-250ms** |

---

## 8. 依赖清单

```
# Python 后端
torch>=2.0
torchaudio>=2.0
sentencepiece
lhotse
k2
fastapi
uvicorn[standard]
websockets
numpy

# 前端
# 无额外依赖，纯 HTML + vanilla JS
```

---

## 9. 可选增强

- [ ] **VAD (语音活动检测)**: 静音段不触发 decoder，减少计算
- [ ] **标点恢复**: 后处理加标点 (可选小模型或规则)
- [ ] **热词/命令词**: 支持用户自定义词表 boost
- [ ] **多用户**: 每个 WebSocket 连接独立 session state
- [ ] **Docker 部署**: 打包为容器，支持 GPU passthrough
- [ ] **Nginx 反代**: WSS 加密 + 域名绑定

---

## 10. 风险与注意事项

1. **Encoder 状态化改造**: Zipformer2 内部有多尺度下采样，chunk-by-chunk 需处理边界对齐
2. **首帧延迟**: 模型加载 + warmup 需 5-10s，建议服务启动时预热
3. **显存占用**: 单模型 ~1GB (FP16)，每并发连接增加 ~50MB (KV cache)
4. **长音频**: decoder token 序列无限增长，建议每 30s 做一次 reset/segmentation
5. **网络环境**: 内网 Demo 用 `ws://`，公网需配 `wss://` (HTTPS + TLS)
