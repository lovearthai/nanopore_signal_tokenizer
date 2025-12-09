import numpy as np
from nanopore_signal_tokenizer import RVQTokenizer
"""
##  配置参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model_ckpt` | 预训练模型路径 | 必填 |
| `device` | 推理设备 | `"cuda"` |
| `cutoff` | 滤波截止频率 (Hz) | `1200` |
| `filter_order` | Butterworth 滤波器阶数 | `6` |
| `default_fs` | 默认采样率 (Hz) | `5000` |
| `chunk_size` | 模型输入长度（必须与训练一致） | `12000` |
| `stride` | 分块滑动步长（用于长 read） | `11880` |
| `discard_feature` | 每块两端丢弃的 token 数（防边界效应） | `0` |
| `downsample_rate` | 编码器总下采样率 | `12` |

> ✅ `token_type`（非初始化参数，用于 `tokenize_data` / `tokenize_read`）可选：`"L1"`, `"L2"`, `"L3"`, `"L4"`（默认 `"L4"`）
"""
tokenizer = RVQTokenizer(
    model_ckpt="models/nanopore_rvq_tokenizer_chunk12k.pth",
    device="cuda:0",
    cutoff=1200,
    chunk_size=12000,
    downsample_rate=12
)

# 模拟一段 1200 点的信号（~240ms @ 5kHz）
signal = np.random.randn(1200).astype(np.float32) * 5 + 100

# 获取全部层级 token (L1–L4)
tokens_all = tokenizer.tokenize_data(signal, fs=5000)
print(tokens_all)
# <|bwav:L1_5336|><|bwav:L2_7466|><|bwav:L3_6973|><|bwav:L4_6340|>...

# 仅获取 L1 层
tokens_L1 = tokenizer.tokenize_data(signal, token_type="L1", fs=5000)
print(tokens_L1)
# <|bwav:L1_5336|><|bwav:L1_434|><|bwav:L1_4037|>...


#输出为 gzip 压缩的 JSONL 格式：
tokenizer.tokenize_fast5_file(
    fast5_path="sample.fast5",
    output_path="output.jsonl.gz"
)
# 输出示例（每行一个 JSON 对象）：
"""
{"id": "read_12345", "text": "<|bwav:L1_123|><|bwav:L2_456|>..."}
{"id": "read_67890", "text": "<|bwav:L1_789|><|bwav:L2_012|>..."}
"""

