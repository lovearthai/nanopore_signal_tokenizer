import numpy as np
from nanopore_signal_tokenizer import RVQTokenizer
"""
##  配置参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model_ckpt` | 预训练模型路径 | 必填 |
| `device` | 推理设备 | `"cuda"`,如果不填 |
| `stride` | 分块滑动步长（用于长 read） | `11880` |
| `discard_feature` | 每块两端丢弃的 token 数（防边界效应） | `0` |
| `downsample_rate` | 编码器总下采样率 | `12` |

> ✅ `token_type`（非初始化参数，用于 `tokenize_data` / `tokenize_read`）可选：`"L1"`, `"L2"`, `"L3"`, `"L4"`（默认 `"L4"`）
"""
tokenizer = RVQTokenizer(
    model_ckpt="../../models/nanopore_rvq_tokenizer_chunk12k.pth",
    token_batch_size=1000
)

# 模拟一段 1200 点的信号（~240ms @ 5kHz）
signal = np.random.randn(1220).astype(np.float32) * 5 + 100
# 获取全部层级 token (L1–L4)
tokens_list = tokenizer.tokenize_data(signal)
print(len(tokens_list))
tokens_str = "".join(tokens_list[:20])
print(tokens_str)
# <|bwav:L1_1309|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|><|bwav:L1_4762|>

tokens_list = tokenizer.tokenize_data(signal,token_type="L4")
tokens_str = "".join(tokens_list[:20])
print(len(tokens_list))
print(tokens_str)
# <|bwav:L1_1309|><|bwav:L2_242|><|bwav:L3_4639|><|bwav:L4_6598|><|bwav:L1_4762|><|bwav:L2_2426|><|bwav:L3_4639|><|bwav:L4_3720|><|bwav:L1_4762|><|bwav:L2_2426|><|bwav:L3_4639|><|bwav:L4_6598|><|bwav:L1_4762|><|bwav:L2_3400|><|bwav:L3_4639|><|bwav:L4_6598|><|bwav:L1_4762|><|bwav:L2_3400|><|bwav:L3_4639|><|bwav:L4_4453|>

signal = np.random.randn(120).astype(np.float32) * 5 + 100
tokens_list = tokenizer.tokenize_data(signal)
print(len(tokens_list))

signal = np.random.randn(12).astype(np.float32) * 5 + 100
tokens_list = tokenizer.tokenize_data(signal)
print(len(tokens_list))

signal = np.random.randn(10).astype(np.float32) * 5 + 100
tokens_list = tokenizer.tokenize_data(signal)
print(len(tokens_list))
print("".join(tokens_list[:20]))

#输出为 gzip 压缩的 JSONL 格式：
tokenizer.tokenize_fast5(
    fast5_path="demo.fast5",
    output_path="demo.jsonl.gz"
)
# 输出示例（每行一个 JSON 对象）：
"""
{"id": "read_12345", "text": "<|bwav:L1_123|><|bwav:L2_456|>..."}
{"id": "read_67890", "text": "<|bwav:L1_789|><|bwav:L2_012|>..."}
"""
#输出为 gzip 压缩的 JSONL 格式：
tokenizer.tokenize_fast5(
    fast5_path="demo.fast5",
    output_path="demo.jsonl.gz",
    token_type="L4"
)

