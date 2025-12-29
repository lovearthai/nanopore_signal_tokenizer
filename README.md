# 🧬 Nanopore Signal Tokenizer

> 将 Nanopore 原始电流信号（5 kHz）转换为离散 token 序列，用于下游语言模型（如 GPT）建模 DNA/RNA 序列。

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🔍 简介
本工具提供两种 Nanopore 原始电流信号（单位：pA）的 token 化方案，可将连续电流信号转换为结构化离散符号序列。适配不同建模需求：适用场景
Nanopore 信号语言建模（Signal LM）；
无参考序列的 RNA/DNA 表征学习；
多模态生物信息学分析流程（pipeline）构建。

1. RVQ Tokenizer（残差矢量量化）
基于自监督残差矢量量化（Residual VQ）模型实现信号 token 化，输出格式示例：<|bwav:L1_5336|><|bwav:L2_7466|><|bwav:L3_6973|><|bwav:L4_6340|>

核心特性：
支持 L1~L4 多层级 token 输出，可灵活适配不同粒度的建模任务；
兼容 FAST5 格式文件与原始浮点信号数组两种输入形式；
内置信号归一化 + Butterworth 滤波流程，提升 token 化鲁棒性；
支持长信号分块处理（滑动窗口 + 重叠策略），适配长序列建模场景

2. KMeans Tokenizer（Kmeans聚类）
基于 faiss K-Means 聚类算法实现信号 token 化：先通过聚类生成指定数量的聚类中心，再将电流信号切片为固定维度向量，通过匹配最相似的聚类中心向量，以聚类编号替换原始信号完成 token 化。注意：faiss keans不支持int64个向量的聚类，所以目前只支持2B（20亿）个向量的聚类。

本工具转换的格式为`<|bwav:5018|><|bwav:8156|><|bwav:23|><|bwav:5725|><|bwav:418|>`, 与后面的vq_tokenizer是一样的格式。

3. VQ tokenizer 

仅仅一层的离散化，支持将一个原始电流信号，转换为 `<|bwav:5808|><|bwav:702|><|bwav:330|><|bwav:6238|><|bwav:2432|>` 格式，其中数字从0拍到8191（在8k codebook情况下）


## ⚙️ 安装

### 从源码安装（推荐）

```bash
git clone https://github.com/lovearthai/nanopore_signal_tokenizer.git
cd nanopore_signal_tokenizer
pip install -e . 
#如果阿里源报错，切换清华源或官方源 
pip install -e . \
  --no-build-isolation \
  -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  --trusted-host pypi.tuna.tsinghua.edu.cn \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org

pip install vector-quantize-pytorch -i https://pypi.org/simple/
conda install -c pytorch faiss-gpu  
```

##  VQ_tokenizer

### 初始化 Tokenizer

```python
from nanopore_signal_tokenizer import VQTokenizer

tokenizer = VQTokenizer(
    model_ckpt="models/model_vq_8k_epoch10.pth",  # 必填：预训练模型路径
    device="cuda",                                # 可选：设备，默认自动选 cuda/cpu
    token_batch_size=100                          # 可选：批处理 token 的内部 batch size, 也就是每次同时转换多少个token
)
```

```bash
✅ Using device: cuda
📂 Loading checkpoint: models/model_vq_8k_epoch10.pth
🎯 Inferred: codebook_size=8192, dim=64
✅ VQTokenizer initialized:
   Checkpoint       : ...
   Device           : cuda
   Codebook size    : 8192
   Latent dim       : 64
   Downsample rate  : 5
   Chunk size       : 500
   Margin           : 25 samples
```
注意：downsample_rate、chunk_size、margin 等参数由模型 checkpoint 自动推断，不可手动覆盖

### 对一段信号进行token化-默认开启huada归一化

```python

import numpy as np

# 模拟一段信号（单位：pA，长度任意 ≥ 25）
signal = np.random.randn(1000).astype(np.float32) * 5 + 100

# 默认返回 token 列表（每个元素是字符串，如 "<|bwav:1234|>"）
tokens_list = tokenizer.tokenize_data(signal)
print(len(tokens_list))        # 输出 token 数量
print("".join(tokens_list[:10]))  # 拼接前10个 token
```

### 对一段信号进行token化-关闭归一化


```
import numpy as np

# 模拟一段信号（单位：pA，长度任意 ≥ 25）
signal = np.random.randn(1000).astype(np.float32) * 5 + 100

# 默认返回 L1 层级的 token 列表（每个元素是字符串，如 "<|bwav:1234|>"）
tokens_list = tokenizer.tokenize_data(signal, do_normalize=False)
print(len(tokens_list))        # 输出 token 数量
print("".join(tokens_list[:10]))  # 拼接前10个 token
```

### 对一段信号进行token化-启用中值滤波(窗口为5) 和低通零相位滤波（截至频率为1000HZ）

```
# 同时启用中值滤波（medf）和低通滤波（lpf）
tokens_list = tokenizer.tokenize_data(signal, medf=5, lpf=1000)

print(len(tokens_list))  # 总 token 数 = 4 × (有效信号块数)
print("".join(tokens_list[:20]))
```

### 如果传入信号少于25个，返回空的list


### 对一个fast5文件进行token化
文件自动 gzip 压缩（.gz 后缀）
自动跳过无效或过短的 reads
支持 single-read 和 multi-read FAST5

```
tokenizer.tokenize_fast5(
    fast5_path="data/sample.fast5",
    output_path="output/sample.jsonl.gz"
)
```

输出的jsonl.gz内容格式

```
{"id": "read_12345", "text": "<|bwav:5808|><|bwav:702|><|bwav:330|>..."}
{"id": "read_67890", "text": "<|bwav:2432|><|bwav:812|>..."}
```

## Kmeans Tokenizer



## RVQ Tokenizer




1. 预训练模型准备
RVQ Tokenizer：将预训练 checkpoint 文件（如 nanopore_rvq_tokenizer_chunk12k.pth）放入项目 models/ 目录；
KMeans Tokenizer：直接使用 models/ 目录下的 centroids.npy 聚类中心文件。
2. 信号 Token 化示例
RVQ Tokenizer 使用示例：参考 test/ 目录下 example_rvq_tokenizer_data.py；
KMeans Tokenizer 使用示例：参考 test/ 目录下 example_kmeans_tokenize_data.py。

