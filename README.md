🧬 Nanopore Signal Tokenizer
将 Nanopore 原始电流信号（5 kHz）转换为离散 token 序列，用于下游语言模型（如 GPT）建模 DNA/RNA 序列。

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

🔍 简介

本工具基于 自监督残差矢量量化（Residual VQ） 模型，将 Nanopore 测序仪输出的原始电流信号（单位：pA）直接 tokenize 为结构化离散符号序列，格式如下：

< bwav:L1_5336 >< bwav:L2_7466 >< bwav:L3_6973 >< bwav:L4_6340 >...
支持 多层级 token 输出（L1 ~ L4），可灵活用于不同粒度的建模任务
兼容 FAST5 文件 和 原始浮点信号数组
内置 信号归一化 + Butterworth 滤波，提升鲁棒性
支持 长信号分块处理（sliding window with overlap）

适用于：
Nanopore 信号语言建模（Signal LM）
无参考序列的 RNA/DNA 表征学习
多模态生物信息学 pipeline 构建

⚙️ 安装

从源码安装（推荐）

bash
git clone https://github.com/lovearthai/nanopore_signal_tokenizer.git
cd nanopore_signal_tokenizer
pip install -e .

🚀 快速开始
1. 加载预训练模型

将你的 checkpoint（如 nanopore_rvq_tokenizer_chunk12k.pth）放入 models/ 目录。
2. Tokenize 模拟信号

python
example_tokenize_data.py
import numpy as np
from nanopore_signal_tokenizer import RVQTokenizer

tokenizer = RVQTokenizer(
model_ckpt="models/nanopore_rvq_tokenizer_chunk12k.pth",
device="cuda:0",
cutoff=1200,
chunk_size=12000,
downsample_rate=12
)
模拟一段 1200 点的信号（~240ms @ 5kHz）
signal = np.random.randn(1200).astype(np.float32) * 5 + 100
获取全部层级 token (L1–L4)
tokens_all = tokenizer.tokenize_data(signal, fs=5000)
print(tokens_all)
< bwav:L1_5336 >< bwav:L2_7466 >< bwav:L3_6973 >< bwav:L4_6340 >...
仅获取 L1 层
tokens_L1 = tokenizer.tokenize_data(signal, token_type="L1", fs=5000)
print(tokens_L1)
< bwav:L1_5336 >< bwav:L1_434 >< bwav:L1_4037 >...
3. Tokenize FAST5 文件

python
tokenizer.tokenize_fast5_file(
fast5_path="sample.fast5",
output_path="output.jsonl.gz"
)

输出为 gzip 压缩的 JSONL 格式：

json
{"id": "read_12345", "text": "< bwav:L1_123 >< bwav:L2_456 >..."}
{"id": "read_67890", "text": "< bwav:L1_789 >< bwav:L2_012 >..."}

🛠️ 配置参数说明

参数 说明 默认值
------ ------ --------
model_ckpt 预训练模型路径 必填
device 推理设备 "cuda"
cutoff 滤波截止频率 (Hz) 1200
filter_order Butterworth 滤波器阶数 6
default_fs 默认采样率 (Hz) 5000
chunk_size 模型输入长度（必须与训练一致） 12000
stride 分块滑动步长（用于长 read） 11880
discard_feature 每块两端丢弃的 token 数（防边界效应） 0
downsample_rate 编码器总下采样率 12
✅ token_type 可选："L1", "L2", "L3", "L4"（默认 "L4"）

📦 项目结构

nanopore_signal_tokenizer/
├── nanopore_signal_tokenizer/
│ ├── __init__.py
│ ├── fast5.py # FAST5 读取工具
│ ├── nanopore.py # 信号归一化 & 滤波
│ ├── rvq_tokenizer.py # RVQTokenizer 核心类
│ └── NanoporeEncoder.py # 自定义编码器
├── models/ # 存放 .pth 模型文件
├── example_tokenize_data.py # 快速示例
└── setup.py

📄 引用与许可
模型架构受 [AudioLM / EnCodec](https://arxiv.org/abs/2210.04678) 启发
使用 [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) 实现 Residual VQ

本项目采用 MIT License —— 见 [LICENSE](LICENSE)。

🙌 致谢
Oxford Nanopore Technologies for public FAST5 data format
ont_fast5_api 提供便捷的 FAST5 读取接口
PyTorch & Hugging Face 社区对自监督学习的推动

🧪 实验性项目：模型性能依赖于训练数据质量。建议在特定数据集上微调以获得最佳效果。

欢迎提交 Issue 或 PR！一起推动 Nanopore 信号语言建模的发展 🧬✨

