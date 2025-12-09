# 🧬 Nanopore Signal Tokenizer

> 将 Nanopore 原始电流信号（5 kHz）转换为离散 token 序列，用于下游语言模型（如 GPT）建模 DNA/RNA 序列。

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🔍 简介
本工具提供两种 Nanopore 原始电流信号（单位：pA）的 token 化方案，可将连续电流信号转换为结构化离散符号序列，适配不同建模需求：
1. RVQ Tokenizer（残差矢量量化）
基于自监督残差矢量量化（Residual VQ）模型实现信号 token 化，输出格式示例：<|bwav:L1_5336|><|bwav:L2_7466|><|bwav:L3_6973|><|bwav:L4_6340|>

核心特性：
支持 L1~L4 多层级 token 输出，可灵活适配不同粒度的建模任务；
兼容 FAST5 格式文件与原始浮点信号数组两种输入形式；
内置信号归一化 + Butterworth 滤波流程，提升 token 化鲁棒性；
支持长信号分块处理（滑动窗口 + 重叠策略），适配长序列建模场景

2. KMeans Tokenizer（K 均值聚类）
基于 K-Means 聚类算法实现信号 token 化：先通过聚类生成指定数量的聚类中心，再将电流信号切片为固定维度向量，通过匹配最相似的聚类中心向量，以聚类编号替换原始信号完成 token 化。

适用场景
Nanopore 信号语言建模（Signal LM）；
无参考序列的 RNA/DNA 表征学习；
多模态生物信息学分析流程（pipeline）构建。

## ⚙️ 安装

### 从源码安装（推荐）

```bash
git clone https://github.com/lovearthai/nanopore_signal_tokenizer.git
cd nanopore_signal_tokenizer
pip install -e .
```

##  快速开始

1. 预训练模型准备
RVQ Tokenizer：将预训练 checkpoint 文件（如 nanopore_rvq_tokenizer_chunk12k.pth）放入项目 models/ 目录；
KMeans Tokenizer：直接使用 models/ 目录下的 centroids.npy 聚类中心文件。
2. 信号 Token 化示例
RVQ Tokenizer 使用示例：参考 test/ 目录下 example_rvq_tokenizer_data.py；
KMeans Tokenizer 使用示例：参考 test/ 目录下 example_kmeans_tokenize_data.py。

