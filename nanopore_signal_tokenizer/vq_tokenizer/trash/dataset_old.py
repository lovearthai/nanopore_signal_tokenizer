# train_nanopore_rvq.py
# 本脚本目标：训练一个自监督模型，将 Nanopore 原始电流信号（5kHz）转换为离散 token 序列，
# 用于后续语言模型（如 GPT）建模 DNA/RNA 序列。
# 所有注释均为工业级详细说明，适合 PyTorch 新手理解。

import os
import torch                     # PyTorch 主库，用于张量计算和深度学习
import torch.nn as nn            # 神经网络模块（如 Conv1d, BatchNorm, SiLU）
import torch.nn.functional as F  # 函数式接口（如 loss, padding）
from torch.utils.data import Dataset, DataLoader  # 数据加载工具
import numpy as np               # 数值计算（生成模拟信号）
from tqdm import tqdm            # 进度条显示

# 替换 encodec RVQ 为轻量级实现
from vector_quantize_pytorch import ResidualVQ


# ----------------------------
# 1. 真实 Nanopore 数据集（从 .npy chunks 目录加载）
# ----------------------------
class NanoporeSignalDataset(Dataset):
    """
    从预处理好的 .npy chunk 文件目录加载真实 Nanopore 信号。
    每个 .npy 文件由 process_fast5_to_chunks.py 生成，格式为 list of dicts:
        {
            'read_id': str,
            'chunk_start_pos': int,
            'chunk_end_pos': int,
            'chunk_data': np.ndarray (shape=(window_size,))
        }
    本 Dataset 将所有 chunk_data 合并为一个扁平列表，每个样本是一段固定长度的信号。
    """
    def __init__(self, npy_dir, expected_chunk_len=32):
        """
        Args:
            npy_dir (str): 包含 .npy chunk 文件的目录路径
            expected_chunk_len (int): 每个 chunk 的预期长度（如 32）
        """
        self.npy_dir = npy_dir
        self.expected_chunk_len = expected_chunk_len
        self.chunks = []  # 存储所有 chunk_data (numpy arrays)

        # 收集所有 .npy 文件
        npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
        if not npy_files:
            raise ValueError(f"No .npy files found in {npy_dir}")

        print(f"📂 Loading chunks from {len(npy_files)} .npy files in {npy_dir}...")
        for fname in tqdm(npy_files, desc="Loading .npy files"):
            path = os.path.join(npy_dir, fname)
            try:
                data = np.load(path, allow_pickle=True)
                for item in data:
                    chunk = item['chunk_data']
                    if chunk.shape[0] != self.expected_chunk_len:
                        print(f"⚠️ Skipping chunk with unexpected length {chunk.shape[0]} in {fname}")
                        continue
                    self.chunks.append(chunk.astype(np.float32))
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")

        print(f"✅ Loaded {len(self.chunks)} valid chunks (each length={expected_chunk_len})")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        """
        返回单个 chunk 作为 [1, T] 张量（T = expected_chunk_len）
        """
        signal = self.chunks[idx]
        # 注意：此处不再做归一化！因为 .npy 已经是 huada_normalisation 处理过的
        # 如果你希望在 Dataset 中再做一次 z-score，可取消下面注释：
        # signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
        return torch.from_numpy(signal).float().unsqueeze(0)  # [1, T]

