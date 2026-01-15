# ----------------------------
# 1. 真实 Nanopore 数据集（高效分布式加载版）
# ----------------------------
import os
import glob
import pickle
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader  # 数据加载工具
from tqdm import tqdm            # 进度条显
import numpy as np               # 数值计算（生成模拟信号
import os
import torch                     # PyTorch 主库，用于张量计算和深度学习
import torch.nn as nn            # 神经网络模块（如 Conv1d, BatchNorm, SiLU）
import torch.nn.functional as F  # 函数式接口（如 loss, padding）
class NanoporeSignalDataset(Dataset):
    """
    高效加载预处理的 Nanopore chunks，支持多 GPU 分布式训练。
    
    关键改进：
    - 不在 __init__ 中加载任何信号数据
    - 只构建 (file_path, local_index) 的索引列表
    - __getitem__ 时才读取单个 chunk（通过 np.load + pickling）
    - 支持任意大小的 .npy 文件（即使单个文件很大）
    """
    def __init__(self, npy_dir: str, expected_chunk_len: int = 12000):
        self.npy_dir = npy_dir
        self.expected_chunk_len = expected_chunk_len
        
        # Step 1: 收集所有 .npy 文件路径
        self.npy_files: List[str] = sorted(
            glob.glob(os.path.join(npy_dir, "*.npy"))
        )
        if not self.npy_files:
            raise ValueError(f"No .npy files found in {npy_dir}")
        
        # Step 2: 构建全局索引表 [(file_idx, local_idx), ...]
        self.index_map: List[Tuple[int, int]] = []
        total_chunks = 0
        
        print(f"📂 Building index from {len(self.npy_files)} .npy files in {npy_dir}...")
        for file_idx, file_path in enumerate(tqdm(self.npy_files, desc="Indexing files")):
            try:
                # 只加载一次以获取长度（不保存数据！）
                data = np.load(file_path, allow_pickle=True)
                num_chunks = len(data)
                # 记录每个 chunk 的 (file_idx, local_idx)
                for local_idx in range(num_chunks):
                    # 可选：提前验证长度（避免 __getitem__ 报错）
                    chunk_len = data[local_idx]['chunk_data'].shape[0]
                    if chunk_len == expected_chunk_len:
                        self.index_map.append((file_idx, local_idx))
                    else:
                        print(f"⚠️ Skipping invalid chunk at {file_path}[{local_idx}] (len={chunk_len})")
                total_chunks += num_chunks
            except Exception as e:
                print(f"❌ Error indexing {file_path}: {e}")
        
        print(f"✅ Indexed {len(self.index_map)} valid chunks (expected length={expected_chunk_len}) "
              f"from {total_chunks} total entries")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int):
        """
        按需加载单个 chunk，避免全量内存占用。
        注意：每次调用都会触发一次磁盘 I/O，可通过 DataLoader(num_workers>0) 并行化。
        """
        file_idx, local_idx = self.index_map[idx]
        file_path = self.npy_files[file_idx]
        
        try:
            # 每次只加载一个文件（但整个文件会被 unpickle）
            # 如果单个 .npy 很大（如 >1GB），建议在预处理阶段拆分成小文件（<100MB）
            data = np.load(file_path, allow_pickle=True)
            chunk = data[local_idx]['chunk_data']
            
            if chunk.shape[0] != self.expected_chunk_len:
                # 理论上不会发生（已在 __init__ 过滤），但保留防御
                raise ValueError(f"Chunk length mismatch: {chunk.shape[0]} vs {self.expected_chunk_len}")
            
            # 转为 tensor [1, T]
            return torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0)
        
        except Exception as e:
            raise RuntimeError(f"Failed to load chunk {idx} from {file_path}[{local_idx}]: {e}")
