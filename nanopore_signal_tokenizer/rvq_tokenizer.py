# nanopore_signal_tokenizer/rvq_tokenizer.py

import os
import json
import gzip
import numpy as np
import torch
from multiprocessing import Process
from math import ceil
from ont_fast5_api.fast5_interface import get_fast5_file
from .nanopore import nanopore_normalize, nanopore_filter
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.signal import medfilt
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
# from NanoporeEncoder import NanoporeEncoder  # 👈 添加这一行


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

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)  # 👈 必须放在最开头！

# ----------------------------
# 2. Nanopore 专用编码器（严格按你提供的配置）
# ----------------------------
class NanoporeEncoder(nn.Module):
    """
    将原始信号 [B, 1, T] 编码为高维潜在表示 [B, 512, T//12]
    结构完全按照你提供的 YAML 配置实现。
    """
    def __init__(self):
        super().__init__()  # 必须调用父类初始化
        layers = []  # 用来存放所有网络层

        # Layer 1: 卷积层
        layers.append(nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias=True))
        layers.append(nn.SiLU())
        layers.append(nn.BatchNorm1d(64))

        # Layer 2
        layers.append(nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=True))
        layers.append(nn.SiLU())
        layers.append(nn.BatchNorm1d(64))

        # Layer 3: 下采样 stride=3
        layers.append(nn.Conv1d(64, 128, kernel_size=9, stride=3, padding=4, bias=True))
        layers.append(nn.SiLU())
        layers.append(nn.BatchNorm1d(128))

        # Layer 4: stride=2
        layers.append(nn.Conv1d(128, 128, kernel_size=9, stride=2, padding=4, bias=True))
        layers.append(nn.SiLU())
        layers.append(nn.BatchNorm1d(128))

        # Layer 5: stride=2
        layers.append(nn.Conv1d(128, 512, kernel_size=5, stride=2, padding=2, bias=True))
        layers.append(nn.SiLU())
        layers.append(nn.BatchNorm1d(512))

        self.net = nn.Sequential(*layers)
        self.total_stride = 1 * 1 * 3 * 2 * 2  # = 12

    def forward(self, x):
        z = self.net(x)
        return z



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
# 3. 完整 Tokenizer 模型（Encoder + RVQ + Decoder）
# ----------------------------
class NanoporeRVQModel(nn.Module):
    """
    完整的自编码器结构：
    - Encoder: 压缩信号
    - RVQ: 将连续 latent 离散化为 tokens
    - Decoder: 从 tokens 重建原始信号（用于自监督训练）
    """
    def __init__(self, n_q=4, codebook_size=1024):
        super().__init__()
        self.encoder = NanoporeEncoder()
        dim = 512

        # 使用 vector_quantize_pytorch 的 ResidualVQ
        self.rvq = ResidualVQ(
            num_quantizers=n_q,
            dim=dim,
            codebook_size=codebook_size,
            kmeans_init=True,           # 更稳定训练
            kmeans_iters=10,
            threshold_ema_dead_code=2   # 防止码本死亡
        )

        # 解码器：上采样 ×12
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(dim, 256, kernel_size=8, stride=2, padding=3),
            nn.SiLU(),
            nn.BatchNorm1d(256),

            nn.ConvTranspose1d(256, 128, kernel_size=12, stride=2, padding=5),
            nn.SiLU(),
            nn.BatchNorm1d(128),

            nn.ConvTranspose1d(128, 64, kernel_size=18, stride=3, padding=8),
            nn.SiLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 1, kernel_size=1),
        )
        self.total_stride = self.encoder.total_stride


    def forward(self, x):
        z = self.encoder(x)  # [B, 512, T_enc]  e.g., [B, 512, 1000]

        # 转置为 [B, T_enc, 512] —— 符合 vector_quantize_pytorch 的要求
        z_transposed = z.permute(0, 2, 1)  # [B, T_enc, D]

        # ResidualVQ expects [B, T, D]
        z_q_transposed, indices, _ = self.rvq(z_transposed)

        # 转回 [B, D, T_enc] 用于 decoder
        z_q = z_q_transposed.permute(0, 2, 1)  # [B, 512, T_enc]

        recon = self.decoder(z_q)  # [B, 1, T_rec]

        # 对齐长度
        if recon.shape[2] > x.shape[2]:
            recon = recon[:, :, :x.shape[2]]
        elif recon.shape[2] < x.shape[2]:
            pad = x.shape[2] - recon.shape[2]
            recon = F.pad(recon, (0, pad))

        # indices is [B, T_enc, n_q] —— 这是合理的
        return recon, indices


class RVQTokenizer:
    """
    Nanopore RVQ Tokenizer 封装类。

    功能：
        - 加载预训练 RVQ 模型
        - tokenize 单个 read / numpy 信号 / 整个 FAST5 目录
    """

    def __init__(
        self,
        model_ckpt: str = "nanopore_rvq_tokenizer.pth",
        device: str = "cuda",
        cutoff: int = 1200,
        filter_order: int = 6,
        default_fs: int = 5000,
        chunk_size: int = 12000,
        stride: int = 11880,  # 👈 替代原来的 stride_factor，例如 12000 * 0.98 = 11760
        discard_feature: int = 5,
        downsample_rate: int = 12,
        token_type:str = "L4"
    ):
        """
        初始化 tokenizer。

        Args:
            model_ckpt (str): RVQ 模型 checkpoint 路径。
            device (str): 推理设备 ('cuda' or 'cpu')。
            cutoff (int): 滤波截止频率 (Hz)。
            filter_order (int): Butterworth 滤波器阶数。
            default_fs (int): 默认采样率 (Hz)，当 read 无 metadata 时使用。
            chunk_size (int): 模型输入 chunk 长度（必须与训练一致，如 12000）。
            stride (int): 滑动窗口步长（单位：信号点），用于长 read 分块。典型值 = chunk_size - 2*discard_signal。
            discard_feature (int): 每端丢弃的 token 数（对应 5 * 12 = 60 信号点）。
            downsample_rate (int): RVQ 下采样率（通常为 12）。
        """
        self.device = device
        self.cutoff = cutoff
        self.filter_order = filter_order
        self.default_fs = default_fs
        self.chunk_size = chunk_size
        self.stride = stride  # 👈 直接使用整数 stride
        self.discard_feature = discard_feature
        self.downsample_rate = downsample_rate
        self.discard_signal = discard_feature * downsample_rate  # e.g., 60

        # Load model
        self.model_ckpt_path = model_ckpt  # 👈 必须加这行！
        self.model = self._load_model(model_ckpt)
        self.n_q = self.model.rvq.num_quantizers  # e.g., 4
    def _load_model(self, ckpt_path):
        model = NanoporeRVQModel(n_q=4, codebook_size=8192)
        state_dict = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        return model

    def _tokenize_chunked_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        对任意长度信号进行分块 tokenize（带 discard 边界），返回扁平 token array。
        内部处理 padding / overlap / discard。
        """
        if signal.ndim != 1:
            raise ValueError("Signal must be 1D.")
        L = len(signal)
        if L == 0:
            T_expected = (L + self.downsample_rate - 1) // self.downsample_rate
            return np.zeros(T_expected * self.n_q, dtype=np.int64)

        if L < self.chunk_size:
            padded = np.pad(signal, (0, self.chunk_size - L), mode='constant')
            x = torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, tokens = self.model(x)
            tokens = tokens.squeeze(0).cpu().numpy()  # [T_full, n_q]

            start_sig = self.discard_signal
            end_sig = L - self.discard_signal
            if start_sig >= end_sig:
                T_expected = (L + self.downsample_rate - 1) // self.downsample_rate
                return np.zeros(T_expected * self.n_q, dtype=np.int64)

            start_tok = int(np.ceil(start_sig / self.downsample_rate))
            end_tok = int(np.floor(end_sig / self.downsample_rate))
            end_tok = min(end_tok, tokens.shape[0])

            if start_tok >= end_tok:
                T_expected = (L + self.downsample_rate - 1) // self.downsample_rate
                return np.zeros(T_expected * self.n_q, dtype=np.int64)

            safe_tokens = tokens[start_tok:end_tok]
            return safe_tokens.flatten()

        # Long signal: sliding window
        all_tokens = []
        start = 0
        while start < L:
            end = start + self.chunk_size
            if end > L:
                chunk = np.pad(signal[start:], (0, end - L), mode='constant')
            else:
                chunk = signal[start:end]

            x = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, tokens = self.model(x)
            tokens = tokens.squeeze(0).cpu().numpy()  # [1000, n_q]

            if start == 0:
                keep = tokens[:-self.discard_feature] if self.discard_feature > 0 else tokens
            elif end >= L:
                keep = tokens[self.discard_feature:] if self.discard_feature > 0 else tokens
            else:
                keep = tokens[self.discard_feature:-self.discard_feature] if self.discard_feature > 0 else tokens

            all_tokens.append(keep)
            start += (self.chunk_size - 2 * self.discard_signal)  # overlap by 2*discard_signal

        if not all_tokens:
            T_expected = (L + self.downsample_rate - 1) // self.downsample_rate
            return np.zeros(T_expected * self.n_q, dtype=np.int64)

        final_tokens = np.concatenate(all_tokens, axis=0)
        T_expected = (L + self.downsample_rate - 1) // self.downsample_rate
        if final_tokens.shape[0] > T_expected:
            final_tokens = final_tokens[:T_expected]
        elif final_tokens.shape[0] < T_expected:
            pad = np.zeros((T_expected - final_tokens.shape[0], self.n_q), dtype=np.int64)
            final_tokens = np.concatenate([final_tokens, pad], axis=0)
        # [L1_t0, L2_t0, L3_t0, L4_t0, L1_t1, L2_t1, L3_t1, L4_t1, ...]
        return final_tokens.flatten()

    def tokenize_data(self, signal: np.ndarray, fs: int = None, token_type: str = "L4") -> str:
        """
        对原始浮点信号进行 normalize + filter + tokenize，并按 token_type 返回格式化字符串。

        Args:
            signal (np.ndarray): 1D 浮点信号（scaled，单位 pA）
            fs (int): 采样率（Hz），若为 None 则用 default_fs
            token_type (str): "L1", "L2", "L3", or "L4"（默认 "L4"）

        Returns:
            str: 格式如 "<|bwav:L1_123|><|bwav:L2_456|>..."
        """
        layer_map = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}
        if token_type not in layer_map:
            raise ValueError(f"token_type must be one of {list(layer_map.keys())}, got {token_type}")
        n_layers = layer_map[token_type]

        if fs is None:
            fs = self.default_fs

        # Normalize
        norm_sig = nanopore_normalize(signal)
        if norm_sig.size == 0:
            return ""
        
        # 原始信号: raw_signal (采样率 5000 Hz)
        # 典型 k-mer 持续时间 ≈ 2–5 ms → 对应 10–25 个采样点

        # 推荐窗口大小：3 ~ 7（奇数）
        med_signal = medfilt(norm_sig, kernel_size=5)

        # Filter
        filtered = nanopore_filter(med_signal, fs=fs, cutoff=self.cutoff, order=self.filter_order)
        if filtered.size == 0 or np.isnan(filtered).any():
            return ""

        # Get flat token array from original method (unchanged)
        flat_tokens = self._tokenize_chunked_signal(filtered)  # shape: (T * 4,)
        if flat_tokens.size == 0:
            return ""

        # Reshape to (T, 4)
        if flat_tokens.size % self.n_q != 0:
            # Should not happen, but safe guard
            T = flat_tokens.size // self.n_q
            flat_tokens = flat_tokens[:T * self.n_q]
        tokens_2d = flat_tokens.reshape(-1, self.n_q)  # (T, 4)

        # Keep only first n_layers columns
        selected = tokens_2d[:, :n_layers]  # (T, n_layers)

        # Build formatted string
        parts = []
        for t in range(selected.shape[0]):
            for q in range(n_layers):
                token_id = int(selected[t, q])
                parts.append(f"<|bwav:L{q+1}_{token_id}|>")
        return "".join(parts)


    def tokenize_read(self, read, token_type: str = "L4") -> str:
        """
        直接 tokenize 一个 ont_fast5_api read 对象，返回格式化 token 字符串。

        Args:
            read: fast5 read object
            token_type: "L1", "L2", "L3", or "L4"

        Returns:
            str: formatted token string
        """
        # --- Scale ---
        channel_info = read.handle[read.global_key + 'channel_id'].attrs
        offset = int(channel_info['offset'])
        scaling = channel_info['range'] / channel_info['digitisation']
        raw = read.handle[read.raw_dataset_name][:]
        scaled = np.array(scaling * (raw + offset), dtype=np.float32)

        # --- Get fs ---
        try:
            fs = int(channel_info['sampling_rate'])
        except KeyError:
            fs = self.default_fs

        return self.tokenize_data(scaled, fs=fs, token_type=token_type)


    def tokenize_fast5_file(self, fast5_path: str, output_path: str):
        print(f"✅ Process {fast5_path}")
        """内部方法：处理单个 FAST5 → JSONL.GZ"""
        results = []
        with get_fast5_file(fast5_path, mode="r") as f5:
            for read in tqdm(f5.get_reads()):
                try:
                    token_str = self.tokenize_read(read)
    
                    results.append({
                        "id": read.read_id,
                        "text": token_str
                    })
                except Exception as e:
                    print(f"❌ Error on read {read.read_id} in {fast5_path}: {e}")
                    continue
    
        # Save
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        print(f"✅ Wrote {len(results)} reads to {output_path}")





