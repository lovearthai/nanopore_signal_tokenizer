# cnn_model.py
"""
纯卷积自编码器（Convolutional Autoencoder）用于 Nanopore 直接 RNA 信号预训练。

该模型仅包含 encoder 和 decoder，不涉及向量量化（VQ），用于第一阶段预训练。
预训练后的 encoder 权重将被加载到后续的 VQ 模型中，以提升训练稳定性。

架构特点：
    - 输入：[B, 1, T]，T 通常为 520（对应 130 bps × 4 kHz / 1000 × 1000 ≈ 520）
    - 总下采样率：5（cnn_type=0/1）或 12（cnn_type=2）
    - 感受野：≈33（type0/1）或 ≈65（type2）采样点
    - 输出重建信号，与输入对齐

支持三种 CNN 架构：
    - cnn_type=0: 大容量非对称模型（通道 1→64→128→256）
    - cnn_type=1: 小容量严格对称模型（通道 1→16→32→64）
    - cnn_type=2: 多阶段下采样模型（通道 1→64→64→128→128→512，总 stride=12）

注意：本模型设计为**确定性重建模型**，不包含随机操作或 VQ。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Tuple


class NanoporeCNNModel(nn.Module):
    """Nanopore 信号重建用纯卷积自编码器（无 VQ）。"""

    def __init__(self, cnn_type: Literal[0, 1, 2] = 1) -> None:
        """初始化自编码器。

        Args:
            cnn_type (Literal[0, 1, 2]): CNN 架构类型。
                - 0: 大容量模型（latent_dim=256, stride=5）
                - 1: 小容量对称模型（latent_dim=64, stride=5）
                - 2: 多阶段下采样模型（latent_dim=512, stride=12）

        Raises:
            ValueError: 若 cnn_type 不为 0、1 或 2。
        """
        super().__init__()

        if cnn_type not in (0, 1, 2):
            raise ValueError(f"`cnn_type` must be 0, 1 or 2, got {cnn_type}.")

        self.cnn_type: int = cnn_type

        # 设置 latent_dim 和 stride
        if cnn_type == 0:
            self.latent_dim = 256
            self.cnn_stride = 5
            self.receptive_field = 33
        elif cnn_type == 1:
            self.latent_dim = 64
            self.cnn_stride = 5
            self.receptive_field = 33
        elif cnn_type == 2:
            self.latent_dim = 512
            self.cnn_stride = 12
            self.receptive_field = 65  # 理论感受野

        # 构建网络
        if cnn_type == 0:
            self._build_encoder_type0()
            self._build_decoder_type0()
        elif cnn_type == 1:
            self._build_encoder_type1()
            self._build_decoder_type1()
        else:  # cnn_type == 2
            self._build_encoder_type2()
            self._build_decoder_type2()

    def _build_encoder_type0(self) -> None:
        """构建大容量 encoder（1 → 64 → 128 → 256），无下采样直到最后一层。"""
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.SiLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.SiLU(),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, self.latent_dim, kernel_size=25, stride=5, padding=12, bias=True),
            nn.BatchNorm1d(self.latent_dim),
        )

    def _build_encoder_type1(self) -> None:
        """构建小容量对称 encoder（1 → 16 → 32 → 64），结构与 decoder 严格对称。"""
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2, bias=True),
            nn.SiLU(),
            nn.BatchNorm1d(16),

            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.SiLU(),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=25, stride=5, padding=12, bias=True),
            nn.BatchNorm1d(64),
        )

    def _build_encoder_type2(self) -> None:
        """cnn_type=2: 多阶段下采样，总 stride=12，输出通道=512"""
        self.encoder = nn.Sequential(
            # Layer 1: 1 → 64, stride=1
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.SiLU(),
            nn.BatchNorm1d(64),

            # Layer 2: 64 → 64, stride=1
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.SiLU(),
            nn.BatchNorm1d(64),

            # Layer 3: 64 → 128, stride=3
            nn.Conv1d(64, 128, kernel_size=9, stride=3, padding=4, bias=True),
            nn.SiLU(),
            nn.BatchNorm1d(128),

            # Layer 4: 128 → 128, stride=2
            nn.Conv1d(128, 128, kernel_size=9, stride=2, padding=4, bias=True),
            nn.SiLU(),
            nn.BatchNorm1d(128),

            # Layer 5: 128 → 512, stride=2
            nn.Conv1d(128, self.latent_dim, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm1d(self.latent_dim),
        )

    def _build_decoder_type0(self) -> None:
        """构建大容量 decoder（256 → 128 → 64 → 1），近似对称于 encoder。"""
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=self.latent_dim,
                out_channels=128,
                kernel_size=25,
                stride=5,
                padding=12,
                output_padding=0,
                bias=True,
            ),
            nn.SiLU(),
            nn.BatchNorm1d(128),

            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 1,kernel_size=5,padding=2),
        )

    def _build_decoder_type1(self) -> None:
        """构建小容量对称 decoder（64 → 32 → 16 → 1），与 encoder 严格对称。"""
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels=64,
                out_channels=32,
                kernel_size=25,
                stride=5,
                padding=12,
                output_padding=0,
                bias=True,
            ),
            nn.SiLU(),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.BatchNorm1d(16),

            nn.Conv1d(16, 1,kernel_size=5,padding=2),
        )

    def _build_decoder_type2(self) -> None:
        """严格对称 decoder: 512 → 128 → 128 → 64 → 64 → 1，上采样顺序与 encoder 下采样逆序对应"""
        # 对于 T = 12000，你的 encoder 各层下采样恰好能整除，所以 不需要 output_padding=1，应设为 0
        self.decoder = nn.Sequential(
            # Inverse of encoder Layer 5: 512 → 128, upsample ×2
            nn.ConvTranspose1d(512, 128, kernel_size=5, stride=2, padding=2, output_padding=0),
            nn.SiLU(),
            nn.BatchNorm1d(128),

            # Inverse of encoder Layer 4: 128 → 128, upsample ×2
            nn.ConvTranspose1d(128, 128, kernel_size=9, stride=2, padding=4, output_padding=0),
            nn.SiLU(),
            nn.BatchNorm1d(128),

            # Inverse of encoder Layer 3: 128 → 64, upsample ×3
            nn.ConvTranspose1d(128, 64, kernel_size=9, stride=3, padding=4, output_padding=0),
            nn.SiLU(),
            nn.BatchNorm1d(64),

            # Inverse of encoder Layer 2: 64 → 64
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.BatchNorm1d(64),

            # Inverse of encoder Layer 1: 64 → 1
            nn.Conv1d(64, 1, kernel_size=5, padding=2),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：重建输入信号。

        Args:
            x (torch.Tensor): 输入原始信号，形状 [batch_size, 1, seq_len]

        Returns:
            torch.Tensor: 重建信号，形状 [batch_size, 1, seq_len]，
                         自动对齐至输入长度。
        """
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected input shape [B, 1, T], got {x.shape}")

        z = self.encoder(x)               # [B, C, T // stride]
        recon = self.decoder(z)           # [B, 1, ~T]

        # 精确对齐输出长度至输入长度（处理边界 padding 误差）
        target_len = x.shape[-1]
        current_len = recon.shape[-1]

        if current_len > target_len:
            recon = recon[..., :target_len]
        elif current_len < target_len:
            recon = F.pad(recon, (0, target_len - current_len))

        return recon

    def get_encoder_state_dict(self) -> dict:
        """获取 encoder 的 state_dict，用于后续加载到 VQ 模型。

        Returns:
            dict: encoder 的参数字典，可直接用于 `load_state_dict()`.
        """
        return self.encoder.state_dict()
