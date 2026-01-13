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
            self.receptive_field = 65

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

    # 现代 CNN  遵循“Conv → BN → Act” 的惯例。
    # nn.SiLU()  # 等价于 x * torch.sigmoid(x)
    # 关键在于：Batch Normalization（BN）的设计假设输入是“未激活”的线性特征。
    # Conv → BatchNorm → Activation（如 SiLU/ReLU）
    # BN 的作用是标准化“线性变换后的分布”
    # BN 的目的是消除 internal covariate shift，让每一层的输入分布稳定。
    # 它假设输入来自一个近似高斯分布的线性空间（即 Conv/Wx + b 的输出）。
    # 如果你先用 SiLU（或 ReLU）非线性扭曲了分布（比如把负值压向 0，造成偏态），再做 BN：
    # BN 要去 normalize 一个高度偏斜、非对称的分布
    # 效果大打折扣，甚至可能放大噪声
    # 
    

    def _build_encoder_type0(self) -> None:
        """构建大容量 encoder（1 → 64 → 128 → 256）"""
        self.encoder = nn.Sequential(
            # Layer 1: Conv -> BN -> SiLU
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 2: Conv -> BN -> SiLU
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 3: 下采样层
            nn.Conv1d(128, self.latent_dim, kernel_size=25, stride=5, padding=12, bias=False),
            nn.BatchNorm1d(self.latent_dim),
            # 注意：最后一层可以不加激活函数，或者使用SiLU保持一致性
            # Bonito 用 Tanh 是因为它要 basecall，而你要 tokenize。
            # 它牺牲信息保真度换取 basecalling 稳定性；
            # 我们必须保留完整信号信息以支持 LLM 级建模。
        )

    def _build_encoder_type1(self) -> None:
        """构建小容量对称 encoder（1 → 16 → 32 → 64）"""
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.SiLU(),

            # Layer 2
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),

            # Layer 3: 下采样层
            nn.Conv1d(32, 64, kernel_size=25, stride=5, padding=12, bias=False),
            nn.BatchNorm1d(64),
        )

    def _build_encoder_type2(self) -> None:
        """cnn_type=2: 多阶段下采样"""
        self.encoder = nn.Sequential(
            # Layer 1
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 2
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 3: stride=3
            nn.Conv1d(64, 128, kernel_size=9, stride=3, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 4: stride=2
            nn.Conv1d(128, 128, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 5: stride=2
            nn.Conv1d(128, self.latent_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(self.latent_dim),
        )

    def _build_decoder_type0(self) -> None:
        """构建大容量 decoder（256 → 128 → 64 → 1）"""
        self.decoder = nn.Sequential(
            # ConvTranspose -> BN -> SiLU
            nn.ConvTranspose1d(
                in_channels=self.latent_dim,
                out_channels=128,
                kernel_size=25,
                stride=5,
                padding=12,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Conv -> BN -> SiLU
            nn.Conv1d(128, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # 最后一层：只卷积，不加BN和激活
            nn.Conv1d(64, 1, kernel_size=5, padding=2, bias=True),
        )

    def _build_decoder_type1(self) -> None:
        """构建小容量对称 decoder（64 → 32 → 16 → 1）"""
        self.decoder = nn.Sequential(
            # ConvTranspose -> BN -> SiLU
            nn.ConvTranspose1d(
                in_channels=64,
                out_channels=32,
                kernel_size=25,
                stride=5,
                padding=12,
                output_padding=0,
                bias=False,
            ),
            nn.BatchNorm1d(32),
            nn.SiLU(),

            # Conv -> BN -> SiLU
            nn.Conv1d(32, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.SiLU(),

            # 最后一层：只卷积，不加BN和激活
            nn.Conv1d(16, 1, kernel_size=5, padding=2, bias=True),
        )

    def _build_decoder_type2(self) -> None:
        """严格对称 decoder"""
        self.decoder = nn.Sequential(
            # Layer 5逆操作
            nn.ConvTranspose1d(512, 128, kernel_size=5, stride=2, padding=2, 
                               output_padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 4逆操作
            nn.ConvTranspose1d(128, 128, kernel_size=9, stride=2, padding=4, 
                               output_padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 3逆操作
            nn.ConvTranspose1d(128, 64, kernel_size=9, stride=3, padding=4, 
                               output_padding=0, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 2逆操作
            nn.Conv1d(64, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # 最后一层：只卷积，不加BN和激活
            nn.Conv1d(64, 1, kernel_size=5, padding=2, bias=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"Expected input shape [B, 1, T], got {x.shape}")

        z = self.encoder(x)
        recon = self.decoder(z)
        
        # 对齐长度
        target_len = x.shape[-1]
        current_len = recon.shape[-1]
        
        if current_len != target_len:
            diff = abs(current_len - target_len)
            if current_len > target_len:
                # 对称裁剪
                crop_left = diff // 2
                recon = recon[..., crop_left:crop_left + target_len]
            else:
                # 对称填充
                pad_left = diff // 2
                pad_right = diff - pad_left
                recon = F.pad(recon, (pad_left, pad_right))
        
        return recon
    
    # 在 cnn_model.py 的 NanoporeCNNModel 类中添加：
    # 输出是torch.Size([1, 64, 2400])
    # [batch, channel, time] 是 PyTorch 的标准格式（NCL），所以 [1, 64, 2400] 是完全正常且正确的。
    # 在 PyTorch 中，1D 卷积层 nn.Conv1d 的输入/输出形状是：
    # (N, C_in, L)  →  (N, C_out, L_out)
    # N: batch size
    # C: number of channels（特征维度）
    # L: sequence length（时间步/信号长度）
    # L: sequence length（时间步/信号长度）
    # 输入电信号通常是 [B, 1, signal_len]（单通道）
    # 经过 CNN 后，变成 [B, 64, T]，其中：
    # 64 是 feature channels（即你想要的“64维特征”）
    # T=2400 是下采样后的时间步数
    # 那为什么你觉得应该是 [1, 2400, 64]？
    # 因为你可能更习惯 Transformer / NLP / 机器学习中的常见格式：
    # (batch, sequence_length, feature_dim)
    # 比如 BERT、LSTM 输出通常是 [B, T, D]。
    # 但在 PyTorch 的 CNN 生态中（尤其是语音、信号处理），默认使用 channel-first（NCL） 格式，因为：
    # Conv1d、BatchNorm1d、MaxPool1d 等模块都假设通道在第1维（索引1）；
    # 这样设计对 GPU 内存访问更高效。

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input signal to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to signal."""
        return self.decoder(z)
