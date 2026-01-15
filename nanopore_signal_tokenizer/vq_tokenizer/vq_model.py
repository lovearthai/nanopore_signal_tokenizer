import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from typing import Tuple, Dict


class NanoporeVQModel(nn.Module):
    """
    Nanopore VQ Tokenizer for Direct RNA Sequencing (130 bps, 4 kHz)

    支持多种 CNN 架构配置，通过 `cnn_type` 切换：
        - cnn_type=0: 大容量非严格对称模型（默认）
        - cnn_type=1: 小容量严格对称模型（通道数 1→16→32→64）

    设计目标通用：
        - 感受野 ≈ 33 采样点（≈1 个 RNA 碱基）
        - 总下采样率 = 5×（每碱基 ≈6 个 tokens）
        - 输出 codebook_dim 维 latent，直接用于 VQ
        - Decoder 在 cnn_type=1 时严格对称于 encoder

    适用于：VQ tokenizer + LLM basecalling pipeline
    """

    def __init__(
        self,
        codebook_size: int = 8192,
        commitment_weight: float = 1.0,
        orthogonal_reg_weight: float = 1.0,
        codebook_diversity_loss_weight: float = 1.0,
        cnn_type: int = 0,
    ):
        """
        初始化 NanoporeVQModel。

        Args:
            codebook_size (int): VQ 码本大小。
            codebook_dim (int): VQ 嵌入维度（即 encoder 最终输出通道数）。
            commitment_weight (float): VQ commitment loss 权重。
            orthogonal_reg_weight (float): 正交正则化权重。
            codebook_diversity_loss_weight (float): 码本多样性损失权重。
            cnn_type (int): CNN 架构类型。
                - 0: 默认大模型（1 → 64 → 128 → codebook_dim）
                - 1: 严格对称小模型（1 → 16 → 32 → 64），此时 codebook_dim 必须为 64
        """
        super().__init__()

        # 设置 codebook_dim 根据 cnn_type
        if cnn_type == 0:
            codebook_dim = 256
        elif cnn_type == 1:
            codebook_dim = 64
        elif cnn_type == 2:
            codebook_dim = 512  # 固定为 512，与你提供的结构一致
        else:
            raise ValueError(f"Unsupported cnn_type: {cnn_type}. Supported: 0, 1, or 2.")

        self.codebook_dim = codebook_dim
        self.cnn_type = cnn_type
        self.latent_dim = codebook_dim
        self.codebook_size = codebook_size
        print(f"codebook_dim:{codebook_dim}")
        # 构建 encoder 和 decoder
        if cnn_type == 0:
            self._build_encoder_type0()
            self._build_decoder_type0()
            self.cnn_stride = 5   # 总下采样率（仅最后一层 stride=5）
            self.RF = 33          # 感受野（采样点），对应 ~1 个 RNA 碱基
        elif cnn_type == 1:
            self._build_encoder_type1()
            self._build_decoder_type1()
            self.cnn_stride = 5   # 总下采样率（仅最后一层 stride=5）
            self.RF = 33          # 感受野（采样点），对应 ~1 个 RNA 碱基
        elif cnn_type == 2:
            self._build_encoder_type2()
            self._build_decoder_type2()
            self.cnn_stride = 12  # 1 * 1 * 3 * 2 * 2
            self.RF = 65  # 
        else:
            raise ValueError(f"Unsupported cnn_type: {cnn_type}. Supported: 0 or 1.")


        # ======================================================================
        # VECTOR QUANTIZATION (VQ)
        # ======================================================================
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.vq = VectorQuantize(
            dim=self.latent_dim,
            codebook_size=codebook_size,
            kmeans_init=True,
            kmeans_iters=10,
            decay=0.99,
            threshold_ema_dead_code=2,
            commitment_weight=commitment_weight,
            codebook_diversity_loss_weight=codebook_diversity_loss_weight,
            orthogonal_reg_weight=orthogonal_reg_weight,
            orthogonal_reg_max_codes=256,
            orthogonal_reg_active_codes_only=True,
        )
        if rank == 0:
            self._print_vq_config()


    def _print_vq_config(self) -> None:
        """打印 VQ 配置信息（仅 rank 0）"""
        print("Intialized VectorQuantize with the following hyperparameters:")
        print(f"  dim: {self.latent_dim}")
        print(f"  codebook_size: {self.codebook_size}")
        print(f"  kmeans_init: True")
        print(f"  kmeans_iters: 10")
        print(f"  decay: 0.99")
        print(f"  threshold_ema_dead_code: 2")
        print(f"  commitment_weight: {self.vq.commitment_weight}")
        print(f"  codebook_diversity_loss_weight: {self.vq.codebook_diversity_loss_weight}")
        print(f"  orthogonal_reg_weight: {self.vq.orthogonal_reg_weight}")
        print(f"  orthogonal_reg_max_codes: 256")
        print(f"  orthogonal_reg_active_codes_only: True")
        print(f"  cnn_type: {self.cnn_type}")
        print("-" * 60)

    # ────────────────────────────────────────────────
    # ENCODER BUILDERS
    # ────────────────────────────────────────────────

    def _build_encoder_type0(self) -> None:
        """构建 cnn_type=0 的 encoder：1 → 64 → 128 → latent_dim（如 256）"""
        self.encoder = nn.Sequential(
            # Layer 1: 超局部特征提取（无下采样）
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 2: 局部上下文聚合（无下采样）
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 3: 下采样 + 升维至 latent space（RF=33, stride=5）
            nn.Conv1d(128, self.latent_dim, kernel_size=25, stride=5, padding=12, bias=False),
            nn.BatchNorm1d(self.latent_dim),
        )

    def _build_encoder_type1(self) -> None:
        """构建 cnn_type=1 的 encoder：1 → 16 → 32 → 64（严格对称）"""
        self.encoder = nn.Sequential(
            # Layer 1: 1 → 16
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.SiLU(),

            # Layer 2: 16 → 32
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.SiLU(),

            # Layer 3: 32 → 64, stride=5, RF=33
            nn.Conv1d(32, 64, kernel_size=25, stride=5, padding=12, bias=False),
            nn.BatchNorm1d(64),
        )
    def _build_encoder_type2(self) -> None:
        """cnn_type=2: 多阶段下采样，总 stride=12，输出通道=512"""
        self.encoder = nn.Sequential(
            # Layer 1: 1 → 64, stride=1
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 2: 64 → 64, stride=1
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Layer 3: 64 → 128, stride=3
            nn.Conv1d(64, 128, kernel_size=9, stride=3, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 4: 128 → 128, stride=2
            nn.Conv1d(128, 128, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Layer 5: 128 → 512, stride=2
            nn.Conv1d(128, self.latent_dim, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(self.latent_dim),
        )
    # ────────────────────────────────────────────────
    # DECODER BUILDERS
    # ────────────────────────────────────────────────

    def _build_decoder_type0(self) -> None:
        """构建 cnn_type=0 的 decoder（近似对称，高维 refine）"""
        self.decoder = nn.Sequential(
            # Upsample ×5: 逆操作 encoder 最后一层
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

            # Refine layer: 消除棋盘伪影
            nn.Conv1d(128, 64, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Final projection to raw signal
            nn.Conv1d(64, 1, kernel_size=5,padding=2,bias=True),
        )

    def _build_decoder_type1(self) -> None:
        """构建 cnn_type=1 的 decoder（严格对称：64 → 32 → 16 → 1）"""
        self.decoder = nn.Sequential(
            # Inverse of encoder Layer 3: 64 → 32
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

            # Inverse of encoder Layer 2: 32 → 16
            nn.Conv1d(32, 16, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(16),
            nn.SiLU(),

            # Inverse of encoder Layer 1: 16 → 1
            nn.Conv1d(16, 1, kernel_size=5, padding=2,bias=True)
        )
    def _build_decoder_type2(self) -> None:
        """严格对称 decoder: 512 → 128 → 128 → 64 → 64 → 1，上采样顺序与 encoder 下采样逆序对应"""
        self.decoder = nn.Sequential(
            # Inverse of encoder Layer 5: 512 → 128, upsample ×2
            nn.ConvTranspose1d(512, 128, kernel_size=5, stride=2, padding=2, output_padding=0,bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Inverse of encoder Layer 4: 128 → 128, upsample ×2
            nn.ConvTranspose1d(128, 128, kernel_size=9, stride=2, padding=4, output_padding=0,bias=False),
            nn.BatchNorm1d(128),
            nn.SiLU(),

            # Inverse of encoder Layer 3: 128 → 64, upsample ×3
            nn.ConvTranspose1d(128, 64, kernel_size=9, stride=3, padding=4, output_padding=0,bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Inverse of encoder Layer 2: 64 → 64
            nn.Conv1d(64, 64, kernel_size=5, padding=2,bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),

            # Inverse of encoder Layer 1: 64 → 1
            nn.Conv1d(64, 1, kernel_size=5,padding=2,bias=True)
        )
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入信号，形状 [B, 1, T]

        Returns:
            recon (torch.Tensor): 重建信号，[B, 1, T]
            indices (torch.Tensor): VQ 离散 token，[B, T//5]
            loss (torch.Tensor): VQ 总损失（标量）
            loss_breakdown (dict): 损失分项（commitment, diversity, ortho...）
        """
        # Encode: [B, 1, T] → [B, C, T//5]
        z_continuous = self.encoder(x)

        # Permute for VQ: [B, C, N] → [B, N, C]
        z_permuted = z_continuous.permute(0, 2, 1)

        # Quantize
        z_quantized_permuted, indices, loss, loss_breakdown = self.vq(
            z_permuted, return_loss_breakdown=True
        )

        # Back to [B, C, N] for decoder
        z_quantized = z_quantized_permuted.permute(0, 2, 1)

        # Decode
        recon = self.decoder(z_quantized)

        # Length alignment: ensure recon length == input length
        target_len = x.shape[-1]
        current_len = recon.shape[-1]
        if current_len > target_len:
            recon = recon[..., :target_len]
        elif current_len < target_len:
            recon = F.pad(recon, (0, target_len - current_len))

        return recon, indices, loss, loss_breakdown
