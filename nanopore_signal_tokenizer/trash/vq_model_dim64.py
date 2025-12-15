import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize


class NanoporeVQModel(nn.Module):
    """
    轻量版 Nanopore VQ Tokenizer（dim=64 + 简化通道）：
    - Encoder: 4 层 Conv，所有中间通道=64，总 stride=12
    - VQ: 在 64 维空间离散化
    - Decoder: 对称上采样重建信号
    
    目标：提取干净、低维、k-mer 可分的骨干信号，适配 LLM。
    """
    def __init__(self, codebook_size=8192, commitment_weight=2.0):
        super().__init__()
        dim = 64  # 核心 latent 维度

        # ========== 简化版 Encoder（通道统一为 64） ==========
        encoder_layers = []

        # Layer 1: 提取局部特征（无下采样）
        encoder_layers.append(nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias=True))
        encoder_layers.append(nn.SiLU())
        encoder_layers.append(nn.BatchNorm1d(64))

        # Layer 2: 下采样 stride=3 → T/3
        encoder_layers.append(nn.Conv1d(64, 64, kernel_size=9, stride=3, padding=4, bias=True))
        encoder_layers.append(nn.SiLU())
        encoder_layers.append(nn.BatchNorm1d(64))

        # Layer 3: 下采样 stride=2 → T/6
        encoder_layers.append(nn.Conv1d(64, 64, kernel_size=9, stride=2, padding=4, bias=True))
        encoder_layers.append(nn.SiLU())
        encoder_layers.append(nn.BatchNorm1d(64))

        # Layer 4: 下采样 stride=2 → T/12，输出 dim=64
        encoder_layers.append(nn.Conv1d(64, dim, kernel_size=5, stride=2, padding=2, bias=True))
        encoder_layers.append(nn.SiLU())
        encoder_layers.append(nn.BatchNorm1d(dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.cnn_stride = 1 * 3 * 2 * 2  # = 12
        self.margin_stride_count = 4     # 现在只有 4 个卷积层参与 stride

        # ========== VQ with dim=64 ==========
        self.vq = VectorQuantize(
            dim=dim,
            codebook_size=codebook_size,
            kmeans_init=True,
            kmeans_iters=10,
            decay=0.99,
            threshold_ema_dead_code=2,
            commitment_weight=commitment_weight
        )

        # ========== Decoder: 上采样 ×12（对称设计） ==========
        self.decoder = nn.Sequential(
            # Upsample 1: ×2 → T/6
            nn.ConvTranspose1d(dim, 64, kernel_size=8, stride=2, padding=3),
            nn.SiLU(),
            nn.BatchNorm1d(64),

            # Upsample 2: ×2 → T/3
            nn.ConvTranspose1d(64, 64, kernel_size=12, stride=2, padding=5),
            nn.SiLU(),
            nn.BatchNorm1d(64),

            # Upsample 3: ×3 → T
            nn.ConvTranspose1d(64, 64, kernel_size=18, stride=3, padding=8),
            nn.SiLU(),
            nn.BatchNorm1d(64),

            # Final projection to raw signal
            nn.Conv1d(64, 1, kernel_size=1),
        )

    def forward(self, x):
        # Encode: [B, 1, T] → [B, 64, T//12]
        z = self.encoder(x)

        # Permute for VQ: [B, T_enc, 64]
        z_transposed = z.permute(0, 2, 1)

        # Quantize
        z_q_transposed, indices, commit_loss = self.vq(z_transposed)

        # Back to [B, 64, T_enc]
        z_q = z_q_transposed.permute(0, 2, 1)

        # Decode: [B, 64, T//12] → [B, 1, T_rec ≈ T]
        recon = self.decoder(z_q)

        # Align length with input
        target_len = x.shape[2]
        if recon.shape[2] > target_len:
            recon = recon[:, :, :target_len]
        elif recon.shape[2] < target_len:
            pad = target_len - recon.shape[2]
            recon = F.pad(recon, (0, pad))

        return recon, indices, commit_loss
