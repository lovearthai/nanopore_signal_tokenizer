import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ

# ----------------------------
# 合并后的完整 Tokenizer 模型（Encoder + RVQ + Decoder）
# ----------------------------
class NanoporeRVQModel(nn.Module):
    """
    完整的自编码器结构：
    - Encoder: 压缩信号（原 NanoporeEncoder 内容内联）
    - RVQ: 将连续 latent 离散化为 tokens
    - Decoder: 从 tokens 重建原始信号（用于自监督训练）
    """
    def __init__(self, n_q=4, codebook_size=8192):
        super().__init__()
        dim = 512

        # ========== 内联原 NanoporeEncoder 的结构 ==========
        encoder_layers = []

        # Layer 1: 卷积层
        encoder_layers.append(nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2, bias=True))
        encoder_layers.append(nn.SiLU())
        encoder_layers.append(nn.BatchNorm1d(64))

        # Layer 2
        encoder_layers.append(nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=2, bias=True))
        encoder_layers.append(nn.SiLU())
        encoder_layers.append(nn.BatchNorm1d(64))

        # Layer 3: 下采样 stride=3
        encoder_layers.append(nn.Conv1d(64, 128, kernel_size=9, stride=3, padding=4, bias=True))
        encoder_layers.append(nn.SiLU())
        encoder_layers.append(nn.BatchNorm1d(128))

        # Layer 4: stride=2
        encoder_layers.append(nn.Conv1d(128, 128, kernel_size=9, stride=2, padding=4, bias=True))
        encoder_layers.append(nn.SiLU())
        encoder_layers.append(nn.BatchNorm1d(128))

        # Layer 5: stride=2 → 输出通道 512
        encoder_layers.append(nn.Conv1d(128, dim, kernel_size=5, stride=2, padding=2, bias=True))
        encoder_layers.append(nn.SiLU())
        encoder_layers.append(nn.BatchNorm1d(dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.cnn_stride = 1 * 1 * 3 * 2 * 2  # = 12
        self.margin_stride_count = 5
        # ========== Residual Vector Quantization (RVQ) ==========
        from vector_quantize_pytorch import ResidualVQ
        self.rvq = ResidualVQ(
            num_quantizers=n_q,
            dim=dim,
            codebook_size=codebook_size,
            kmeans_init=True,           # 更稳定训练
            kmeans_iters=10,
            threshold_ema_dead_code=2   # 防止码本死亡
        )

        # ========== Decoder: 上采样 ×12 ==========
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

    def forward(self, x):
        # 编码：[B, 1, T] → [B, 512, T_enc]
        z = self.encoder(x)

        # 转置为 [B, T_enc, 512] —— 符合 vector_quantize_pytorch 的输入要求
        z_transposed = z.permute(0, 2, 1)

        # 量化：得到离散 token indices
        z_q_transposed, indices, _ = self.rvq(z_transposed)

        # 转回 [B, 512, T_enc] 用于解码
        z_q = z_q_transposed.permute(0, 2, 1)

        # 解码重建：[B, 512, T_enc] → [B, 1, T_rec]
        recon = self.decoder(z_q)

        # 对齐原始信号长度
        target_len = x.shape[2]
        if recon.shape[2] > target_len:
            recon = recon[:, :, :target_len]
        elif recon.shape[2] < target_len:
            pad = target_len - recon.shape[2]
            recon = F.pad(recon, (0, pad))

        # indices shape: [B, T_enc, n_q]
        return recon, indices
