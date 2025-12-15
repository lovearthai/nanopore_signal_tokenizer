import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize


class NanoporeVQModel(nn.Module):
    """
    Nanopore VQ Tokenizer for Direct RNA Sequencing (130 bps, 4 kHz)
    
    设计目标：
    - 感受野 ≈ 33 采样点（≈1 个 RNA 碱基）
    - 总下采样率 = 5×（每碱基 ≈6 个 tokens，高分辨率）
    - 通道数渐进增长：1 → 16 → 32 → 64
    - 输出 64 维 latent，直接用于 VQ（避免额外投影）
    - Decoder 严格对称于 encoder 的下采样操作（仅逆最后一层）

    适用于：VQ tokenizer + LLM basecalling pipeline
    """

    def __init__(self, codebook_size=8192, commitment_weight=2.0):
        super().__init__()
        self.latent_dim = 64  # VQ embedding 维度，也是 encoder 最终输出通道数
        # ======================================================================
        # ENCODER: 3 层 Conv1D，逐步提取局部 squiggle 特征
        # 总 stride = 1 * 1 * 5 = 5
        # 感受野 = 5 → 9 → 33（计算见下方）
        # ======================================================================
        encoder_layers = []

        # ── Layer 1: 提取超局部特征（无下采样）
        #   输入: [B, 1, T]
        #   kernel=5, stride=1, padding=2 → 输出长度 = T
        #   通道: 1 → 16
        encoder_layers.append(nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2, bias=True))
        encoder_layers.append(nn.SiLU())          # Swish 的 PyTorch 实现
        encoder_layers.append(nn.BatchNorm1d(16))

        # ── Layer 2: 聚合局部上下文（仍无下采样）
        #   kernel=5, stride=1, padding=2 → 输出长度 = T
        #   通道: 16 → 32
        encoder_layers.append(nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=True))
        encoder_layers.append(nn.SiLU())
        encoder_layers.append(nn.BatchNorm1d(32))

        # ── Layer 3: 关键下采样层（stride=5），同时升维至 latent_dim
        #   kernel=25, stride=5, padding=12 → 输出长度 ≈ T // 5
        #   感受野计算:
        #       RF1 = 1 + (5-1)*1 = 5
        #       RF2 = 5 + (5-1)*1 = 9
        #       RF3 = 9 + (25-1)*1 = 33  ← ≈1 个 RNA 碱基 (4000/130 ≈ 31)
        #   通道: 32 → 64
        #   注意: 使用 Tanh 而非 SiLU —— 限制输出范围，利于 VQ 稳定训练
        encoder_layers.append(nn.Conv1d(32, self.latent_dim, kernel_size=25, stride=5, padding=12, bias=True))
        encoder_layers.append(nn.Tanh())          # 推荐：避免 VQ 输入动态范围过大
        encoder_layers.append(nn.BatchNorm1d(self.latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.cnn_stride = 1 * 1 * 5  # = 5
        self.margin_stride_count = 5
        self.RF = 33
        # ======================================================================
        # VECTOR QUANTIZATION (VQ)
        # 在 64 维空间中离散化连续表示，生成可学习的 codebook
        # ======================================================================
        self.vq = VectorQuantize(
            dim=self.latent_dim,
            codebook_size=codebook_size,
            kmeans_init=True,           # 启动时用 K-Means 初始化 codebook
            kmeans_iters=10,
            decay=0.99,                 # EMA 更新 codes
            threshold_ema_dead_code=2,  # 激活低频 codes
            commitment_weight=commitment_weight  # 控制 z_e 与 e 的对齐强度
        )

        # ======================================================================
        # DECODER: 对称重建原始信号
        # 只需逆操作 encoder 的最后一层（因为前两层无下采样）
        # 使用 ConvTranspose1d(kernel=25, stride=5, padding=12) 严格对称
        # 额外添加 refine 层以消除上采样伪影
        # ======================================================================
        self.decoder = nn.Sequential(
            # ── Upsample ×5: 逆操作 encoder Layer 3
            #    输入: [B, 64, T//5]
            #    输出: [B, 64, ≈T]
            nn.ConvTranspose1d(
                in_channels=self.latent_dim,
                out_channels=64,
                kernel_size=25,         # 与 encoder 最后一层相同
                stride=5,               # 与 encoder 相同
                padding=12,             # 与 encoder 相同（保证中心对齐）
                output_padding=0,       # 若长度偏差 ≤1，靠 final pad 补偿
                bias=False
            ),
            nn.SiLU(),
            nn.BatchNorm1d(64),

            # ── Refine Layer: 消除棋盘伪影（checkerboard artifacts）
            #    这是 ConvTranspose 后的标准做法
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.BatchNorm1d(64),

            # ── Final Projection: 回归到原始信号维度
            nn.Conv1d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, 1, T] —— 标准化后的原始电流信号（pA）

        Returns:
            recon:     [B, 1, T] —— 重建信号
            indices:   [B, T//5] —— VQ 离散 token 序列（用于 LLM）
            commit_loss: scalar —— VQ commitment loss
        """
        # ── Encode to continuous latent
        #    [B, 1, T] → [B, 64, T_enc], where T_enc = T // 5
        z_continuous = self.encoder(x)

        # ── Permute for VQ: vector_quantize_pytorch expects [B, N, D]
        #    [B, 64, T_enc] → [B, T_enc, 64]
        z_permuted = z_continuous.permute(0, 2, 1)

        # ── Quantize
        z_quantized_permuted, indices, commit_loss = self.vq(z_permuted)

        # ── Back to [B, 64, T_enc] for decoder
        z_quantized = z_quantized_permuted.permute(0, 2, 1)

        # ── Decode to reconstructed signal
        recon = self.decoder(z_quantized)  # [B, 1, T_rec]

        # ── Length alignment: ensure recon length == input length
        target_len = x.shape[2]
        current_len = recon.shape[2]

        if current_len > target_len:
            recon = recon[:, :, :target_len]
        elif current_len < target_len:
            pad = target_len - current_len
            recon = F.pad(recon, (0, pad))  # right-pad with zeros

        return recon, indices, commit_loss
