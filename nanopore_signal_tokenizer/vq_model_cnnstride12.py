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

    def __init__(self, codebook_size=8192, codebook_dim= 64, commitment_weight=1.0,orthogonal_reg_weight=1.0,codebook_diversity_loss_weight=1.0):
        super().__init__()
        self.latent_dim = codebook_dim  # VQ embedding 维度，也是 encoder 最终输出通道数
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
        encoder_layers.append(nn.Conv1d(32, self.latent_dim, kernel_size=25, stride=12, padding=12, bias=True))
        encoder_layers.append(nn.Tanh())          # 推荐：避免 VQ 输入动态范围过大
        encoder_layers.append(nn.BatchNorm1d(self.latent_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.cnn_stride = 1 * 1 * 12  # = 12
        self.margin_stride_count = 12
        self.RF = 33
        # ======================================================================
        # VECTOR QUANTIZATION (VQ)
        # 在 64 维空间中离散化连续表示，生成可学习的 codebook
        # ======================================================================
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        if rank == 0:
            print("Intialized VectorQuantize with the following hyperparameters:")
            print(f"  dim: {self.latent_dim}")
            print(f"  codebook_size: {codebook_size}")
            print(f"  kmeans_init: True")
            print(f"  kmeans_iters: 10")
            print(f"  decay: 0.99")
            print(f"  threshold_ema_dead_code: 2")
            print(f"  commitment_weight: {commitment_weight}")
            print(f"  codebook_diversity_loss_weight: {codebook_diversity_loss_weight}")
            print(f"  orthogonal_reg_weight: {orthogonal_reg_weight}")
            print(f"  orthogonal_reg_max_codes: 256")
            print(f"  orthogonal_reg_active_codes_only: True")
            print("-" * 60)
        self.vq = VectorQuantize(
            dim=self.latent_dim,
            codebook_size=codebook_size,
            kmeans_init=True,           # 启动时用 K-Means 初始化 codebook
            kmeans_iters=10,
            decay=0.99,                 # EMA 更新 codes
            threshold_ema_dead_code=2,  # 激活低频 codes
            commitment_weight=commitment_weight,  # 控制 z_e 与 e 的对齐强度
            codebook_diversity_loss_weight = codebook_diversity_loss_weight,
            orthogonal_reg_weight = orthogonal_reg_weight,                 # in paper, they recommended a value of 10
            orthogonal_reg_max_codes = 256,             
            # this would randomly sample from the codebook for the orthogonal regularization loss, for limiting memory usage
            orthogonal_reg_active_codes_only = True
            # 每次计算正交损失时，最多使用 256 个码向量；如果当前 batch 激活的唯一码 ≤256，则全部使用；否则随机采样 256 个。
            # 当 orthogonal_reg_active_codes_only=True 时，正交正则化只作用于当前 batch 中实际被“使用”（即被匹配到）的码本向量，而不是整个码本。
            # set this to True if you have a very large codebook, and would only like to enforce the loss on the activated codes per batch
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
                stride=12,               # 与 encoder 相同
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
        # 在 PyTorch 中，当你对一个 nn.Module 子类的实例（比如 self.vq）使用 函数调用语法：
        # output = self.vq(input)
        # 这实际上等价于：
        # output = self.vq.forward(input)
        z_quantized_permuted, indices, loss,loss_breakdown = self.vq(z_permuted,return_loss_breakdown=True)

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

        return recon, indices, loss, loss_breakdown
