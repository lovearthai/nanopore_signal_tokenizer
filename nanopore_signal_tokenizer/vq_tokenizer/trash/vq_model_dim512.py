import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
# ----------------------------
# 合并后的完整 Tokenizer 模型（Encoder + RVQ + Decoder）
# ----------------------------
class NanoporeVQModel(nn.Module):
    """
    完整的自编码器结构：
    - Encoder: 压缩信号（原 NanoporeEncoder 内容内联）
    - RVQ: 将连续 latent 离散化为 tokens
    - Decoder: 从 tokens 重建原始信号（用于自监督训练）
    """
    def __init__(self, codebook_size=8192, commitment_weight=0.25):
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

        self.vq = VectorQuantize(
            dim=dim,                        # 输入向量的维度（例如 512）
            codebook_size=codebook_size,    # 码本中离散 token 的数量（如 8192）
    
            # —————— 初始化相关 ——————
            kmeans_init=True,               # ✅ 是否用 K-Means 初始化码本（强烈推荐开启）
                                    #   - 默认随机初始化容易导致码本 collapse（部分 code 从未被使用）
                                    #   - 开启后：在第一个 forward 时用前几个 batch 的 encoder 输出做 K-Means 聚类，
                                    #     将聚类中心作为初始码本，大幅提升训练稳定性与码本利用率。
    
            kmeans_iters=10,                # K-Means 初始化时的迭代次数（默认 10 足够）
    
            # —————— 码本更新机制（EMA） ——————
            decay=0.99,                     # 码本向量使用 EMA（指数移动平均）更新时的衰减系数
                                    #   - 更新公式：codebook = decay * old_codebook + (1 - decay) * new_embedding
                                    #   - 值越接近 1（如 0.99 或 0.999），码本更新越慢、越稳定
                                    #   - 太小（如 <0.9）会导致码本震荡；太大（如 >0.999）可能难以适应数据分布
    
            threshold_ema_dead_code=2,      # “死亡”码字复活阈值
                                    #   - 如果某个 code 在过去若干 steps 中被使用的总次数 < 此值，
                                    #     则认为它是“dead code”（未被充分利用）
                                    #   - 系统会自动将其替换为当前 batch 中最常出现但未被良好编码的 embedding，
                                    #     从而提升码本多样性（防止 collapse）
                                    #   - 对于大码本（如 8192+），建议设为 1~5；小码本可设为 1
    
            # —————— 损失函数权重 ——————
            commitment_weight=commitment_weight          # ⭐⭐⭐ 核心参数：commitment loss 的权重
# ⭐ commitment_weight 参数详解（核心超参）：
# - 该参数控制 commitment loss 的权重，用于解决 VQ 中编码器与码本之间的不对称优化问题。
# - 在标准 VQ-VAE 中，总损失为：reconstruction_loss + commitment_weight * ||z_e - sg(e_k)||^2。
# - 其中 z_e 是编码器输出的连续向量，e_k 是最接近的码本向量，sg(·) 表示 stop-gradient（不更新码本）。
# - commitment loss 仅对编码器施加梯度，迫使 z_e 向最近的码本靠拢，从而提升量化效率和码本利用率。
# - 若不加此 loss，编码器可能输出远离所有码字的向量，导致量化误差大或码本 collapse（大量 code 从未被使用）。
# - 在 vector_quantize_pytorch 库中，返回的 commit_loss 已经内部乘上了 commitment_weight，
#   因此在训练时应直接写：total_loss = recon_loss + commit_loss，无需再手动乘权重。
# - 推荐初始值为 0.25（来自 SoundStream 和 VQ-VAE 的常用设置），安全探索范围为 0.1 ~ 0.5。
# - 调参指南：
#     • 如果训练后码本利用率低（例如 <60%），说明编码器未充分使用码本，可增大该值（如 0.5）；
#     • 如果重建信号严重失真或模糊，说明离散化过强，可减小该值（如 0.1）；
#     • 若目标是构建高质量 tokenizer（而非高保真重建），可适当提高该权重以强化离散表示的稳定性与多样性。
# - 该参数需与 kmeans_init=True 和 threshold_ema_dead_code 配合使用，以最大化码本激活率。
#值范围	效果
#太小（<0.1）	编码器不关心码本 → 码本利用率低、重建差、训练不稳定
#太大（>1.0）	编码器过度压缩到码本 → 表达能力受限、重建模糊、可能欠拟合
#推荐值	0.25（SoundStream / VQ-VAE 常用值）
#0.1 ~ 0.5 是安全探索区间
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
        z_q_transposed, indices, commit_loss = self.vq(z_transposed)

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
         # 返回：重建信号、离散索引、commitment loss
        return recon, indices, commit_loss
