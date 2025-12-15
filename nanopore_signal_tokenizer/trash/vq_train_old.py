import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 相对导入核心组件
from .dataset import NanoporeSignalDataset
from .vq_model import NanoporeVQModel


def vq_train(
    npy_dir: str,
    output_model_path: str,
    batch_size: int = 16,
    lr: float = 3e-4,
    num_epochs: int = 10,
    codebook_size: int = 8192,
    chunk_size: int = 12000,
    num_workers: int = 8,
    val_size: int = 100,
    do_evaluate: bool = True,
    commitment_weight: float = 0.25
):
    """
    分布式训练 Nanopore VQ tokenizer。
    现在会分别打印：重建损失、commitment 损失、总损失。
    """
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_device_id)
    device = f"cuda:{local_device_id}"

    if rank == 0:
        print(f"🚀 Using {world_size} GPUs for training.")
        print(f"📂 Data directory: {npy_dir}")
        print(f"💾 Model will be saved to: {output_model_path}")
        print(f"⚙️  Hyperparameters: "
              f"batch_size={batch_size}, lr={lr}, epochs={num_epochs}, "
              f"codebook_size={codebook_size}, chunk_size={chunk_size}, "
              f"do_evaluate={do_evaluate}")

    # ========== 数据加载 ==========
    dataset = NanoporeSignalDataset(npy_dir=npy_dir, expected_chunk_len=chunk_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    # ========== 可选：验证集（仅用于评估）==========
    val_loader = None
    if do_evaluate:
        val_dataset = NanoporeSignalDataset(npy_dir=npy_dir, expected_chunk_len=chunk_size)
        actual_val_size = min(val_size, len(val_dataset))
        indices = np.random.choice(len(val_dataset), size=actual_val_size, replace=False)
        val_subset = torch.utils.data.Subset(val_dataset, indices)
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(2, num_workers // 2),
            pin_memory=True
        )

    # ========== 模型与优化器 ==========
    model = NanoporeVQModel(codebook_size=codebook_size,commitment_weight = commitment_weight).to(device)
    model = DDP(model, device_ids=[local_device_id])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ========== 评估函数（仅在 do_evaluate=True 时调用）==========
    def evaluate_codebook_usage():
        model.eval()
        used_codes = set()
        total_tokens = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                _, indices, _ = model(x)
                indices = indices.cpu().numpy().flatten()
                used_codes.update(indices.tolist())
                total_tokens += indices.size
        usage_ratio = len(used_codes) / codebook_size
        model.train()
        return usage_ratio, total_tokens

    # ========== 训练循环 ==========
    model.train()
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)

        # 分别记录三种损失
        total_recon_loss = torch.tensor(0.0, device=device)
        total_commit_loss = torch.tensor(0.0, device=device)
        total_total_loss = torch.tensor(0.0, device=device)
        num_batches = torch.tensor(len(dataloader), device=device)

        pbar = tqdm(dataloader, desc=f"Rank {rank} | Epoch {epoch+1}/{num_epochs}", disable=(rank != 0))
        for batch in pbar:
            x = batch.to(device)
            # commit_loss 是否已包含 commitment_weight？
            # 在 vector_quantize_pytorch 中，返回的 commit_loss 已经是乘过 commitment_weight 的（默认 0.25）
            # 因为 VectorQuantize 返回的 commit_loss 是：
            # commit_loss = (z_e - e_k.detach()).pow(2).mean() * self.commitment_weight
            # 它是一个 requires_grad=False 的 scalar tensor，位于与输入相同的设备上（GPU）。
            # 所以 commit_loss 本身就是 GPU tensor，不需要 .item()。
            recon, indices, commit_loss = model(x)
            # 如果你想弱化重建、强调离散表示质量，可以加一个超参数：
            # recon_weight = 0.01  # << 降低重建权重
            # loss = recon_weight * F.mse_loss(recon, x) + commit_loss
            # 这样模型会更关注“编码器贴紧码本”和“码本分散”，而不是像素级还原信号——非常适合做 tokenizer。
            recon_loss = F.mse_loss(recon, x)
            total_loss = recon_loss + commit_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 累加各部分损失（注意：commit_loss 是标量 tensor）
            total_recon_loss += recon_loss
            total_commit_loss += commit_loss
            total_total_loss += total_loss

        # 聚合所有 GPU 的损失（求和）
        dist.all_reduce(total_recon_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_commit_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)

        # 计算平均损失（只在 rank 0 打印）
        avg_recon = total_recon_loss.item() / num_batches.item()
        avg_commit = total_commit_loss.item() / num_batches.item()
        avg_total = total_total_loss.item() / num_batches.item()

        if rank == 0:
            if do_evaluate and epoch < num_epochs - 1:
                usage_ratio, total_tokens = evaluate_codebook_usage()
                print(
                    f"Epoch {epoch+1} - "
                    f"Recon Loss: {avg_recon:.6f} | "
                    f"Commit Loss: {avg_commit:.6f} | "
                    f"Total Loss: {avg_total:.6f} | "
                    f"Codebook Usage: {usage_ratio:.1%} (tokens={total_tokens:,})"
                )
            else:
                print(
                    f"Epoch {epoch+1} - "
                    f"Recon Loss: {avg_recon:.6f} | "
                    f"Commit Loss: {avg_commit:.6f} | "
                    f"Total Loss: {avg_total:.6f}"
                )

    # 保存模型（仅 rank 0）
    if rank == 0:
        torch.save(model.module.state_dict(), output_model_path)
        print(f"✅ Model saved to {output_model_path}")

    dist.barrier()
    dist.destroy_process_group()
