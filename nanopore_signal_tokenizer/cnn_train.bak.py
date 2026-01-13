import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import csv
import time
from typing import Dict, List

# 相对导入
from .dataset import NanoporeSignalDataset
from .cnn_model import NanoporeCNNModel  # ← 替换为 CNN 模型

# ========== 打印训练参数 ==========
def print_training_args(**kwargs):
    from pprint import pformat
    print("\n" + "="*60)
    print(" 🚀 Starting CNN Autoencoder Training with the following configuration:")
    print("="*60)
    print(pformat(kwargs, width=100, sort_dicts=False))
    print("="*60 + "\n")


# ========== 保存完整检查点 ==========
def save_full_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    rank: int
):
    if rank != 0:
        return

    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        'numpy_rng_state': np.random.get_state(),
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, path)
    print(f"✅ Full checkpoint saved to {path}")


# ========== 日志与 CSV 保存 ==========
def log_and_save(
    epoch: int,
    step: int,
    total_epochs: int,
    total_steps: int,
    epoch_start_time: float,
    epoch_total_steps: int,
    avg_recon_loss: float,
    lr: float,
    loss_csv_path: str,
):
    import time

    current_time = time.time()
    elapsed_seconds = current_time - epoch_start_time
    steps_done = step % epoch_total_steps or 1
    avg_time_per_step = elapsed_seconds / steps_done
    remaining_seconds = avg_time_per_step * max(0, epoch_total_steps - steps_done)

    def format_hms(seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

    elapsed_str = format_hms(elapsed_seconds)
    remaining_str = format_hms(remaining_seconds)

    epoch_width = len(str(total_epochs))
    step_width = len(str(total_steps))

    print(
        f"[Epoch {epoch+1:>{epoch_width}}/{total_epochs} | "
        f"Step {step:>{step_width}}/{total_steps} | "
        f"{elapsed_str}<{remaining_str}] "
        f"Recon Loss: {avg_recon_loss:>8.6f} | "
        f"LR: {lr:>7.2e} |"
    )

    with open(loss_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([epoch + 1, step, avg_recon_loss, lr])


# ========== 主训练函数 ==========
def cnn_train(
    npy_dir: str,
    output_model_path: str,
    batch_size: int = 16,
    lr: float = 1e-4,
    num_epochs: int = 10,
    chunk_size: int = 12000,
    num_workers: int = 8,
    prefetch_factor: int = 128,
    val_ratio: float = 0.1,
    do_evaluate: bool = True,
    loss_log_interval: int = 10,
    loss_csv_path: str = "cnn_train_loss.csv",
    use_wandb: bool = True,
    wandb_project: str = "nanopore_cnn",
    wandb_name: str = "default_cnn_run",
    lr_scheduler_type: str = "cosine",
    warmup_steps: int = 1000,
    warmup_start_factor: float = 1e-6,
    warmup_end_factor: float = 1.0,
    main_scheduler_end_factor: float = 1e-5,
    save_checkpoint_every_epoch: int = 1,
    checkpoint_path: str = None,
    cnn_type: int = 1,
):
    print_training_args(
        npy_dir=npy_dir,
        output_model_path=output_model_path,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        chunk_size=chunk_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        val_ratio=val_ratio,
        do_evaluate=do_evaluate,
        loss_csv_path=loss_csv_path,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        cnn_type=cnn_type,
        save_checkpoint_every_epoch=save_checkpoint_every_epoch,
    )

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    # 初始化分布式
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_device_id)
    device = f"cuda:{local_device_id}"

    # WandB（仅 rank 0）
    if rank == 0 and use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                "batch_size": batch_size,
                "lr": lr,
                "num_epochs": num_epochs,
                "chunk_size": chunk_size,
                "cnn_type": cnn_type,
                "world_size": world_size,
            }
        )
    else:
        wandb = None

    if rank == 0:
        print(f"🚀 Using {world_size} GPUs.")
        print(f"📂 Data: {npy_dir}")
        print(f"💾 Model saved to: {output_model_path}")

        # 初始化 CSV
        with open(loss_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'recon_loss', 'lr'])

    # 数据集
    dataset = NanoporeSignalDataset(shards_dir=npy_dir)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        drop_last=True
    )

    # 验证集（仅 rank 0 创建）
    val_loader = None
    if do_evaluate and rank == 0:
        actual_val_size = max(1, int(val_ratio * len(dataset)))
        np.random.seed(42)
        indices = np.random.choice(len(dataset), size=actual_val_size, replace=False)
        val_subset = torch.utils.data.Subset(dataset, indices)
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(2, num_workers // 2),
            pin_memory=True
        )

    # 模型
    model = NanoporeCNNModel(cnn_type=cnn_type).to(device)
    model = DDP(model, device_ids=[local_device_id])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 学习率调度器
    total_training_steps = len(dataloader) * num_epochs
    scheduler = None

    if rank == 0 and lr_scheduler_type != "constant":
        print(f"📈 LR Scheduler: {lr_scheduler_type}, warmup={warmup_steps}")

    if lr_scheduler_type != "constant":
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        warmup_scheduler = LinearLR(optimizer, start_factor=warmup_start_factor, end_factor=warmup_end_factor, total_iters=warmup_steps)
        main_steps = max(1, total_training_steps - warmup_steps)

        if lr_scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(optimizer, T_max=main_steps)
        elif lr_scheduler_type == "linear":
            rel_factor = max(1e-8, min(1.0, main_scheduler_end_factor / warmup_end_factor))
            main_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=rel_factor, total_iters=main_steps)
        else:
            raise ValueError(f"Unsupported scheduler: {lr_scheduler_type}")

        scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_steps])

    # 加载 checkpoint
    start_epoch = 0
    start_global_step = 0
    if checkpoint_path and rank == 0:
        if os.path.isfile(checkpoint_path):
            print(f"📥 Loading checkpoint: {checkpoint_path}")
        else:
            print(f"⚠️ Checkpoint not found. Training from scratch.")
            checkpoint_path = None

    # 同步是否加载
    load_flag = torch.tensor([1 if checkpoint_path else 0], dtype=torch.int32, device=device)
    if rank == 0:
        load_flag[0] = int(os.path.isfile(checkpoint_path)) if checkpoint_path else 0
    dist.broadcast(load_flag, src=0)

    if load_flag.item() == 1:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if rank == 0:
            torch.set_rng_state(ckpt['rng_state'])
            if ckpt.get('cuda_rng_state') is not None:
                torch.cuda.set_rng_state(ckpt['cuda_rng_state'])
            np.random.set_state(ckpt['numpy_rng_state'])
            start_epoch = ckpt.get('epoch', -1) + 1
            start_global_step = ckpt.get('global_step', 0)
            print(f"✅ Resuming from epoch {start_epoch}")

    # 训练循环
    global_step = start_global_step
    total_steps = len(dataloader) * num_epochs

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        sampler.set_epoch(epoch)
        model.train()

        recon_losses = []

        for step, batch in enumerate(dataloader):
            global_step += 1
            x = batch.to(device)  # [B, 1, T]

            recon = model(x)
            loss = F.mse_loss(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            recon_losses.append(loss.item())

            # 日志 & 保存（每 loss_log_interval 步）
            if (step + 1) % loss_log_interval == 0 or step == len(dataloader) - 1:
                avg_recon = np.mean(recon_losses)
                recon_losses.clear()

                # 同步平均损失（多卡）
                avg_tensor = torch.tensor(avg_recon, device=device)
                dist.all_reduce(avg_tensor, op=dist.ReduceOp.AVG)
                avg_recon = avg_tensor.item()

                if rank == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    log_and_save(
                        epoch=epoch,
                        step=global_step,
                        total_epochs=num_epochs,
                        total_steps=total_steps,
                        epoch_start_time=epoch_start_time,
                        epoch_total_steps=len(dataloader),
                        avg_recon_loss=avg_recon,
                        lr=current_lr,
                        loss_csv_path=loss_csv_path,
                    )

                    if use_wandb:
                        wandb.log({
                            "train/recon_loss": avg_recon,
                            "learning_rate": current_lr,
                            "epoch": epoch + 1,
                        }, step=global_step)

        # 每 epoch 保存一次（可选）
        if rank == 0 and (epoch + 1) % save_checkpoint_every_epoch == 0:
            ckpt_path = f"{output_model_path}.epoch{epoch+1}.pth"
            save_full_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                rank=rank
            )

    # 保存最终模型
    if rank == 0:
        save_full_checkpoint(
            path=output_model_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=num_epochs - 1,
            global_step=global_step,
            rank=rank
        )
        print(f"✅ Final model saved to {output_model_path}")
        if use_wandb:
            wandb.finish()

    dist.barrier()
    dist.destroy_process_group()
