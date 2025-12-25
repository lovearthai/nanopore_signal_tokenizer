import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import csv  # ✅ 新增：用于写入 CSV
import time  # 确保已导入
# 相对导入核心组件
from .dataset import NanoporeSignalDataset
from .vq_model import NanoporeVQModel
from typing import Dict, List
import collections
from .dwa import DynamicWeightAverager 

def log_and_save(
    epoch: int,
    step: int,
    total_epochs: int,
    total_steps: int,
    epoch_start_time: float,          # ← 替换 elapsed_time / remaining_time
    epoch_total_steps: int,           # ← 当前 epoch 的总步数（用于估算剩余时间）
    avg_recon_loss: float,
    avg_total_loss: float,
    avg_comit_loss: float,
    avg_diver_loss: float,
    avg_ortho_loss: float,
    codebook_usage: float,
    loss_csv_path: str,
    dynamic_recon_weight: float,
    dynamic_comit_weight: float,
    dynamic_ortho_weight: float,
    dynamic_diver_weight: float,
    lr: float,
):
    """
    打印当前训练状态并保存到CSV文件。
    时间字符串在函数内部生成，格式为 H:MM:SS（若 >=1h）或 MM:SS。
    """
    import time

    # === 🕒 动态计算时间 ===
    current_time = time.time()
    elapsed_seconds = current_time - epoch_start_time
    steps_done = step % epoch_total_steps or epoch_total_steps  # 防止 step=0
    if steps_done == 0:
        steps_done = 1
    avg_time_per_step = elapsed_seconds / steps_done
    remaining_steps = epoch_total_steps - steps_done
    remaining_seconds = avg_time_per_step * max(0, remaining_steps)

    def format_hms(seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        else:
            return f"{m:02d}:{s:02d}"

    elapsed_str = format_hms(elapsed_seconds)
    remaining_str = format_hms(remaining_seconds)

    # === 🔢 动态对齐 ===
    epoch_width = len(str(total_epochs))
    step_width = len(str(total_steps))

    # === 🖨️ 打印日志 ===
    print(
        f"[Epoch {epoch+1:>{epoch_width}}/{total_epochs} | "
        f"Step {step:>{step_width}}/{total_steps} | "
        f"{elapsed_str}<{remaining_str}] "
        f"Total: {avg_total_loss:>8.6f} | "
        f"Recon: {avg_recon_loss:>8.6f} | "
        f"Comit: {avg_comit_loss:>8.6f} | "
        f"Ortho: {avg_ortho_loss:>8.6f} | "
        f"Diver: {avg_diver_loss:>3.2f} | "
        f"Usage: {codebook_usage*100:>3.1f}% | "
        f"LR: {lr:>7.2e} |"
    )

    # === 💾 写入 CSV ===
    row_data = [
        epoch + 1,
        step,
        avg_recon_loss,
        avg_total_loss,
        avg_comit_loss,
        avg_diver_loss,
        avg_ortho_loss,
        codebook_usage * 100,  # 保存为百分比更直观（可选）
        dynamic_recon_weight,
        dynamic_comit_weight,
        dynamic_ortho_weight,
        dynamic_diver_weight,
        lr
    ]
    with open(loss_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)

def vq_train(
    npy_dir: str,
    output_model_path: str,
    batch_size: int = 16,
    lr: float = 1e-4,
    num_epochs: int = 10,
    codebook_size: int = 8192,
    codebook_dim: int = 512,
    chunk_size: int = 12000,
    num_workers: int = 8,
    update_loss_weight_every: int = 10,
    prefetch_factor: int = 128,
    val_ratio: int = 0.1,
    do_evaluate: bool = True,
    commitment_weight: float = 1.0,
    codebook_diversity_loss_weight: float = 1.0,
    orthogonal_reg_weight: float = 1.0,
    loss_log_interval: int = 10,
    loss_csv_path: str = "train_loss.csv",  # ✅ 新增参数：loss 日志 CSV 路径
    use_wandb: bool = True,                 # 是否启用 wandb
    wandb_project: str = "nanopore_vq",     # wandb 项目名
    wandb_name: str = "default_wandb_runname",  # 运行名称（可选
    # ====== 📈 学习率调度器参数（新增）======
    lr_scheduler_type: str = "cosine",          # 'cosine', 'linear', 'constant'
    warmup_steps: int = 500,                    # 预热步数（全局 step）
    warmup_start_factor: float = 1e-6,          # warmup 起始 lr = lr * start_factor
    warmup_end_factor: float = 1.0,             # warmup 结束 lr = lr * end_factor
    main_scheduler_end_factor: float = 1e-6,    # 主调度器最终 lr = lr * end_factor（仅 linear 用）
    save_checkpoint_every_spoch: int = 1000,    # 每多少个update_loss_weight_every进行一次检查点保存
    evaluate_every_spoch: int = 1000,           # 每多少个update_loss_weight_every进行一次evaluate
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

    # ========== 初始化 wandb（仅 rank 0）==========
    if rank == 0 and use_wandb:
       import wandb
       wandb.init(
           project=wandb_project,
           name=wandb_name,
           config={
               "batch_size": batch_size,
               "lr": lr,
               "num_epochs": num_epochs,
               "codebook_size": codebook_size,
               "codebook_dim": codebook_dim,
               "chunk_size": chunk_size,
               "update_loss_weight_every": update_loss_weight_every,
               "commitment_weight": commitment_weight,
               "codebook_diversity_loss_weight": codebook_diversity_loss_weight,
               "orthogonal_reg_weight": orthogonal_reg_weight,
               "world_size": world_size,
           }
        )
    else:
        wandb = None  # 避免未定义


    if rank == 0:
        print(f"🚀 Using {world_size} GPUs for training.")
        print(f"📂 Data directory: {npy_dir}")
        print(f"💾 Model will be saved to: {output_model_path}")
        print(f"⚙️  Hyperparameters: "
              f"batch_size={batch_size}, lr={lr}, epochs={num_epochs}, "
              f"codebook_size={codebook_size}, codebook_dim={codebook_dim}, chunk_size={chunk_size}, "
              f"do_evaluate={do_evaluate}, save_checkpoint_every_spoch={save_checkpoint_every_spoch}")

        # ✅ 初始化 CSV 文件（仅 rank 0）
        with open(loss_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = [
                'epoch', 'step',
                'recon_loss', 'total_loss', 'comit_loss', 'diver_loss', 'ortho_loss', 'codebook_usage',
                'wv_recon', 'wv_comit', 'wv_ortho', 'wv_diver',  # ← 新增
                'lr'
            ]
            writer.writerow(header)

    # ========== 数据加载 ==========
    dataset = NanoporeSignalDataset(shards_dir=npy_dir)
    # ====== 新增：只取前 N 个样本（或任意子集）======
    #subset_size = int(1.0 * len(dataset))  # 例如：只用 10% 的数据
    # 或者指定绝对数量：
    # subset_size = 100_000
    # 确保不超限
    #subset_size = min(subset_size, len(dataset))
    # 固定子集选择的随机性（仅影响 subset 选取，不影响训练中的 shuffle）
    #torch.manual_seed(42)
    #indices = torch.randperm(len(dataset)).tolist()[:subset_size]
    #dataset = torch.utils.data.Subset(dataset, indices)
    # 注意：这个 seed 只控制 subset 选取，不影响 DataLoader 内部的 shuffle=True 或 DistributedSampler 的打乱行为。


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

    # ========== 可选：验证集（仅用于评估）==========
    val_loader = None
    if do_evaluate and rank == 0:  # ⭐ 只在 rank 0 创建 val_loader（其他 rank 不需要）
        actual_val_size = int(val_ratio *len(dataset))
        if actual_val_size < 1:
            actual_val_size = 1
        indices = np.random.choice(len(dataset), size=actual_val_size, replace=False)
        val_subset = torch.utils.data.Subset(dataset, indices)  # ← 复用 dataset
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(2, num_workers // 2),
            pin_memory=True
        )
    # ========== 模型与优化器 ==========
    model = NanoporeVQModel(
            codebook_size=codebook_size, 
            codebook_dim=codebook_dim, 
            commitment_weight=commitment_weight,
            codebook_diversity_loss_weight=codebook_diversity_loss_weight,
            orthogonal_reg_weight=orthogonal_reg_weight
            ).to(device)
    model = DDP(model, device_ids=[local_device_id])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)




    # 只对前三个做动态加权
    if rank == 0:
        # 自定义初始权重（例如更重视 recon_loss）
        init_w = {
            "recon_loss": 0.25,
            "comit_loss": 0.25,
            "ortho_loss": 0.25,
            "diver_loss": 0.25
        }
        # 定义权重边界
        bounds = {
            "recon_loss": (0.01, 0.99),
            "comit_loss": (0.01, 0.99),
            "ortho_loss": (0.01, 0.99),
            "diver_loss": (0.01, 0.99),
        }

        dwa = DynamicWeightAverager(
            loss_names=["recon_loss", "comit_loss", "ortho_loss", "diver_loss","total_loss"],
            weighted_loss_names=["recon_loss", "comit_loss", "ortho_loss","diver_loss"],
            initial_weights=init_w,
            weight_bounds=bounds,
            warmup_steps=10,          # 前 200 步固定用 init_w
            temperature=1.0,
            window_size=50,
            slow_window=45,
            fast_window=5,
            device=device
        )

    # ========== 学习率调度器 ==========
    if rank == 0:
        total_training_steps = len(dataloader) * num_epochs
        print(f"🔢 Total training steps: {total_training_steps}, Warmup steps: {warmup_steps}")


    # ========== 学习率调度器（完全参数化）==========
    scheduler = None
    total_training_steps = len(dataloader) * num_epochs

    if rank == 0:
        print(f"🔢 Total training steps: {total_training_steps}")
        if lr_scheduler_type != "constant":
            print(f"📈 Using LR scheduler: {lr_scheduler_type}, "
                  f"warmup_steps={warmup_steps}, "
                  f"warmup: {warmup_start_factor}→{warmup_end_factor}, "
                  f"main_end_factor={main_scheduler_end_factor}")

    if lr_scheduler_type != "constant":
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

        # Warmup 阶段：从 warmup_start_factor * lr 到 warmup_end_factor * lr
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=warmup_end_factor,
            total_iters=warmup_steps
        )

        main_steps = max(1, total_training_steps - warmup_steps)

        if lr_scheduler_type == "cosine":
            # Cosine 退火：从当前 lr（即 warmup_end_factor * lr）退火到 0
            main_scheduler = CosineAnnealingLR(optimizer, T_max=main_steps)
        elif lr_scheduler_type == "linear":
            # Linear 衰减：从当前 lr 衰减到 main_scheduler_end_factor * 原始 lr
            # 注意：LinearLR 的 end_factor 是相对于 warmup 结束时的 lr
            # 所以目标 lr = (main_scheduler_end_factor * lr) / (warmup_end_factor * lr) = main_scheduler_end_factor / warmup_end_factor
            relative_end_factor = main_scheduler_end_factor / warmup_end_factor if warmup_end_factor > 0 else 0.0
            relative_end_factor = max(1e-8, min(1.0, relative_end_factor))  # 安全 clamp
            main_scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=relative_end_factor,
                total_iters=main_steps
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler_type: {lr_scheduler_type}")

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
    # else: scheduler remains None → constant LR

    # ========== 评估函数（仅在 do_evaluate=True 时调用）==========

    def evaluate_codebook_usage():
        if val_loader is None:  # ⭐ 安全检查
            return 0.0, 0
        model.eval()
        used_codes = set()
        total_tokens = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device)
                _, indices, _, _ = model.module(x)
                indices = indices.cpu().numpy().flatten()
                used_codes.update(indices.tolist())
                total_tokens += indices.size
        usage_ratio = len(used_codes) / codebook_size
        model.train()
        return usage_ratio, total_tokens
    # ========== 训练循环 ==========
    model.train()
    codebook_usage = 0.0
    total_steps = len(dataloader)*num_epochs
    epoch_total_steps = len(dataloader)  # 当前 epoch 的本地 step 数（每个 rank 相同）
    # 👇 新增：缓存权重（初始值可设为 1.0）
    cached_wvalue = torch.tensor([0.25, 0.25, 0.25,0.25], device=device)  # [recon, comit, ortho]
    # 在 for epoch in range(num_epochs): 之前
    loss_buffer = {
        "recon": [],
        "comit": [],
        "ortho": [],
        "diver": []
    }
    # 每10个step就是一个spoch
    spoch = 0
    total_spochs = int(total_steps/update_loss_weight_every)
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # ← 新增：记录 epoch 开始时间
        sampler.set_epoch(epoch)
        num_batches = torch.tensor(len(dataloader), device=device)
        for step, batch in enumerate(dataloader):
            x = batch.to(device)
            # break_loss 是否已包含 commitment_weight？
            # 在 vector_quantize_pytorch 中，返回的 break_loss 已经是乘过 commitment_weight 的（默认 0.25）
            # 因为 VectorQuantize 返回的 break_loss 是：
            # break_loss = (z_e - e_k.detach()).pow(2).mean() * self.commitment_weight
            # 它是一个 requires_grad=False 的 scalar tensor，位于与输入相同的设备上（GPU）。
            # 所以 break_loss 本身就是 GPU tensor，不需要 .item()。
            recon, indices,break_loss, loss_breakdown = model(x)
            # 如果你想弱化重建、强调离散表示质量，可以加一个超参数：
            # recon_weight = 0.01  # << 降低重建权重
            # loss = recon_weight * F.mse_loss(recon, x) + break_loss
            # 这样模型会更关注“编码器贴紧码本”和“码本分散”，而不是像素级还原信号——非常适合做 tokenizer。
            recon_loss = F.mse_loss(recon, x)
            comit_loss = loss_breakdown.commitment
            diver_loss = loss_breakdown.codebook_diversity
            ortho_loss = loss_breakdown.orthogonal_reg
            #print("comit_loss grad:", comit_loss.requires_grad) # True
            total_loss = (recon_loss + 
                comit_loss * (commitment_weight+epoch) + 
                ortho_loss * orthogonal_reg_weight + 
                diver_loss * codebook_diversity_loss_weight)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # 👇 更新学习率（每个 step）
            if scheduler is not None:
                scheduler.step()
            # 👇 只缓存标量值（无梯度）
            loss_buffer["recon"].append(recon_loss.item())
            loss_buffer["comit"].append(comit_loss.item())
            loss_buffer["ortho"].append(ortho_loss.item())
            loss_buffer["diver"].append(diver_loss.item())
            # ====== 🔁 动态权重更新逻辑（每隔 update_every 步） ======
            wv_recon, wv_comit, wv_ortho,wv_diver = cached_wvalue.tolist()
            should_update_weights = (step + 1) % update_loss_weight_every == 0 or  (step == len(dataloader) - 1)
            if should_update_weights:
                spoch += 1
                # 计算当前窗口平均（防止空）
                def safe_mean(lst):
                    return sum(lst) / len(lst) if lst else 0.0
                local_avg_losses = torch.tensor([
                    safe_mean(loss_buffer["recon"]),
                    safe_mean(loss_buffer["comit"]),
                    safe_mean(loss_buffer["ortho"]),
                    safe_mean(loss_buffer["diver"])
                ], device=device)
                # 👇 全局同步：求所有 rank 的平均
                # ← 所有 rank 在这里同步，loss 已平均 本身就起到了 隐式的 barrier 作用，无需再手动加 dist.barri
                dist.all_reduce(local_avg_losses, op=dist.ReduceOp.AVG)
                global_avg_recon, global_avg_comit, global_avg_ortho, global_avg_diver = local_avg_losses.tolist()
                global_avg_total = (
                            global_avg_recon +
                            global_avg_comit * commitment_weight +
                            global_avg_ortho * orthogonal_reg_weight +
                            global_avg_diver * codebook_diversity_loss_weight )

                if rank == 0:
                    current_losses = {
                        "recon_loss": global_avg_recon,
                        "comit_loss": global_avg_comit,
                        "ortho_loss": global_avg_ortho,
                        "diver_loss": global_avg_diver,
                        "total_loss": global_avg_total
                    }
                    wvalue = dwa.update_and_get_weights(current_losses)
                    wvalue_tensor = torch.tensor([
                        wvalue["recon_loss"],
                        wvalue["comit_loss"],
                        wvalue["ortho_loss"],
                        wvalue["diver_loss"],
                    ], device=device)
                else:
                    wvalue_tensor = torch.empty(4, device=device)
                # 广播新权重
                dist.broadcast(wvalue_tensor, src=0) # ← 所有 rank 在这里同步，收到广播的权重  本身就起到了 隐式的 barrier 作用，无需再手动加 dist.barrier()。
                cached_wvalue = wvalue_tensor  # 更新缓存
                # 🔁 清空 buffer，为下一个窗口准备
                loss_buffer = {k: [] for k in loss_buffer}
                    

                if rank == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    # 获取最新 fast loss（可用于日志、调试、监控）
                    global_step = epoch * len(dataloader) + (step + 1)
                    log_and_save(
                        epoch=epoch,
                        step=global_step,
                        total_epochs=num_epochs,
                        total_steps=total_steps,
                        epoch_start_time=epoch_start_time,      # ✅ 传入时间戳
                        epoch_total_steps=len(dataloader),      # ✅ 用于估算剩余时间
                        avg_recon_loss=global_avg_recon,
                        avg_total_loss=global_avg_total,
                        avg_comit_loss=global_avg_comit,
                        avg_diver_loss=global_avg_diver,
                        avg_ortho_loss=global_avg_ortho,
                        codebook_usage=codebook_usage,
                        loss_csv_path=loss_csv_path,
                        dynamic_recon_weight=wv_recon,
                        dynamic_comit_weight=wv_comit,
                        dynamic_ortho_weight=wv_ortho,
                        dynamic_diver_weight=wv_diver,
                        lr=current_lr
                    )
                    # === 📊 wandb 日志 ===
                    log_dict = {
                        "train/recon_loss": global_avg_recon,
                        "train/comit_loss": global_avg_comit,
                        "train/ortho_loss": global_avg_ortho,
                        "train/diver_loss": global_avg_diver,
                        "train/total_loss": global_avg_total,
                        "train/codebook_usage": codebook_usage,
                        "weights/recon": wv_recon,
                        "weights/comit": wv_comit,
                        "weights/ortho": wv_ortho,
                        "weights/diver": wv_diver,
                        "epoch": epoch + 1,
                        "learning_rate": current_lr,  # 如果使用 scheduler，可动态获取
                    }
                    if use_wandb:
                        wandb.log(log_dict, step=global_step)

                if rank == 0 and (spoch + 1)% evaluate_every_spoch == 0 and spoch < total_spochs:
                    codebook_usage, total_tokens = evaluate_codebook_usage()
                    print(
                        f"Spoch {spoch+1} - "
                        f"Codebook Usage: {codebook_usage:.2%} "
                        )
                if rank == 0 and (spoch + 1)% save_checkpoint_every_spoch == 0:
                    # ✅ 检查点保存逻辑（仅 rank 0）
                    checkpoint_path = f"{output_model_path}.spoch{spoch+1}.pth"
                    torch.save(model.module.state_dict(), checkpoint_path)
                    print(f"✅ Checkpoint saved to {checkpoint_path}")

    # 保存最终模型（仅 rank 0）
    if rank == 0:
        torch.save(model.module.state_dict(), output_model_path)
        print(f"✅ Final model saved to {output_model_path}")
        if use_wandb:
            wandb.finish()  # ✅ 正确关闭
    dist.barrier()
    dist.destroy_process_group()
