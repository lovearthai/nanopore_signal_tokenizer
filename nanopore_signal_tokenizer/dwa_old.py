from typing import Dict, List, Optional
import collections
import torch
import math

class DynamicWeightAverager:
    def __init__(
        self,
        loss_names: List[str] = ["recon_loss", "comit_loss", "ortho_loss"],
        weighted_loss_names: List[str] = None,
        window_size: int = 30,
        fast_window: int = 5,
        slow_window: int = 20,
        temperature: float = 1.0,
        min_weight: float = 0.1,
        weight_lr: float = 0.2,
        initial_weights: Optional[Dict[str, float]] = None,  # 👈 新增：自定义初始权重
        warmup_steps: int = 100,                             # 👈 新增：前 N 步冻结权重
        device: torch.device = None
    ):
        self.loss_names = loss_names
        self.weighted_loss_names = weighted_loss_names or loss_names
        if not set(self.weighted_loss_names).issubset(set(self.loss_names)):
            raise ValueError("weighted_loss_names must be subset of loss_names")

        if window_size < max(fast_window, slow_window):
            raise ValueError(f"window_size ({window_size}) must be >= max(fast_window={fast_window}, slow_window={slow_window})")

        # 验证 initial_weights
        if initial_weights is not None:
            if set(initial_weights.keys()) != set(self.weighted_loss_names):
                raise ValueError(
                    f"initial_weights keys {set(initial_weights.keys())} must exactly match weighted_loss_names {set(self.weighted_loss_names)}"
                )
            # 归一化初始权重（可选但推荐）
            total = sum(initial_weights.values())
            if abs(total - 1.0) > 1e-6:
                print(f"[DWA] Initial weights not normalized (sum={total:.4f}), normalizing...")
                initial_weights = {k: v / total for k, v in initial_weights.items()}
        else:
            # 默认均匀初始化
            n = len(self.weighted_loss_names)
            initial_weights = {name: 1.0 / n for name in self.weighted_loss_names}

        self.window_size = window_size
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.temperature = temperature
        self.min_weight = min_weight
        self.weight_lr = weight_lr
        self.warmup_steps = warmup_steps
        self.device = device or torch.device("cpu")

        self.step_counter = 0

        # 单一滑动窗口
        self.loss_queues = {
            name: collections.deque(maxlen=window_size) for name in self.loss_names
        }

        # 保存初始权重（用于 warmup 期间和重置）
        self.initial_weights = initial_weights.copy()
        self.raw_weights = initial_weights.copy()
        self.smooth_weights = initial_weights.copy()

    def _compute_uniform_weights(self) -> Dict[str, float]:
        n = len(self.weighted_loss_names)
        return {name: 1.0 / n for name in self.weighted_loss_names}

    def update_and_get_weights(self, current_losses: Dict[str, float]) -> Dict[str, float]:
        self.step_counter += 1

        # 1️⃣ 推入当前 loss（即使 warmup 期间也记录，为之后做准备）
        for name in self.loss_names:
            if name not in current_losses:
                raise KeyError(f"Missing loss '{name}' in current_losses")
            self.loss_queues[name].append(current_losses[name])

        # 2️⃣ Warmup 期间：直接返回初始权重，不更新
        if self.step_counter <= self.warmup_steps:
            return self.initial_weights.copy()

        # 3️⃣ 超出 warmup：正常更新 DWA 权重
        if all(len(self.loss_queues[name]) > 0 for name in self.weighted_loss_names):
            exp_vals = []
            names_list = []

            for name in self.weighted_loss_names:
                q = list(self.loss_queues[name])
                # 安全取最近 N 个（即使队列长度 < window）
                fast_avg = sum(q[-self.fast_window:]) / min(len(q), self.fast_window)
                slow_avg = sum(q[-self.slow_window:]) / min(len(q), self.slow_window)

                ratio = fast_avg / (slow_avg + 1e-8)
                exp_val = math.exp(-ratio / self.temperature)
                exp_vals.append(exp_val)
                names_list.append(name)

            total_exp = sum(exp_vals)
            raw_weights = {}
            for i, name in enumerate(names_list):
                w = exp_vals[i] / (total_exp + 1e-8)
                w = max(w, self.min_weight)
                raw_weights[name] = w

            # 归一化 raw weights
            weight_sum = sum(raw_weights.values())
            if weight_sum > 1.0:
                raw_weights = {k: v / weight_sum for k, v in raw_weights.items()}

            self.raw_weights = raw_weights

            # 平滑更新
            for name in self.weighted_loss_names:
                self.smooth_weights[name] = (
                    (1 - self.weight_lr) * self.smooth_weights[name] +
                    self.weight_lr * raw_weights[name]
                )

            # 确保平滑后仍满足 min_weight 和归一化
            for name in self.weighted_loss_names:
                self.smooth_weights[name] = max(self.smooth_weights[name], self.min_weight)

            smooth_sum = sum(self.smooth_weights.values())
            if smooth_sum > 1.0:
                self.smooth_weights = {k: v / smooth_sum for k, v in self.smooth_weights.items()}

        return self.smooth_weights.copy()

    def get_current_loss_averages(self, last_n: int = None) -> Dict[str, float]:
        result = {}
        for name, q in self.loss_queues.items():
            if not q:
                result[name] = 0.0
            else:
                vals = list(q)
                if last_n is not None:
                    vals = vals[-last_n:]
                result[name] = sum(vals) / len(vals)
        return result

    def get_current_weights(self) -> Dict[str, float]:
        return self.smooth_weights.copy()

    def get_raw_weights(self) -> Dict[str, float]:
        return self.raw_weights.copy()

    def get_step(self) -> int:
        return self.step_counter
