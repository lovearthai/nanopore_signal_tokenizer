from typing import Dict, List, Optional, Tuple
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
        min_weight: float = 0.001,
        weight_lr: float = 0.001,
        initial_weights: Optional[Dict[str, float]] = None,
        warmup_steps: int = 100,
        weight_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
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
            total = sum(initial_weights.values())
            if abs(total - 1.0) > 1e-6:
                print(f"[DWA] Initial weights not normalized (sum={total:.4f}), normalizing...")
                initial_weights = {k: v / total for k, v in initial_weights.items()}
        else:
            n = len(self.weighted_loss_names)
            initial_weights = {name: 1.0 / n for name in self.weighted_loss_names}

        # 处理 weight_bounds
        if weight_bounds is not None:
            if set(weight_bounds.keys()) != set(self.weighted_loss_names):
                raise ValueError("weight_bounds keys must exactly match weighted_loss_names")
            processed_bounds = {}
            for name in self.weighted_loss_names:
                low, high = weight_bounds[name]
                if low < 0 or high > 1 or low > high:
                    raise ValueError(f"Invalid weight_bounds for '{name}': ({low}, {high})")
                if low < min_weight:
                    print(f"[DWA Warning] weight_bounds['{name}'] lower bound {low} < min_weight {min_weight}. Using {min_weight} as effective lower bound.")
                    low = min_weight
                processed_bounds[name] = (float(low), float(high))
            self.weight_bounds = processed_bounds
        else:
            self.weight_bounds = {
                name: (min_weight, 1.0) for name in self.weighted_loss_names
            }

        self.window_size = window_size
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.temperature = temperature
        self.min_weight = min_weight
        self.weight_lr = weight_lr
        self.warmup_steps = warmup_steps
        self.device = device or torch.device("cpu")

        self.step_counter = 0
        self.loss_queues = {
            name: collections.deque(maxlen=window_size) for name in self.loss_names
        }
        self.initial_weights = initial_weights.copy()
        self.raw_weights = initial_weights.copy()
        self.smooth_weights = initial_weights.copy()

    def _project_weights_to_box_constraints(
        self,
        raw_weights: Dict[str, float],
        max_iter: int = 10,
        eps: float = 1e-8
    ) -> Dict[str, float]:
        """
        Project raw_weights into the box constraints defined by self.weight_bounds,
        while ensuring sum(weights) == 1.0.
        Uses iterative proportional redistribution.
        """
        names = list(raw_weights.keys())
        w = {name: raw_weights[name] for name in names}
        lower = {name: self.weight_bounds[name][0] for name in names}
        upper = {name: self.weight_bounds[name][1] for name in names}

        # Initial clipping
        for name in names:
            w[name] = min(upper[name], max(lower[name], w[name]))

        for _ in range(max_iter):
            total = sum(w.values())
            residual = 1.0 - total
            if abs(residual) < eps:
                break

            flexible = [name for name in names if w[name] < upper[name] - eps]
            if not flexible:
                # All at upper bound — normalize
                scale = 1.0 / total
                w = {name: w[name] * scale for name in names}
                break

            total_raw_flex = sum(raw_weights[name] for name in flexible)
            if total_raw_flex < eps:
                add_val = residual / len(flexible)
                for name in flexible:
                    w[name] += add_val
            else:
                for name in flexible:
                    ratio = raw_weights[name] / total_raw_flex
                    w[name] += residual * ratio

            # Re-clip after adjustment
            for name in names:
                w[name] = min(upper[name], max(lower[name], w[name]))

        # Final normalization to handle numerical drift
        final_sum = sum(w.values())
        if abs(final_sum - 1.0) > 1e-5:
            w = {name: w[name] / final_sum for name in names}
        return w

    def update_and_get_weights(self, current_losses: Dict[str, float]) -> Dict[str, float]:
        self.step_counter += 1

        # Push current losses into queues
        for name in self.loss_names:
            if name not in current_losses:
                raise KeyError(f"Missing loss '{name}' in current_losses")
            self.loss_queues[name].append(current_losses[name])

        # Warmup phase: use initial weights
        if self.step_counter <= self.warmup_steps:
            return self.initial_weights.copy()

        # Normal DWA update after warmup
        if all(len(self.loss_queues[name]) > 0 for name in self.weighted_loss_names):
            exp_vals = []
            names_list = []

            for name in self.weighted_loss_names:
                q = list(self.loss_queues[name])
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
                raw_weights[name] = w

            # ✅ Apply box constraints with redistribution
            raw_weights = self._project_weights_to_box_constraints(raw_weights)
            self.raw_weights = raw_weights

            # Smooth update
            for name in self.weighted_loss_names:
                self.smooth_weights[name] = (
                    (1 - self.weight_lr) * self.smooth_weights[name] +
                    self.weight_lr * raw_weights[name]
                )

            # Post-smoothing clipping to bounds (for numerical safety)
            for name in self.weighted_loss_names:
                low, high = self.weight_bounds[name]
                self.smooth_weights[name] = min(high, max(low, self.smooth_weights[name]))

            # Final normalization (tiny drift correction)
            smooth_sum = sum(self.smooth_weights.values())
            if abs(smooth_sum - 1.0) > 1e-5:
                self.smooth_weights = {k: v / smooth_sum for k, v in self.smooth_weights.items()}

        #return self.smooth_weights.copy()
        return self.raw_weights.copy()

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
