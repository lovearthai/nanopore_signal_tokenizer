# nanopore_signal_tokenizer/rvq_tokenizer.py
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*pkg_resources is deprecated.*",
    category=UserWarning,
    module="ont_fast5_api"
)

import os
import json
import gzip
import numpy as np
import torch
from math import ceil
from ont_fast5_api.fast5_interface import get_fast5_file
from .nanopore import nanopore_normalize, nanopore_filter
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.signal import medfilt
from tqdm import tqdm
from .rvq_model import NanoporeRVQModel

class RVQTokenizer:
    """
    Nanopore RVQ Tokenizer 封装类。
    功能：
        - 加载预训练 RVQ 模型
        - tokenize 单个 read / numpy 信号 / 整个 FAST5 目录

    """

    """
    模型结构如下，你可以用这个打印去生成模型参数提取的代码

    📂 Loading nanopore_rvq_tokenizer_chunk12k.pth...

🔍 Type of checkpoint: <class 'collections.OrderedDict'>
============================================================
🔑 Top-level keys:
  - encoder.net.0.weight: shape=(64, 1, 5), dtype=torch.float32
  - encoder.net.0.bias: shape=(64,), dtype=torch.float32
  - encoder.net.2.weight: shape=(64,), dtype=torch.float32
  - encoder.net.2.bias: shape=(64,), dtype=torch.float32
  - encoder.net.2.running_mean: shape=(64,), dtype=torch.float32
  - encoder.net.2.running_var: shape=(64,), dtype=torch.float32
  - encoder.net.2.num_batches_tracked: shape=(), dtype=torch.int64
  - encoder.net.3.weight: shape=(64, 64, 5), dtype=torch.float32
  - encoder.net.3.bias: shape=(64,), dtype=torch.float32
  - encoder.net.5.weight: shape=(64,), dtype=torch.float32
  - encoder.net.5.bias: shape=(64,), dtype=torch.float32
  - encoder.net.5.running_mean: shape=(64,), dtype=torch.float32
  - encoder.net.5.running_var: shape=(64,), dtype=torch.float32
  - encoder.net.5.num_batches_tracked: shape=(), dtype=torch.int64
  - encoder.net.6.weight: shape=(128, 64, 9), dtype=torch.float32
  - encoder.net.6.bias: shape=(128,), dtype=torch.float32
  - encoder.net.8.weight: shape=(128,), dtype=torch.float32
  - encoder.net.8.bias: shape=(128,), dtype=torch.float32
  - encoder.net.8.running_mean: shape=(128,), dtype=torch.float32
  - encoder.net.8.running_var: shape=(128,), dtype=torch.float32
  - encoder.net.8.num_batches_tracked: shape=(), dtype=torch.int64
  - encoder.net.9.weight: shape=(128, 128, 9), dtype=torch.float32
  - encoder.net.9.bias: shape=(128,), dtype=torch.float32
  - encoder.net.11.weight: shape=(128,), dtype=torch.float32
  - encoder.net.11.bias: shape=(128,), dtype=torch.float32
  - encoder.net.11.running_mean: shape=(128,), dtype=torch.float32
  - encoder.net.11.running_var: shape=(128,), dtype=torch.float32
  - encoder.net.11.num_batches_tracked: shape=(), dtype=torch.int64
  - encoder.net.12.weight: shape=(512, 128, 5), dtype=torch.float32
  - encoder.net.12.bias: shape=(512,), dtype=torch.float32
  - encoder.net.14.weight: shape=(512,), dtype=torch.float32
  - encoder.net.14.bias: shape=(512,), dtype=torch.float32
  - encoder.net.14.running_mean: shape=(512,), dtype=torch.float32
  - encoder.net.14.running_var: shape=(512,), dtype=torch.float32
  - encoder.net.14.num_batches_tracked: shape=(), dtype=torch.int64
  - rvq.layers.0._codebook.initted: shape=(), dtype=torch.bool
  - rvq.layers.0._codebook.cluster_size: shape=(1, 8192), dtype=torch.float32
  - rvq.layers.0._codebook.embed_avg: shape=(1, 8192, 512), dtype=torch.float32
  - rvq.layers.0._codebook.embed: shape=(1, 8192, 512), dtype=torch.float32
  - rvq.layers.1._codebook.initted: shape=(), dtype=torch.bool
  - rvq.layers.1._codebook.cluster_size: shape=(1, 8192), dtype=torch.float32
  - rvq.layers.1._codebook.embed_avg: shape=(1, 8192, 512), dtype=torch.float32
  - rvq.layers.1._codebook.embed: shape=(1, 8192, 512), dtype=torch.float32
  - rvq.layers.2._codebook.initted: shape=(), dtype=torch.bool
  - rvq.layers.2._codebook.cluster_size: shape=(1, 8192), dtype=torch.float32
  - rvq.layers.2._codebook.embed_avg: shape=(1, 8192, 512), dtype=torch.float32
  - rvq.layers.2._codebook.embed: shape=(1, 8192, 512), dtype=torch.float32
  - rvq.layers.3._codebook.initted: shape=(), dtype=torch.bool
  - rvq.layers.3._codebook.cluster_size: shape=(1, 8192), dtype=torch.float32
  - rvq.layers.3._codebook.embed_avg: shape=(1, 8192, 512), dtype=torch.float32
  - rvq.layers.3._codebook.embed: shape=(1, 8192, 512), dtype=torch.float32
  - decoder.0.weight: shape=(512, 256, 8), dtype=torch.float32
  - decoder.0.bias: shape=(256,), dtype=torch.float32
  - decoder.2.weight: shape=(256,), dtype=torch.float32
  - decoder.2.bias: shape=(256,), dtype=torch.float32
  - decoder.2.running_mean: shape=(256,), dtype=torch.float32
  - decoder.2.running_var: shape=(256,), dtype=torch.float32
  - decoder.2.num_batches_tracked: shape=(), dtype=torch.int64
  - decoder.3.weight: shape=(256, 128, 12), dtype=torch.float32
  - decoder.3.bias: shape=(128,), dtype=torch.float32
  - decoder.5.weight: shape=(128,), dtype=torch.float32
  - decoder.5.bias: shape=(128,), dtype=torch.float32
  - decoder.5.running_mean: shape=(128,), dtype=torch.float32
  - decoder.5.running_var: shape=(128,), dtype=torch.float32
  - decoder.5.num_batches_tracked: shape=(), dtype=torch.int64
  - decoder.6.weight: shape=(128, 64, 18), dtype=torch.float32
  - decoder.6.bias: shape=(64,), dtype=torch.float32
  - decoder.8.weight: shape=(64,), dtype=torch.float32
  - decoder.8.bias: shape=(64,), dtype=torch.float32
  - decoder.8.running_mean: shape=(64,), dtype=torch.float32
  - decoder.8.running_var: shape=(64,), dtype=torch.float32
  - decoder.8.num_batches_tracked: shape=(), dtype=torch.int64
  - decoder.9.weight: shape=(1, 64, 1), dtype=torch.float32
  - decoder.9.bias: shape=(1,), dtype=torch.float32

    📊 Sample tensor stats (first 3 parameters):
    encoder.net.0.weight: mean=0.0004, std=0.2398, min=-0.6499, max=0.6512
    encoder.net.0.bias: mean=-0.1649, std=0.5765, min=-1.3233, max=1.3681
    encoder.net.2.weight: mean=0.6895, std=0.2447, min=0.2171, max=1.2953

    """

    def __init__(
        self,
        model_ckpt: str = "nanopore_rvq_tokenizer.pth",
        device: str="cuda",
        token_batch_size: int = 1000,
    ):
        try:

                    # --- Device setup: auto-select if not specified ---
            if device is None:
                # User didn't specify → auto choose
                if torch.cuda.is_available():
                    final_device = "cuda"
                    print("✅ CUDA available. Using GPU (device='cuda').")
                else:
                    final_device = "cpu"
                    print("⚠️ CUDA not available. Falling back to CPU.")
            else:
                # User specified a device → validate and use it
                device = device.strip()
                if device.startswith("cuda"):
                    if torch.cuda.is_available():
                        # Optional: validate GPU index if provided
                        if ":" in device:
                            try:
                                idx = int(device.split(":")[1])
                                if idx >= torch.cuda.device_count():
                                    print(f"⚠️ Warning: CUDA device '{device}' not found. Available GPUs: 0-{torch.cuda.device_count() - 1}.")
                                    final_device = "cuda:0"
                                else:
                                    final_device = device
                            except (ValueError, IndexError):
                                print(f"⚠️ Warning: Invalid CUDA device format '{device}'. Using 'cuda:0'.")
                                final_device = "cuda:0"
                        else:
                            final_device = "cuda"  # normalize "cuda" → same as "cuda:0"
                    else:
                        print(f"⚠️ Warning: CUDA not available. Ignoring requested device '{device}', falling back to CPU.")
                        final_device = "cpu"
                elif device == "cpu":
                    final_device = "cpu"
                else:
                    raise ValueError(f"Unsupported device: '{device}'. Use 'cpu', 'cuda', or 'cuda:N'.")

            self.device = final_device
            # --- Load checkpoint ---
            print(f"📂 Loading checkpoint from: {model_ckpt}")
            ckpt_data = torch.load(model_ckpt, map_location='cpu')

            # --- Infer n_q and codebook_size from RVQ embed keys ---
            embed_keys = [k for k in ckpt_data.keys() if '._codebook.embed' in k]
            if not embed_keys:
                raise RuntimeError(
                    "No RVQ codebook embedding found. Expected keys containing '._codebook.embed'."
                )

            # Sort to ensure consistent order (e.g., layers.0 first)
            embed_keys = sorted(embed_keys)
            print(f"🔍 Found {len(embed_keys)} RVQ codebook embed keys.")

            # Get codebook_size from shape: expected (1, codebook_size, dim)
            first_embed = ckpt_data[embed_keys[0]]
            if not hasattr(first_embed, 'shape') or len(first_embed.shape) < 2:
                raise RuntimeError(f"Unexpected tensor shape for {embed_keys[0]}: {first_embed.shape}")
            codebook_size = int(first_embed.shape[1])

            # Infer n_q from layer indices in key names
            layer_indices = set()
            for k in embed_keys:
                parts = k.split('.')
                try:
                    # Format: 'rvq.layers.2._codebook.embed' → parts[2] = '2'
                    if len(parts) >= 3 and parts[1] == 'layers':
                        idx = int(parts[2])
                        layer_indices.add(idx)
                except (ValueError, IndexError):
                    continue

            if not layer_indices:
                raise RuntimeError("Could not parse any valid layer index from codebook keys.")
            n_q = max(layer_indices) + 1

            self.n_q = n_q
            self.codebook_size = codebook_size
            print(f"🎯 Inferred model config: n_q={n_q}, codebook_size={codebook_size}")

            # --- Instantiate model ---
            self.model = NanoporeRVQModel(n_q=n_q, codebook_size=codebook_size)

            # Ensure model defines cnn_stride (your downsample rate)
            if not hasattr(self.model, 'cnn_stride'):
                raise AttributeError(
                    "NanoporeRVQModel must define 'self.cnn_stride' as the total downsampling rate (e.g., 12)."
                )

            if token_batch_size < 1:
                token_batch_size = 1
                print(f"token_batch_size forced to be {token_batch_size} due to miniumu requierment")

            self.downsample_rate = self.model.cnn_stride
            self.chunk_size = token_batch_size*self.downsample_rate
            self.margin_stride_count = self.model.margin_stride_count
            self.margin = self.model.margin_stride_count * self.downsample_rate
            # --- Load state dict ---
            self.model.load_state_dict(ckpt_data)
            self.model.eval()
            self.model.to(self.device)

            # --- Final summary ---
            print("\n✅ RVQTokenizer initialized successfully with the following configuration:")
            print(f"   Model checkpoint : {os.path.abspath(model_ckpt)}")
            print(f"   Device           : {self.device}")
            print(f"   n_q              : {self.n_q}")
            print(f"   Codebook size    : {self.codebook_size}")
            print(f"   Downsample rate  : {self.downsample_rate} (from model.cnn_stride)")
            print(f"   Chunk size       : {self.chunk_size}")
            print(f"   Margin           : {self.margin} points")
            print("-" * 60)

        except Exception as e:
            print(f"❌ Failed to initialize RVQTokenizer: {e}")
            raise

    def _tokenize_chunked_signal(self, signal: np.ndarray) -> np.ndarray:
        """Tokenize long signal using symmetric sliding window with margin handling."""
        if signal.ndim != 1:
            raise ValueError("Signal must be 1D.")
        L = len(signal)
        if L == 0:
            return np.zeros(0, dtype=np.int64)

        T_expected = (L + self.downsample_rate - 1) // self.downsample_rate

        # ────────────────────────
        # Case 1: Short signal (no sliding needed)
        # ────────────────────────
        if L <= self.chunk_size:
            padded = np.pad(signal, (0, self.chunk_size - L), mode='constant')
            x = torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, tokens = self.model(x)
            tokens = tokens.squeeze(0).cpu().numpy()
            # Only keep tokens corresponding to real signal
            return tokens[:T_expected].flatten()

        # ────────────────────────
        # Case 2: Long signal — sliding window
        # ────────────────────────
        margin_samples = self.margin_stride_count * self.downsample_rate
        step_samples = self.chunk_size - 2 * margin_samples
        if step_samples <= 0:
            raise ValueError("chunk_size too small for the given margin.")

        all_tokens = []
        start = 0
        chunk_index = 0

        while start < L:
            # Extract real signal segment
            real_len = min(self.chunk_size, L - start)
            chunk = signal[start:start + real_len]

            # Pad only if necessary (for model input shape)
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), mode='constant')

            # Run model
            x = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, tokens = self.model(x)
            tokens = tokens.squeeze(0).cpu().numpy()  # [T, n_q]

            # Compute how many tokens are valid (from real signal, not padding)
            T_valid = (real_len + self.downsample_rate - 1) // self.downsample_rate

            # ───── Margin handling ─────
            kept_tokens = np.empty((0, self.n_q), dtype=tokens.dtype)

            if chunk_index == 0:
                # First chunk: discard tail margin
                if self.margin_stride_count > 0:
                    end_idx = min(tokens.shape[0] - self.margin_stride_count, T_valid)
                    kept_tokens = tokens[:end_idx]
                else:
                    kept_tokens = tokens[:T_valid]

            elif start + step_samples >= L:
                # Last chunk: discard head margin, then respect T_valid
                if self.margin_stride_count > 0:
                    start_idx = self.margin_stride_count
                    max_len = T_valid - self.margin_stride_count
                    if max_len > 0:
                        kept_tokens = tokens[start_idx : start_idx + max_len]
                    # else: remains empty
                else:
                    kept_tokens = tokens[:T_valid]

            else:
                # Middle chunk: discard both ends
                if self.margin_stride_count > 0 and tokens.shape[0] > 2 * self.margin_stride_count:
                    max_len = T_valid - 2 * self.margin_stride_count
                    if max_len > 0:
                        kept_tokens = tokens[
                            self.margin_stride_count : self.margin_stride_count + max_len
                        ]
                elif self.margin_stride_count == 0:
                    kept_tokens = tokens[:T_valid]

            # Append if any valid tokens remain
            if kept_tokens.shape[0] > 0:
                all_tokens.append(kept_tokens)

            # Move window
            start += step_samples
            chunk_index += 1

        # ────────────────────────
        # Final assembly
        # ────────────────────────
        if not all_tokens:
            return np.zeros(T_expected * self.n_q, dtype=np.int64)

        final_tokens = np.concatenate(all_tokens, axis=0)

        # Trim or pad to expected length (optional in practice)
        if final_tokens.shape[0] > T_expected:
            final_tokens = final_tokens[:T_expected]
        elif final_tokens.shape[0] < T_expected:
            pad = np.zeros((T_expected - final_tokens.shape[0], self.n_q), dtype=np.int64)
            final_tokens = np.concatenate([final_tokens, pad], axis=0)

        return final_tokens.flatten()

    def tokenize_data(self, signal: np.ndarray, token_type: str = "L1", do_normalize: bool = True) -> list:
        try:
            # === 可选：标准化信号 ===
            if do_normalize:
                signal = nanopore_normalize(signal)
            layer_map = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6, "L7": 7, "L8": 8}
            if token_type not in layer_map:
                raise ValueError(f"token_type must be one of {list(layer_map.keys())}, got {token_type}")
            n_layers = layer_map[token_type]
            if n_layers > self.n_q:
                raise ValueError(f"Requested {token_type} (n_layers={n_layers}), but model only has {self.n_q} quantizers.")
            flat_tokens = self._tokenize_chunked_signal(signal)
            if flat_tokens.size == 0:
                return []
            if flat_tokens.size % self.n_q != 0:
                T = flat_tokens.size // self.n_q
                flat_tokens = flat_tokens[:T * self.n_q]
            tokens_2d = flat_tokens.reshape(-1, self.n_q)
            selected = tokens_2d[:, :n_layers]
            parts = []
            for t in range(selected.shape[0]):
                for q in range(n_layers):
                    token_id = int(selected[t, q])
                    parts.append(f"<|bwav:L{q+1}_{token_id}|>")
            return parts

        except Exception as e:
            print(f"❌ tokenize_data failed on signal of length {len(signal)}: {e}")
            return []


    def tokenize_read(self, read, token_type: str = "L1", do_normalize: bool = True) -> list:
        try:
            channel_info = read.handle[read.global_key + 'channel_id'].attrs
            offset = int(channel_info['offset'])
            scaling = channel_info['range'] / channel_info['digitisation']
            raw = read.handle[read.raw_dataset_name][:]
            scaled = np.array(scaling * (raw + offset), dtype=np.float32)
            return self.tokenize_data(scaled, token_type=token_type, do_normalize=do_normalize)
        except Exception as e:
            fast5_path = getattr(read.handle, 'filename', 'unknown.fast5')
            print(f"❌ Error on read {read.read_id} in {fast5_path}: {e}")
            return []


    def tokenize_fast5(self, fast5_path: str, output_path: str, token_type: str = "L1", do_normalize: bool = True):
        print(f"✅ Processing {fast5_path} (normalize={do_normalize})")
        results = []
        with get_fast5_file(fast5_path, mode="r") as f5:
            for read in tqdm(f5.get_reads(), desc=os.path.basename(fast5_path)):
                try:
                    token_list = self.tokenize_read(read, token_type=token_type, do_normalize=do_normalize)
                    token_str = "".join(token_list)
                    results.append({"id": read.read_id, "text": token_str})
                except Exception as e:
                    print(f"❌ Error on read {read.read_id} in {fast5_path}: {e}")
                    continue

        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        print(f"✅ Wrote {len(results)} reads to {output_path}")
