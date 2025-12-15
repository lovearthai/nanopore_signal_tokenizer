# nanopore_signal_tokenizer/vq_tokenizer.py
# Suppress known deprecation warnings

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
from tqdm import tqdm
from scipy.signal import medfilt


# Import your model definition (must define NanoporeVQModel)
from .vq_model import NanoporeVQModel


class VQTokenizer:
    """
    Nanopore Single-Layer VQ Tokenizer.
    - Uses VectorQuantize (not RVQ)
    - No reconstruction loss needed
    - Designed for diversity + waveform backbone retention
    """

    def __init__(
        self,
        model_ckpt: str = "nanopore_vq_tokenizer.pth",
        device: str = "cuda",
        token_batch_size: int = 1000,
    ):
        # --- Device setup ---
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device.strip()
            if device.startswith("cuda"):
                if not torch.cuda.is_available():
                    print("⚠️ CUDA not available, falling back to CPU.")
                    self.device = "cpu"
                else:
                    self.device = device
            elif device == "cpu":
                self.device = "cpu"
            else:
                raise ValueError(f"Unsupported device: {device}")
        
        print(f"✅ Using device: {self.device}")

        # --- Load checkpoint ---
        print(f"📂 Loading checkpoint: {model_ckpt}")
        ckpt_data = torch.load(model_ckpt, map_location="cpu")

        # --- Infer codebook_size and dim from embed key ---
        embed_keys = [k for k in ckpt_data.keys() if "_codebook.embed" in k]
        if not embed_keys:
            raise RuntimeError("No codebook embedding found in checkpoint.")

        # Assume single quantizer: key like 'quantizer._codebook.embed'
        embed_key = embed_keys[0]
        embed_tensor = ckpt_data[embed_key]  # shape: [codebook_size, dim] or [1, codebook_size, dim]

        if len(embed_tensor.shape) == 3:
            codebook_size = int(embed_tensor.shape[1])
            dim = int(embed_tensor.shape[2])
        elif len(embed_tensor.shape) == 2:
            codebook_size = int(embed_tensor.shape[0])
            dim = int(embed_tensor.shape[1])
        else:
            raise RuntimeError(f"Unexpected codebook shape: {embed_tensor.shape}")

        self.codebook_size = codebook_size
        self.dim = dim
        print(f"🎯 Inferred: codebook_size={codebook_size}, dim={dim}")

        # --- Instantiate model ---
        self.model = NanoporeVQModel(codebook_size=codebook_size)

        if not hasattr(self.model, 'cnn_stride'):
            raise AttributeError("Model must define 'cnn_stride' (total downsampling rate).")
        if not hasattr(self.model, 'margin_stride_count'):
            self.model.margin_stride_count = 2  # default fallback

        self.downsample_rate = self.model.cnn_stride
        self.margin_stride_count = self.model.margin_stride_count
        self.margin = self.margin_stride_count * self.downsample_rate
        self.model_RF = self.model.RF
        if token_batch_size < 1:
            token_batch_size = 1
        self.chunk_size = token_batch_size * self.downsample_rate

        # --- Load state dict ---
        self.model.load_state_dict(ckpt_data)
        self.model.eval()
        self.model.to(self.device)

        print("\n✅ VQTokenizer initialized:")
        print(f"   Checkpoint       : {os.path.abspath(model_ckpt)}")
        print(f"   Device           : {self.device}")
        print(f"   Codebook size    : {self.codebook_size}")
        print(f"   Latent dim       : {self.dim}")
        print(f"   Downsample rate  : {self.downsample_rate}")
        print(f"   Chunk size       : {self.chunk_size}")
        print(f"   Margin           : {self.margin} samples")
        print("-" * 60)
    

    def _tokenize_chunked_signal(self, signal: np.ndarray) -> np.ndarray:
        """Tokenize 1D signal using sliding window with margin."""
        if signal.ndim != 1:
            raise ValueError("Signal must be 1D.")
        L = len(signal)
        if L < self.model_RF:
            return np.array([], dtype=np.int64)

        if L == 0:
            return np.array([], dtype=np.int64)

        T_expected = (L + self.downsample_rate - 1) // self.downsample_rate

        if L <= self.chunk_size:
            padded = np.pad(signal, (0, self.chunk_size - L), mode='constant')
            x = torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                recon,tokens,commit_loss = self.model(x)  # returns [B, T] or [B, T, 1] → squeeze to [T]
            tokens = tokens.squeeze(0).cpu().numpy()
            if tokens.ndim == 2:
                tokens = tokens[:, 0]  # take first (and only) layer
            return tokens[:T_expected].astype(np.int64)

        # Long signal: sliding window
        margin_samples = self.margin
        step_samples = self.chunk_size - 2 * margin_samples
        if step_samples <= 0:
            raise ValueError("chunk_size too small for margin.")

        all_tokens = []
        start = 0
        chunk_index = 0

        while start < L:
            real_len = min(self.chunk_size, L - start)
            chunk = signal[start:start + real_len]
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), mode='constant')

            x = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                recon,tokens,commit_loss = self.model(x)
            tokens = tokens.squeeze(0).cpu().numpy()
            if tokens.ndim == 2:
                tokens = tokens[:, 0]

            T_valid = (real_len + self.downsample_rate - 1) // self.downsample_rate

            kept_tokens = np.array([], dtype=np.int64)

            if chunk_index == 0:
                end_idx = T_valid - self.margin_stride_count if self.margin_stride_count > 0 else T_valid
                kept_tokens = tokens[:max(0, end_idx)]
            elif start + step_samples >= L:
                start_idx = self.margin_stride_count
                max_len = T_valid - self.margin_stride_count
                if max_len > 0:
                    kept_tokens = tokens[start_idx : start_idx + max_len]
            else:
                if self.margin_stride_count > 0 and len(tokens) > 2 * self.margin_stride_count:
                    max_len = T_valid - 2 * self.margin_stride_count
                    if max_len > 0:
                        kept_tokens = tokens[
                            self.margin_stride_count : self.margin_stride_count + max_len
                        ]
                else:
                    kept_tokens = tokens[:T_valid]

            if kept_tokens.size > 0:
                all_tokens.append(kept_tokens)

            start += step_samples
            chunk_index += 1

        if not all_tokens:
            return np.zeros(T_expected, dtype=np.int64)

        final_tokens = np.concatenate(all_tokens, axis=0)
        if len(final_tokens) > T_expected:
            final_tokens = final_tokens[:T_expected]
        elif len(final_tokens) < T_expected:
            final_tokens = np.pad(final_tokens, (0, T_expected - len(final_tokens)), constant_values=0)

        return final_tokens.astype(np.int64)

    def tokenize_data(self, signal: np.ndarray,do_normalize: bool = True, medf: int = 0, lpf: int = 0) -> list:
        
        if do_normalize:
            signal = nanopore_normalize(signal)
        if medf > 0:
            signal = medfilt(signal,kernel_size=medf)
        if lpf >  0:
            signal = nanopore_filter(signal,cutoff=lpf)
        flat_tokens = self._tokenize_chunked_signal(signal)
        if flat_tokens.size == 0:
            return []

        parts = []
        for token_id in flat_tokens:
            parts.append(f"<|bwav:{int(token_id)}|>")
        return parts

    def tokenize_read(self, read, do_normalize: bool = True) -> list:
        try:
            channel_info = read.handle[read.global_key + 'channel_id'].attrs
            offset = int(channel_info['offset'])
            scaling = channel_info['range'] / channel_info['digitisation']
            raw = read.handle[read.raw_dataset_name][:]
            scaled = np.array(scaling * (raw + offset), dtype=np.float32)
            return self.tokenize_data(scaled, do_normalize)
        except Exception as e:
            fast5_path = getattr(read.handle, 'filename', 'unknown.fast5')
            print(f"❌ Error on read {read.read_id} in {fast5_path}: {e}")
            return []

    def tokenize_fast5(self, fast5_path: str, output_path: str,do_normalize: bool = True):
        print(f"✅ Processing {fast5_path}")
        results = []
        with get_fast5_file(fast5_path, mode="r") as f5:
            for read in tqdm(f5.get_reads(), desc=os.path.basename(fast5_path)):
                try:
                    token_list = self.tokenize_read(read, do_normalize)
                    token_str = "".join(token_list)
                    results.append({"id": read.read_id, "text": token_str})
                except Exception as e:
                    print(f"❌ Failed on read {read.read_id}: {e}")
                    continue

        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        print(f"✅ Wrote {len(results)} reads to {output_path}")
