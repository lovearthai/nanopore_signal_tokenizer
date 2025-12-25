# nanopore_signal_tokenizer/fast5.py

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import os
import numpy as np
import glob
from ont_fast5_api.fast5_interface import get_fast5_file
from .nanopore import nanopore_normalize, nanopore_filter
from scipy.signal import medfilt
from pathos.multiprocessing import ProcessPool
from multiprocessing import cpu_count


class Fast5Dir:
    """
    将 Nanopore 原始 .fast5 文件批量转换为预处理后的 chunked .npy 文件。

    📌 信号处理流水线（所有步骤在 to_chunks_parallel 中控制）：
        1. 【缩放】raw → pA；
        2. 【归一化】median-MAD（可选）；
        3. 【中值滤波】kernel=5（可选）；
        4. 【低通滤波】Butterworth（可选）；
        5. 【分块】滑动窗口。

    📦 输出：每个 .fast5 → 一个 .npy，内容为 list[dict]，含 read_id、位置、chunk_data。
    """

    def __init__(self, fast5_dir: str, default_fs: int = 5000):
        """
        初始化目录处理器。

        Args:
            fast5_dir (str): 包含 .fast5 文件的目录。
            default_fs (int): 默认采样率（Hz），用于缺失 metadata 的情况。
        """
        if not os.path.isdir(fast5_dir):
            raise ValueError(f"FAST5 directory does not exist: {fast5_dir}")

        self.fast5_dir = fast5_dir
        self.fast5_files = sorted(glob.glob(os.path.join(fast5_dir, "*.fast5")))
        self.default_fs = default_fs

        if not self.fast5_files:
            raise FileNotFoundError(f"No .fast5 files found in {fast5_dir}")

    @staticmethod
    def get_sampling_rate_from_read(read):
        try:
            channel_info = read.handle[read.global_key + 'channel_id'].attrs
            return int(channel_info['sampling_rate'])
        except Exception:
            return None

    def _sliding_window_chunks_with_pos(self, signal: np.ndarray, window_size: int, stride: int):
        n_points = len(signal)
        if n_points < window_size:
            return []

        chunks = []
        start = 0
        while start + window_size <= n_points:
            end = start + window_size
            chunks.append({
                'chunk_start': start,
                'chunk_end': end,
                'chunk_data': signal[start:end].copy()
            })
            start += stride
        return chunks

    def _process_single_fast5(
        self,
        fast5_path: str,
        output_dir: str,
        window_size: int,
        stride: int,
        do_normalize: bool,
        do_medianfilter: bool,
        do_lowpassfilter: bool,
    ):
        """处理单个 FAST5 文件，使用传入的处理选项。"""
        all_chunks = []
        try:
            with get_fast5_file(fast5_path, mode="r") as f5:
                for read in f5.get_reads():
                    # Step 1: raw → pA
                    channel_info = read.handle[read.global_key + 'channel_id'].attrs
                    offset = int(channel_info['offset'])
                    scaling = channel_info['range'] / channel_info['digitisation']
                    raw = read.handle[read.raw_dataset_name][:]
                    signal = np.array(scaling * (raw + offset), dtype=np.float32)

                    # Step 2: normalize
                    if do_normalize:
                        signal = nanopore_normalize(signal)
                    if signal.size == 0 or np.isnan(signal).any():
                        print(f"⚠️ Invalid signal after normalization for read {read.read_id}, skipped.")
                        continue

                    # Step 3: median filter
                    if do_medianfilter:
                        signal = medfilt(signal, kernel_size=5).astype(np.float32)

                    # Step 4: low-pass filter
                    if do_lowpassfilter:
                        fs_from_read = self.get_sampling_rate_from_read(read)
                        fs = fs_from_read if fs_from_read is not None else self.default_fs
                        try:
                            filtered_signal = nanopore_filter(signal, fs=fs)
                        except Exception as e:
                            print(f"⚠️ Filtering failed for read {read.read_id} (fs={fs}): {e}, skipped.")
                            continue
                        if filtered_signal.size == 0 or np.isnan(filtered_signal).any():
                            print(f"⚠️ Invalid signal after filtering for read {read.read_id}, skipped.")
                            continue
                        signal = filtered_signal

                    # Step 5: chunking
                    chunks = self._sliding_window_chunks_with_pos(signal, window_size, stride)
                    if not chunks:
                        print(f"⚠️ Read {read.read_id} too short (<{window_size} points), skipped.")
                        continue

                    for ch in chunks:
                        all_chunks.append({
                            'read_id': read.read_id,
                            'chunk_start_pos': ch['chunk_start'],
                            'chunk_end_pos': ch['chunk_end'],
                            'chunk_data': ch['chunk_data']
                        })

            # Save
            if all_chunks:
                basename = os.path.basename(fast5_path).rsplit('.', 1)[0]
                save_path = os.path.join(output_dir, f"{basename}.npy")
                np.save(save_path, all_chunks)
                print(f"✅ Saved {len(all_chunks)} chunks from {basename} to {save_path}")
            else:
                print(f"⚠️ No valid chunks in {os.path.basename(fast5_path)}, skipping save.")

        except Exception as e:
            print(f"❌ Critical error processing {fast5_path}: {e}")

    def to_chunks(
        self,
        output_dir: str,
        window_size: int = 32,
        stride: int = 8,
        do_normalize: bool = True,
        do_medianfilter: bool = False,
        do_lowpassfilter: bool = False,
        n_jobs: int = -1,
    ):
        """
        并行处理整个 FAST5 目录，生成 chunked .npy 文件。

        Args:
            output_dir (str): 输出目录。
            window_size (int): 每个 chunk 的长度（默认 32）。
            stride (int): 滑动步长（默认 8）。
            do_normalize (bool): 是否执行 median-MAD 归一化。
            do_medianfilter (bool): 是否应用中值滤波。
            do_lowpassfilter (bool): 是否应用低通滤波。
            n_jobs (int): 并行进程数。-1 表示使用全部 CPU 核心。
        """
        os.makedirs(output_dir, exist_ok=True)

        if n_jobs == -1:
            n_jobs = cpu_count()

        print(f"📁 Processing {len(self.fast5_files)} FAST5 files from: {self.fast5_dir}")
        print(f"ParallelGroup: using {n_jobs} processes")
        print(f"⚙️  Signal pipeline:")
        print(f"    - Normalize: {'ON' if do_normalize else 'OFF'}")
        print(f"    - Median Filter: {'ON' if do_medianfilter else 'OFF'}")
        print(f"    - Low-pass Filter: {'ON' if do_lowpassfilter else 'OFF'}")
        print(f"💾 Saving chunks to: {output_dir}")

        # 构造参数列表
        args_list = [
            (
                fp,
                output_dir,
                window_size,
                stride,
                do_normalize,
                do_medianfilter,
                do_lowpassfilter,
            )
            for fp in self.fast5_files
        ]

        # 使用 pathos 并行处理
        with ProcessPool(nodes=n_jobs) as pool:
            pool.map(self._process_single_fast5_wrapper, args_list)

    def _process_single_fast5_wrapper(self, args):
        """供 pathos 调用的包装器。"""
        (
            fast5_path,
            output_dir,
            window_size,
            stride,
            do_normalize,
            do_medianfilter,
            do_lowpassfilter,
        ) = args
        return self._process_single_fast5(
            fast5_path=fast5_path,
            output_dir=output_dir,
            window_size=window_size,
            stride=stride,
            do_normalize=do_normalize,
            do_medianfilter=do_medianfilter,
            do_lowpassfilter=do_lowpassfilter,
        )
