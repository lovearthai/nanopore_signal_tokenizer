# nanopore_signal_tokenizer/fast5.py

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
from multiprocessing import Pool, cpu_count
import functools
import os
import numpy as np
import glob
from ont_fast5_api.fast5_interface import get_fast5_file
from .nanopore import nanopore_normalize, nanopore_filter
from scipy.signal import medfilt

class Fast5Dir:
    """
    处理一个包含 .fast5 文件的目录，将其转换为 chunked .npy 文件。
    
    信号处理流程：
        raw → scaled → med-mad normalized → Butterworth low-pass filtered → chunked
    
    每个 .fast5 → 一个 .npy，每个 chunk 是 dict：
        {
            'read_id': str,
            'chunk_start_pos': int,
            'chunk_end_pos': int,
            'chunk_data': np.ndarray (shape=(window_size,))
        }
    """

    def __init__(self, fast5_dir: str):
        """
        初始化处理器。

        Args:
            fast5_dir (str): 包含 .fast5 文件的目录路径。
            default_fs (int): 全局默认采样率（Hz），当 read 中无 sampling_rate 时使用。
        """
        if not os.path.isdir(fast5_dir):
            raise ValueError(f"FAST5 directory does not exist: {fast5_dir}")
        self.fast5_dir = fast5_dir
        self.fast5_files = sorted(glob.glob(os.path.join(fast5_dir, "*.fast5")))
        self.default_fs = 5000
        if not self.fast5_files:
            raise FileNotFoundError(f"No .fast5 files found in {fast5_dir}")

    @staticmethod
    def get_sampling_rate_from_read(read):
        """尝试从 read 的 metadata 中提取 sampling_rate"""
        try:
            channel_info = read.handle[read.global_key + 'channel_id'].attrs
            return int(channel_info['sampling_rate'])
        except Exception:
            return None  # 表示未找到

    def _sliding_window_chunks_with_pos(self, signal, window_size=32, stride=8):
        n_points = len(signal)
        if n_points < window_size:
            return []

        chunks = []
        start = 0
        while start + window_size <= n_points:
            end = start + window_size
            chunk_data = signal[start:end].copy()
            chunks.append({
                'chunk_start': start,
                'chunk_end': end,
                'chunk_data': chunk_data
            })
            start += stride
        return chunks

    def _process_single_fast5(
        self,
        fast5_path: str,
        output_dir: str,
        window_size: int,
        stride: int,
    ):
        all_chunks = []
        try:
            with get_fast5_file(fast5_path, mode="r") as f5:
                for read in f5.get_reads():
                    # --- 1. 缩放原始信号 ---
                    channel_info = read.handle[read.global_key + 'channel_id'].attrs
                    offset = int(channel_info['offset'])
                    scaling = channel_info['range'] / channel_info['digitisation']
                    raw = read.handle[read.raw_dataset_name][:]
                    signal = np.array(scaling * (raw + offset), dtype=np.float32)

                    # --- 2. 归一化 ---
                    if do_normalize:
                        signal = nanopore_normalize(signal)
                    if signal.size == 0:
                        print(f"⚠️ Empty after normalization for read {read.read_id}, skipped.")
                        continue
                    
                    # 原始信号: raw_signal (采样率 5000 Hz)
                    # 典型 k-mer 持续时间 ≈ 2–5 ms → 对应 10–25 个采样点

                    # 推荐窗口大小：3 ~ 7（奇数）
                    if do_medfilter:
                        signal = medfilt(signal, kernel_size=5)


                    # --- 3. 确定采样率：优先 read 自带，否则用全局默认 ---
                    if do_lowpassfilter:
                        try:
                            fs_from_read = self.get_sampling_rate_from_read(read)
                            fs = fs_from_read if fs_from_read is not None else self.default_fs
                            filtered_signal = nanopore_filter(
                                signal, fs=fs
                            )
                        except Exception as e:
                            print(f"⚠️ Filtering failed for read {read.read_id} (fs={fs}): {e}, skipped.")
                            continue

                    if filtered_signal.size == 0 or np.isnan(filtered_signal).any():
                        print(f"⚠️ Invalid signal after filtering for read {read.read_id}, skipped.")
                        continue

                    # --- 5. 切 chunk ---
                    chunks = self._sliding_window_chunks_with_pos(
                        filtered_signal, window_size=window_size, stride=stride
                    )
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

            # --- 保存结果 ---
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
    ):
        """
        将整个 FAST5 目录转换为 chunked .npy 文件。

        Args:
            output_dir (str): 输出目录。
            window_size (int): 每个 chunk 的长度。
            stride (int): 滑动窗口步长。
            cutoff (int): 滤波截止频率（Hz）。
            order (int): Butterworth 滤波器阶数。
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Processing {len(self.fast5_files)} FAST5 files from: {self.fast5_dir}")
        print(f"⚙️  Signal pipeline: scale → normalize → filter (cutoff={cutoff}Hz, order={order}) → chunk")
        print(f"   ⏱️ Sampling rate: per-read if available, else global default fs={self.default_fs} Hz")
        print(f"💾 Saving chunks to: {output_dir}")

        for i, fp in enumerate(self.fast5_files):
            print(f"\n[{i+1}/{len(self.fast5_files)}] Processing: {os.path.basename(fp)}")
            self._process_single_fast5(
                fp,
                output_dir=output_dir,
                window_size=window_size,
                stride=stride,
            )

    # 在 Fast5Dir 类中
    def to_chunks_parallel(
        self,
        output_dir: str,
        window_size: int = 32,
        stride: int = 8,
        n_jobs: int = None
    ):
        from pathos.multiprocessing import ProcessPool
        import os

        os.makedirs(output_dir, exist_ok=True)

        if n_jobs is None or n_jobs == -1:
            from multiprocessing import cpu_count
            n_jobs = cpu_count()

        print(f"📁 Processing {len(self.fast5_files)} FAST5 files from: {self.fast5_dir}")
        print(f"ParallelGroup: using {n_jobs} processes")

        # 准备参数：每个任务是一个 fast5 文件路径
        # 我们将调用 self._process_single_fast5(fp, ...)
        args_list = [
            (fp, output_dir, window_size, stride)
            for fp in self.fast5_files
        ]

        # 使用 pathos 的 ProcessPool，它能 pickle 方法
        with ProcessPool(nodes=n_jobs) as pool:
            results = pool.map(self._process_single_fast5_wrapper_for_pathos, args_list)

        for res in results:
            print(res)


    def _process_single_fast5_wrapper_for_pathos(self, args):
        """供 pathos 调用的包装器（仍是类方法）"""
        fp, output_dir, window_size, stride = args
        return self._process_single_fast5(
            fp,
            output_dir=output_dir,
            window_size=window_size,
            stride=stride
        )
