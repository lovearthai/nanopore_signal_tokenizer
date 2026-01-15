# nanopore_signal_tokenizer/fast5.py

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import os
import json  # 【新增】用于保存全局汇总报告
import numpy as np
import glob
from ont_fast5_api.fast5_interface import get_fast5_file
from .signal import (
    nanopore_remove_spikes,
    nanopore_normalize_novel,
    nanopore_normalize_huada,
    nanopore_repair_errors 
    )
from scipy.signal import medfilt
from pathos.multiprocessing import ProcessPool
from multiprocessing import cpu_count
import tqdm
from scipy.ndimage import median_filter
from typing import List, Dict, Any  # 【新增】类型提示，提升可读性
import time

class Fast5Dir:
    """
    将 Nanopore 原始 .fast5 文件批量转换为预处理后的 chunked .npy 文件。

    📌 信号处理流水线（所有步骤在 to_chunks_parallel 中控制）：
        1. 【缩放】raw → pA；
        2. 【归一化】median-MAD（可选）；
        3. 【中值滤波】kernel=5（可选）；
        4. 【低通滤波】Butterworth（可选）；
        5. 【分块】滑动窗口 + 末尾兜底 + 多头裁剪。

    📦 输出：每个 .fast5 → 一个 .npy，内容为 list[dict]，含 read_id、位置、chunk_data、head_cut。
    
    【新增说明 - 全局汇总】
    本类在完成全部文件处理后，会自动生成一份全局统计摘要（global_summary.json），
    包含：总文件数、成功/失败文件数、有效 reads 数、各类跳过原因计数、总 chunks 数、
    每个输出文件的 chunk 数量分布、以及严重错误日志。便于质量控制与流程监控。
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

        # 【新增】全局统计字典，用于收集整个处理过程的汇总信息
        self.global_stats = {
            "total_fast5_files": len(self.fast5_files),
            "processed_fast5_files": 0,
            "skipped_fast5_files": 0,
            "total_reads": 0,
            "valid_reads": 0,
            "skipped_reads": {
                "no_reads_in_file": 0,
                "signal_extraction_failed": 0,
                "signal_out_of_raw_range": 0,
                "signal_out_of_med_range": 0,
                "signal_out_of_norm_range": 0,
                "empty_or_nan_after_norm": 0,
                "filtering_failed": 0,
                "too_short": 0,
            },
            "total_chunks": 0,
            "chunks_per_file": {},  # key: basename, value: chunk count
            "errors": [],  # 记录严重错误（如文件无法打开）
            "processing_start_time": None,
            "processing_end_time": None,
            "total_elapsed_seconds": 0,
        }

    @staticmethod
    def get_sampling_rate_from_read(read):
        """从 FAST5 read 中提取采样率，失败时返回 None。"""
        try:
            channel_info = read.handle[read.global_key + 'channel_id'].attrs
            return int(channel_info['sampling_rate'])
        except Exception:
            return None

    def _sliding_window_chunks_with_tail(
        self,
        signal: np.ndarray,
        window_size: int,
        stride: int,
        tail_threshold: int,
    ):
        """
        对一维信号进行滑动窗口切分，并在末尾不足一个窗口但满足最小长度时补充一个 chunk。

        切分策略：
          - 主体使用固定 stride 滑动；
          - 若末尾剩余片段长度 ≥ tail_threshold，则从信号末尾倒数 window_size 点再切一个 chunk；
          - 避免与最后一个滑动窗口重复。

        Args:
            signal (np.ndarray): 输入一维信号。
            window_size (int): 每个 chunk 的长度（点数）。
            stride (int): 滑动步长（点数）。
            tail_threshold (int): 触发末尾补 chunk 的最小剩余长度。

        Returns:
            List[Dict]: 每个元素包含 'chunk_start', 'chunk_end', 'chunk_data'。
        """
        n_points = len(signal)
        if n_points < window_size:
            return []

        chunks = []
        start = 0
        end = 0
        # 主滑动窗口循环
        while start + window_size <= n_points:
            end = start + window_size
            chunks.append({
                'chunk_start': start,
                'chunk_end': end,
                'chunk_data': signal[start:end].copy()
            })
            start += stride

        # 末尾兜底：若剩余部分足够长且未被覆盖，则从末尾切一个完整窗口
        if n_points - end  >= tail_threshold:
            chunks.append({
                'chunk_start': start,
                'chunk_end': n_points,
                'chunk_data': signal[n_points-window_size:n_points].copy()
            })

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
        cut_head_all: int,
        cut_head_step: int,
        tail_threshold: int,
        max_chunks_per_file: int = 100000,
        signal_min_value: int = -1000,
        signal_max_value: int = 1000,
        normal_min_value: float = -10.0,
        normal_max_value: float = 10.0
    ):
        """
        【增强版】处理单个 FAST5 文件，将 chunks 按数量分片保存，并更新全局统计。

        新增行为：
          - 在函数内部维护 local_stats，记录本文件的处理结果；
          - 处理结束后返回 local_stats，供主进程聚合到 global_stats。

        Args:
            ...（原有参数不变）...
            max_chunks_per_file (int): 每个输出文件最大 chunk 数量，默认 100000。

        Returns:
            dict: 本地统计信息，包含本文件的 reads/chunks/errors 计数。
        """
        NORM_SIG_MIN = normal_min_value
        NORM_SIG_MAX = normal_max_value

        # 【新增】初始化本文件的局部统计
        local_stats = {
            "fast5_path": fast5_path,
            "reads_total": 0,
            "reads_valid": 0,
            "reads_skipped": {k: 0 for k in self.global_stats["skipped_reads"]},
            "chunks_saved": 0,
            "output_files": [],
            "error": None
        }

        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.basename(fast5_path).rsplit('.', 1)[0]
        buffer = []
        part_idx = 0

        try:
            with get_fast5_file(fast5_path, mode="r") as f5:
                read_ids = f5.get_read_ids()
                if not read_ids:
                    print(f"⚠️ No reads found in {fast5_path}")
                    local_stats["reads_skipped"]["no_reads_in_file"] += 1
                    return local_stats

                reads = list(f5.get_reads())
                local_stats["reads_total"] = len(reads)

                for read in reads:
                    # --- 信号预处理（同前）---
                    #
                    try:
                        channel_info = read.handle[read.global_key + 'channel_id'].attrs
                        offset = int(channel_info['offset'])
                        scaling = channel_info['range'] / channel_info['digitisation']
                        raw = read.handle[read.raw_dataset_name][:]
                        signal_raw = np.array(scaling * (raw + offset), dtype=np.float32)
                    except Exception as e:
                        print(f"⚠️ Failed to extract signal for read {read.read_id}: {e}, skipped.")
                        local_stats["reads_skipped"]["signal_extraction_failed"] += 1
                        continue

                    # 使用 nanopore_repair_error 过滤极端异常值
                    signal_clr = nanopore_repair_errors(signal_raw, signal_min_value, signal_max_value)
                    # 再次检查是否仍有越界（理论上应已修复，但保险起见）
                    if np.any(signal_clr < signal_min_value) or np.any(signal_clr > signal_max_value):
                        print(f"⚠️ Read {read.read_id} still out of raw range after filtering, skipped.")
                        local_stats["reads_skipped"]["signal_out_of_raw_range"] += 1
                        continue
                    # 把nanopore_repair_error没有修复的包含在[signal_min_value,signal_max_value]范围内的数据给修复掉
                    signal_nos = nanopore_remove_spikes(signal_clr, window_size=window_size, spike_threshold=5.0)
                    
                    # 因为repair里有abs(raw-med)这一步，所以必须在这步前修复数据，把极端大的值给干掉,也就是必须repair
                    signal_nom, global_mad = nanopore_normalize_novel(signal_nos)

                    #signal_nom = nanopore_repair_normal(signal_nom, NORM_SIG_MIN, NORM_SIG_MAX,window_size=33)
                    # 应用中值滤波（注意：此处原代码已强制开启，但参数控制仍保留）
                    signal_med = medfilt(signal_nom, kernel_size=5).astype(np.float32)
                    
                    signal = signal_med
                    signal_chk = signal_nom

                    # 检查归一化后是否在允许范围内
                    if np.any(signal_chk < NORM_SIG_MIN) or np.any(signal_chk > NORM_SIG_MAX):
                        actual_min = signal.min()
                        actual_max = signal.max()
                        print(f"⚠️ Ignored read {fast5_path} {read.read_id} due to out-of-range signal values after normalization. "
                              f"Actual range: [{actual_min:.3f}, {actual_max:.3f}], "
                              f"Allowed: [{NORM_SIG_MIN}, {NORM_SIG_MAX}]")

                        # 🔁【恢复】多阶段上下文打印（来自 fast5.bak.py）
                        outlier_mask = (signal_chk < NORM_SIG_MIN) | (signal_chk > NORM_SIG_MAX)
                        outlier_indices = np.where(outlier_mask)[0]
                        # === 新增：过滤掉与前一个异常点距离小于5的点 ===
                        print_half_window_size = 5
                        if outlier_indices.size > 0:
                            keep = [True]  # 第一个点总是保留
                            last_kept = outlier_indices[0]
                            for idx in outlier_indices[1:]:
                                if idx - last_kept >= print_half_window_size:
                                    keep.append(True)
                                    last_kept = idx
                                else:
                                    keep.append(False)
                            outlier_indices = outlier_indices[keep]
                        max_print = 5
                        for i, idx in enumerate(outlier_indices[:max_print]):
                            start = max(0, idx - print_half_window_size)
                            end = min(len(signal_chk), idx + print_half_window_size)
                            context = signal[start:end]
                            context_raw = signal_raw[start:end]
                            context_clr = signal_clr[start:end]
                            context_nos = signal_nos[start:end]
                            context_nom = signal_nom[start:end]
                            context_med = signal_med[start:end]
                            print(f"  → Outlier #{i+1} at index {idx}: value = {signal[idx]:.3f}")
                            print(f"    Raw ({start}–{end-1}): {[f'{x:.3f}' for x in context_raw]}")
                            print(f"    Clr ({start}–{end-1}): {[f'{x:.3f}' for x in context_clr]}")
                            print(f"    Nom ({start}–{end-1}): {[f'{x:.3f}' for x in context_nom]}")
                            print(f"    Nos ({start}–{end-1}): {[f'{x:.3f}' for x in context_nos]}")
                            print(f"    Med ({start}–{end-1}): {[f'{x:.3f}' for x in context_med]}")
                        if len(outlier_indices) > max_print:
                            print(f"  → ... and {len(outlier_indices) - max_print} more outliers.")

                    if np.any(signal_nom < NORM_SIG_MIN) or np.any(signal_nom > NORM_SIG_MAX):
                        local_stats["reads_skipped"]["signal_out_of_norm_range"] += 1
                        continue

                    if signal.size == 0 or np.isnan(signal).any():
                        print(f"⚠️ Invalid signal after normalization for read {read.read_id}, skipped.")
                        local_stats["reads_skipped"]["empty_or_nan_after_norm"] += 1
                        continue

                    if do_lowpassfilter:
                        fs_from_read = self.get_sampling_rate_from_read(read)
                        fs = fs_from_read if fs_from_read is not None else self.default_fs
                        try:
                            filtered_signal = nanopore_filter(signal, fs=fs)
                        except Exception as e:
                            print(f"⚠️ Filtering failed for read {read.read_id} (fs={fs}): {e}, skipped.")
                            local_stats["reads_skipped"]["filtering_failed"] += 1
                            continue
                        if filtered_signal.size == 0 or np.isnan(filtered_signal).any():
                            print(f"⚠️ Invalid signal after filtering for read {read.read_id}, skipped.")
                            local_stats["reads_skipped"]["filtering_failed"] += 1
                            continue
                        signal = filtered_signal

                    if len(signal) < window_size:
                        print(f"⚠️ Read {read.read_id} too short (<{window_size} points), skipped.")
                        local_stats["reads_skipped"]["too_short"] += 1
                        continue

                    # ✅ 此 read 通过所有检查
                    local_stats["reads_valid"] += 1

                    max_head = min(cut_head_all, len(signal) - 1)
                    head_cuts = list(range(0, max_head + 1, cut_head_step)) or [0]

                    read_chunks = []
                    for head_cut in head_cuts:
                        if head_cut >= len(signal):
                            continue
                        trimmed_signal = signal[head_cut:]
                        chunks = self._sliding_window_chunks_with_tail(
                            trimmed_signal, window_size, stride, tail_threshold
                        )
                        for ch in chunks:
                            read_chunks.append({
                                'read_id': read.read_id,
                                'head_cut': head_cut,
                                'chunk_start_pos': head_cut + ch['chunk_start'],
                                'chunk_end_pos': head_cut + ch['chunk_end'],
                                'chunk_data': ch['chunk_data']
                            })

                    if read_chunks:
                        buffer.extend(read_chunks)

                        if len(buffer) >= max_chunks_per_file:
                            save_path = os.path.join(output_dir, f"{basename}_part{part_idx:05d}.npy")
                            np.save(save_path, buffer[:max_chunks_per_file])
                            local_stats["chunks_saved"] += len(buffer[:max_chunks_per_file])
                            local_stats["output_files"].append(save_path)
                            print(f"✅ Saved {len(buffer[:max_chunks_per_file])} chunks to {save_path}")
                            buffer = buffer[max_chunks_per_file:]
                            part_idx += 1

                # 保存剩余 buffer
                if buffer:
                    save_path = os.path.join(output_dir, f"{basename}_part{part_idx:05d}.npy")
                    np.save(save_path, buffer)
                    saved_count = len(buffer)
                    local_stats["chunks_saved"] += saved_count
                    local_stats["output_files"].append(save_path)
                    print(f"✅ Saved final {saved_count} chunks to {save_path}")

        except Exception as e:
            error_msg = f"❌ Critical error processing {fast5_path}: {e}"
            print(error_msg)
            local_stats["error"] = str(e)
            return local_stats

        return local_stats
    def _aggregate_and_save_global_summary(self, output_dir: str):
        """
        【新增】聚合所有局部统计，生成并保存全局汇总报告到 global_summary.json。
        """
        summary_path = os.path.join(output_dir, "global_summary.json")
        try:
            # 补充时间信息
            self.global_stats["processing_end_time"] = time.time()
            self.global_stats["total_elapsed_seconds"] = (
                self.global_stats["processing_end_time"] - self.global_stats["processing_start_time"]
            )

            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(self.global_stats, f, indent=4, ensure_ascii=False)
            print(f"📊 全局汇总已保存至: {summary_path}")
        except Exception as e:
            print(f"⚠️ 无法保存全局汇总: {e}")

    def to_chunks(
        self,
        output_dir: str,
        window_size: int = 32,
        stride: int = 8,
        do_normalize: bool = True,
        do_medianfilter: bool = False,
        do_lowpassfilter: bool = False,
        cut_head_all: int = 5,
        cut_head_step: int = 2,
        tail_threshold: int = 16,
        n_jobs: int = -1,
        signal_min_value: int = -1000,
        signal_max_value: int = 1000,
        normal_min_value: float = -5.0,
        normal_max_value: float = 5.0
    ):
        """
        并行处理整个 FAST5 目录，生成 chunked .npy 文件。

        🎯 多头裁剪说明：
            为适配下游 CNN 的下采样 stride（如 12），需覆盖所有可能的输入对齐相位。
            通过设置 cut_head_all=11, cut_head_step=1，可生成 12 种起始偏移（0~11），
            确保模型学习到平移鲁棒的 token 表示。

        🎯 末尾兜底说明：
            当滑动窗口结束后，若剩余信号长度 ≥ tail_threshold，
            则从信号末尾强制切出一个完整 window，避免信息浪费。

        【新增】全局汇总：
            所有文件处理完毕后，自动聚合各进程返回的局部统计，
            生成 global_summary.json，包含完整 QC 指标。

        Args:
            output_dir (str): 输出目录。
            window_size (int): 每个 chunk 的长度（默认 32）。
            stride (int): 滑动步长（默认 8）。
            do_normalize (bool): 是否执行 median-MAD 归一化。
            do_medianfilter (bool): 是否应用中值滤波。
            do_lowpassfilter (bool): 是否应用低通滤波。
            cut_head_all (int): 最大开头裁剪长度（inclusive），建议设为 stride-1。
            cut_head_step (int): 裁剪步长，控制相位覆盖密度。
            tail_threshold (int): 末尾最小保留点数，用于决定是否补 chunk。
            n_jobs (int): 并行进程数。-1 表示使用全部 CPU 核心。
        """
        os.makedirs(output_dir, exist_ok=True)

        if n_jobs == -1:
            n_jobs = cpu_count()

        # 日志输出
        head_cuts_preview = list(range(0, min(cut_head_all + 1, 20), cut_head_step))  # 防止打印过长
        if cut_head_all >= 20:
            head_cuts_preview.append("...")

        print(f"📁 Processing {len(self.fast5_files)} FAST5 files from: {self.fast5_dir}")
        print(f"ParallelGroup: using {n_jobs} processes")
        print(f"⚙️  Signal pipeline:")
        print(f"    - Normalize: {'ON' if do_normalize else 'OFF'}")
        print(f"    - Median Filter: {'ON' if do_medianfilter else 'OFF'}")
        print(f"    - Low-pass Filter: {'ON' if do_lowpassfilter else 'OFF'}")
        print(f"    - Head cuts: all={cut_head_all}, step={cut_head_step} → sample phases={head_cuts_preview}")
        print(f"    - Tail threshold: {tail_threshold} (fallback chunk if tail ≥ this)")
        print(f"💾 Saving chunks to: {output_dir}")

        # 【新增】记录开始时间
        import time  # 局部导入避免污染
        self.global_stats["processing_start_time"] = time.time()

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
                cut_head_all,
                cut_head_step,
                tail_threshold,
                signal_min_value,
                signal_max_value,
                normal_min_value,
                normal_max_value
            )
            for fp in self.fast5_files
        ]

        # 使用 pathos 并行处理（支持 pickle 不友好的对象）
        with ProcessPool(nodes=n_jobs) as pool:
            results = pool.map(self._process_single_fast5_wrapper, args_list)

        # 【新增】聚合全局统计
        for res in results:
            if res is None:
                self.global_stats["skipped_fast5_files"] += 1
                continue

            self.global_stats["processed_fast5_files"] += 1
            self.global_stats["total_reads"] += res["reads_total"]
            self.global_stats["valid_reads"] += res["reads_valid"]
            self.global_stats["total_chunks"] += res["chunks_saved"]

            basename = os.path.basename(res["fast5_path"]).rsplit('.', 1)[0]
            self.global_stats["chunks_per_file"][basename] = res["chunks_saved"]

            # 累加各类跳过原因
            for reason, count in res["reads_skipped"].items():
                self.global_stats["skipped_reads"][reason] += count

            # 记录错误
            if res.get("error"):
                self.global_stats["errors"].append({
                    "file": res["fast5_path"],
                    "error": res["error"]
                })

        # 【新增】保存全局汇总
        self._aggregate_and_save_global_summary(output_dir)

    def _process_single_fast5_wrapper(self, args):
        """
        供 pathos.multiprocessing 调用的参数解包包装器。
        """
        (
            fast5_path,
            output_dir,
            window_size,
            stride,
            do_normalize,
            do_medianfilter,
            do_lowpassfilter,
            cut_head_all,
            cut_head_step,
            tail_threshold,
            signal_min_value,
            signal_max_value,
            normal_min_value,
            normal_max_value
        ) = args
        return self._process_single_fast5(
            fast5_path=fast5_path,
            output_dir=output_dir,
            window_size=window_size,
            stride=stride,
            do_normalize=do_normalize,
            do_medianfilter=do_medianfilter,
            do_lowpassfilter=do_lowpassfilter,
            cut_head_all=cut_head_all,
            cut_head_step=cut_head_step,
            tail_threshold=tail_threshold,
            signal_min_value=signal_min_value,
            signal_max_value=signal_max_value,
            normal_min_value=normal_min_value,
            normal_max_value=normal_max_value
        )
