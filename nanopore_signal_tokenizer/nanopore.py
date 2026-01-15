# --- 华大归一化函数 ---
def med_mad(data, factor=None, axis=None, keepdims=False):
    if factor is None:
        factor = 1.4826
    dmed = np.median(data, axis=axis, keepdims=True)
    dmad = factor * np.median(np.abs(data - dmed), axis=axis, keepdims=True)
    if axis is None:
        dmed = dmed.flatten()[0]
        dmad = dmad.flatten()[0]
    elif not keepdims:
        dmed = dmed.squeeze(axis)
        dmad = dmad.squeeze(axis)
    return dmed, dmad

def med_mad_norm(x, dtype='f4'):
    med, mad = med_mad(x)
    if mad == 0:
        return np.array([]), med, mad
    else:
        normed_x = (x - med) / mad
        return normed_x.astype(dtype), med, mad

def nanopore_normalize(norm_signal):
    norm_signal, _, _ = med_mad_norm(norm_signal)
    return norm_signal

from scipy.ndimage import median_filter
# MAD 对“平坦区域”过于敏感
# 在低噪声、高重复性区域（如 homopolymer 区域），信号变化极小，MAD 自然趋近于 0。
# 但 Nanopore 信号即使在“平坦”区也有微小波动，若采样精度高（float32），MAD 可能远小于 1e-3。
# 改进建议（工业级实践）
# 方案 1：提高 min_mad 的下限（最简单有效
def nanopore_normalize_local(signal, window_size=2000, factor=1.4826, min_mad=1.0):
    if window_size % 2 == 0:
        window_size += 1
    local_med = median_filter(signal, size=window_size, mode='reflect')
    abs_dev = np.abs(signal - local_med)
    local_mad = median_filter(abs_dev, size=window_size, mode='reflect')
    local_mad = factor * local_mad
    local_mad = np.clip(local_mad, a_min=min_mad, a_max=None)
    return (signal - local_med) / local_mad


import numpy as np
from scipy.ndimage import median_filter

def nanopore_normalize_hybrid_v1(signal, window_size=2000, mad_factor=1.4826, min_mad=1.0):
    """
    Hybrid normalization:
      - Local median (sliding window) to remove baseline drift
      - Global MAD (robust scale) for stable normalization
    Parameters:
        signal: 1D array
        window_size: size of median filter window (for local med)
        mad_factor: 1.4826 for Gaussian consistency
        min_mad: avoid division by near-zero
    Returns:
        normalized signal
    """
    # Compute global MAD (robust scale estimate)
    global_med = np.median(signal)
    global_mad = mad_factor * np.median(np.abs(signal - global_med))
    global_mad = max(global_mad, min_mad)  # clamp to min_mad
    # Compute local median (to track baseline drift)
    if window_size % 2 == 0:
        window_size += 1
    local_med = median_filter(signal, size=window_size, mode='reflect')
    # Normalize
    normalized = (signal - local_med) / global_mad
    return normalized.astype(np.float32),global_mad

def nanopore_normalize_hybrid(signal, window_size=5000, mad_factor=1.4826, min_mad=1.0):
    """
    Hybrid normalization: remove baseline drift with local median, scale with global MAD of residuals.
    """
    # Ensure odd window size for median filter
    if window_size % 2 == 0:
        window_size += 1
    # Local median to track baseline drift
    # 下面两句代码相当于高通滤波
    local_med = median_filter(signal, size=window_size, mode='reflect')

    # Residual after removing baseline
    residual = signal - local_med
    # Global MAD on residuals (robust scale estimate)
    global_mad = mad_factor * np.median(np.abs(residual))
    global_mad = max(global_mad, min_mad)
    # Normalize by global MAD
    normalized = residual / global_mad
    return normalized.astype(np.float32), global_mad


def nanopore_normalize_new(signal):
    """
    Normalize by subtracting global median and scaling with robust MAD
    estimated from central 98% of residuals (1st to 99th percentile).
    """
    signal_MED = np.median(signal)
    residual = signal - signal_MED

    # Use 1st and 99th percentiles to exclude extreme outliers
    q01, q99 = np.quantile(residual, [0.01, 0.99])
    masked_residual = residual[(residual >= q01) & (residual <= q99)]

    # Robust scale estimate (MAD)
    global_MAD = 1.4826 * np.median(np.abs(masked_residual))
    global_MAD = max(global_MAD, 1.0)  # avoid division by near-zero

    normalized = residual / global_MAD
    return normalized.astype(np.float32), global_MAD  # ✅ fixed variable name

import numpy as np
from scipy.ndimage import median_filter
# 这个函数将会遍历输入的信号数组，并对每个小于 min_value 或大于 max_value 的值进行修复。修复的方法是使用该点附近（包括自身）共 window_size 个点计算出的中位数来替换原始值。注意，对于数组边界处的点，我们只能使用到边界的那些点来计算中位数。
def nanopore_repair_normal(signal, min_value, max_value, window_size):
    """
    高效修复超出 [min_value, max_value] 范围的信号点：
    将异常点替换为以该点为中心、长度为 window_size 的窗口中位数。

    Parameters:
        signal (np.ndarray): 1D 信号数组
        min_value (float): 下界阈值
        max_value (float): 上界阈值
        window_size (int): 滑动窗口大小（必须为奇数）

    Returns:
        np.ndarray: 修复后的信号
    """
    if window_size % 2 == 0:
        raise ValueError("window_size 必须为奇数")

    signal = np.asarray(signal, dtype=np.float32)

    # 创建异常点掩码：True 表示需要修复
    mask = (signal < min_value) | (signal > max_value)

    if not np.any(mask):
        return signal.copy()  # 无异常，直接返回
    # 计算整个信号的滑动中位数（含边界自动处理）
    median_filtered = median_filter(signal, size=window_size, mode='nearest')
    # 仅将异常点替换为中位数，正常点保留原值
    repaired = np.where(mask, median_filtered, signal)
    return repaired
import numpy as np

#明白了！我们去掉 search_full 参数，统一使用一个 search_range（整数）参数，表示：
#对每个异常点，只在左右各 search_range 个采样点内寻找合法值。
#如果在这个窗口内：
#左右都找到 → 用均值；
#只有一侧找到 → 用该侧值；
#都没找到 → 用 min_value 或 max_value 替代
def nanopore_repair_error_bak(signal, min_value, max_value, search_range=10):
    """
    Replace outliers in signal using local neighborhood within `search_range`.
    
    For each point outside [min_value, max_value]:
      - Search up to `search_range` points to the left and right for valid values.
      - If both sides have valid neighbors: use their mean.
      - If only one side has: use that value.
      - If neither side has: clamp to min_value or max_value.
    
    Parameters:
        signal (np.ndarray): 1D input signal.
        min_value (float): Lower bound of valid range.
        max_value (float): Upper bound of valid range.
        search_range (int): Number of points to search left/right (default=10).
    
    Returns:
        np.ndarray: Cleaned signal.
    """
    signal = np.asarray(signal, dtype=np.float32)
    cleaned = signal.copy()
    n = len(signal)

    # Valid mask: True where signal is within bounds
    valid_mask = (signal >= min_value) & (signal <= max_value)

    for i in range(n):
        if valid_mask[i]:
            continue  # Skip valid points

        original_val = signal[i]

        # --- Search left: from i-1 down to max(0, i - search_range) ---
        left_val = None
        start_left = max(0, i - search_range)
        for j in range(i - 1, start_left - 1, -1):  # inclusive of start_left
            if valid_mask[j]:
                left_val = signal[j]
                break

        # --- Search right: from i+1 up to min(n-1, i + search_range) ---
        right_val = None
        end_right = min(n - 1, i + search_range)
        for j in range(i + 1, end_right + 1):  # inclusive of end_right
            if valid_mask[j]:
                right_val = signal[j]
                break

        # --- Decide replacement ---
        if left_val is not None and right_val is not None:
            cleaned[i] = (left_val + right_val) / 2.0
        elif left_val is not None:
            cleaned[i] = left_val
        elif right_val is not None:
            cleaned[i] = right_val
        else:
            # No valid neighbor in search window → clamp
            if original_val > max_value:
                cleaned[i] = max_value
            else:
                cleaned[i] = min_value

    return cleaned

import numpy as np

def nanopore_repair_error(signal, min_value, max_value):
    """
    Fast version: only process outlier indices in increasing order.
    Uses the fact that cleaned[i] depends only on cleaned[i-1],
    and i-1 is always processed before i if we go left-to-right.
    """
    signal = np.asarray(signal, dtype=np.float32)
    if np.any(signal < min_value) or np.any(signal > max_value):
        do_repair = True
    else:
        do_repair = False
    if not do_repair:
        return signal

    cleaned = signal.copy()
    n = cleaned.size

    if n == 0:
        return cleaned

    # Find all outlier indices
    valid_mask = (cleaned >= min_value) & (cleaned <= max_value)
    outlier_indices = np.where(~valid_mask)[0]

    if outlier_indices.size == 0:
        return cleaned
    # Process outliers from left to right (they are already sorted)
    for i in outlier_indices:
        if i < 1:
            # First point: clamp
            if cleaned[0] > max_value:
                cleaned[0] = max_value
            else:
                cleaned[0] = min_value
        else:
            # Use immediate left neighbor (which is already final)
            cleaned[i] = cleaned[i - 1]
    return cleaned


def nanopore_remove_spikes(
    signal,
    window_size=5001,
    mad_factor=1.4826,
    min_mad=1.0,
    spike_threshold=5.0
):
    """
    Detect and remove spikes using global MAD on baseline-removed residual.
    Spikes are repaired using forward-fill (left-to-right).
    
    Returns:
        cleaned: np.ndarray, repaired signal (same shape as input)
    """
    signal = np.asarray(signal, dtype=np.float32)
    
    # 1. Estimate baseline with median filter
    local_med = median_filter(signal, size=window_size, mode='reflect')
    
    # 2. Compute residual
    residual = signal - local_med
    
    # 3. Global MAD on residual
    global_mad = mad_factor * np.median(np.abs(residual))
    global_mad = max(global_mad, min_mad)
    
    # 4. Detect spikes
    is_spike = np.abs(residual) > (spike_threshold * global_mad)
    
    if not np.any(is_spike):
        return signal.copy()

    # 5. Repair spikes using forward-fill
    cleaned = signal.copy()
    outlier_indices = np.where(is_spike)[0]
    for i in outlier_indices:
        if i == 0:
            cleaned[0] = local_med[0]
        else:
            cleaned[i] = cleaned[i - 1]
    return cleaned


from scipy import signal
import numpy as np

def nanopore_filter(signal_data, fs=5000, cutoff=1000, order=6):
    """
    对 Nanopore 信号进行零相位低通滤波

    Args:
        signal_data: 原始电流信号 (1D array)
        fs: 采样率 (Hz), 默认 5000
        cutoff: 截止频率 (Hz), 推荐 800–1500
        order: Butterworth 滤波器阶数, 推荐 4–8

    Returns:
        filtered_signal: 滤波后的信号
    """
    # 归一化截止频率 (0 ~ 1, 1 = Nyquist = fs/2)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    # 设计 Butterworth 低通滤波器
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

    # 使用 filtfilt 实现零相位滤波（无延迟、无相位失真）
    filtered_signal = signal.filtfilt(b, a, signal_data)
     # ✅ 关键修复：确保返回 C-contiguous 的副本，避免负 stride
    return np.ascontiguousarray(filtered_signal, dtype=np.float32)




from scipy.signal import medfilt

