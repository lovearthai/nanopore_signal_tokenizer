import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import median_filter

def nanopore_normalize(norm_signal):
    """
    华大归一化方法
    """
    norm_signal, _, _ = _med_mad_norm(norm_signal)
    return norm_signal

# window_size默认5000，不同于kmeans的滑动窗口
def nanopore_filter_signal(signal_clr, window_size=5000, spike_threshold=5.0):

    # 把nanopore_repair_error没有修复的包含在[signal_min_value,signal_max_value]范围内的数据给修复掉
    signal_nos = _nanopore_remove_spikes(signal_clr, window_size=window_size, spike_threshold=spike_threshold)

    # 因为repair里有abs(raw-med)这一步，所以必须在这步前修复数据，把极端大的值给干掉,也就是必须repair
    signal_nom, global_mad = _nanopore_normalize_hybrid(signal_nos, window_size=window_size)

    #signal_nom = nanopore_repair_normal(signal_nom, NORM_SIG_MIN, NORM_SIG_MAX,window_size=33)
    # 应用中值滤波（注意：此处原代码已强制开启，但参数控制仍保留）
    signal_med = medfilt(signal_nom, kernel_size=5).astype(np.float32)
    return signal_med


def normalize_read(read):
    """对信号进行 Z-score 归一化"""
    channel_info = read.handle[read.global_key + 'channel_id'].attrs
    offset = int(channel_info['offset'])
    scaling = channel_info['range'] / channel_info['digitisation']

    raw = read.handle[read.raw_dataset_name][:]
    scaled = np.array(scaling * (raw + offset), dtype=np.float32)
    norm_signal = nanopore_normalize(scaled)
    return norm_signal, offset, scaling

def _nanopore_normalize_hybrid(signal, window_size=2000, mad_factor=1.4826, min_mad=1.0):
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

def _nanopore_remove_spikes(
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



def _med_mad(data, factor=None, axis=None, keepdims=False):
    if factor is None:
        factor = 1.4826
    dmed = np.median(data, axis=axis, keepdims=True)
    dmad = factor * np.median(abs(data - dmed), axis=axis, keepdims=True)
    if axis is None:
        dmed = dmed.flatten()[0]
        dmad = dmad.flatten()[0]
    elif not keepdims:
        dmed = dmed.squeeze(axis)
        dmad = dmad.squeeze(axis)
    return dmed, dmad
def _med_mad_norm(x, dtype='f4'):
	med, mad = _med_mad(x)
	if mad == 0:
		return np.array([]), med, mad
	else:
		normed_x = (x - med) / mad
		return normed_x.astype(dtype), med, mad