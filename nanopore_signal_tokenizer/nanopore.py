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

