import numpy as np
from nanopore_signal_tokenizer import KmeansTokenizer

tokenizer = KmeansTokenizer(
    centroids_path="../models/centroids_with_meta.npy",
)
window_size = tokenizer.window_size
stride = tokenizer.stride
# 模拟一段 1200 点的信号（~240ms @ 5kHz）
signal = np.random.randn(1200).astype(np.float32) * 5 + 100

tokens_all = tokenizer.tokenize_data(signal)
print(tokens_all)
