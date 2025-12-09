import numpy as np
from nanopore_signal_tokenizer import KmeansTokenizer

tokenizer = KmeansTokenizer(
    window_size=32,
    stride=5,
    centroids_path="nanopore_signal_tokenizer/kmeans/0.4b_centroids_8192.npy",
)

# 模拟一段 1200 点的信号（~240ms @ 5kHz）
signal = np.random.randn(1200).astype(np.float32) * 5 + 100

tokens_all = tokenizer.tokenize_data(signal)
print(tokens_all)
