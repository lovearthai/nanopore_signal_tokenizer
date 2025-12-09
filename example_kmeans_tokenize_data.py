# example_tokenize_data.py
import numpy as np
from nanopore_signal_tokenizer.kmeans_tokenizer import KmeansTokenizer

tokenizer = KmeansTokenizer(
    window_size=32,
    stride=5,
    centroids_path="/mnt/nas_syy/dataset/huada_rna_80G/rvq_home/nanopore_signal_tokenizer/nanopore_signal_tokenizer/kmeans/0.4b_centroids_8192.npy",
)

# 模拟一段 1200 点的信号（~240ms @ 5kHz）
signal = np.random.randn(1200).astype(np.float32) * 5 + 100

# 获取全部层级 token (L1–L4)
tokens_all = tokenizer.tokenize_data(signal)
print(tokens_all)
# <|bwav:L1_5336|><|bwav:L2_7466|><|bwav:L3_6973|><|bwav:L4_6340|>...
