from nanopore_signal_tokenizer import Fast5Dir

# 初始化时指定全局默认 fs=5000（可选）
processor = Fast5Dir(
    fast5_dir="fast5",default_fs= 5000
)

processor.to_chunks_parallel(
    output_dir="fast5_chunks_w12k",
    window_size=12000,
    stride=11400,
    n_proc=1
)
