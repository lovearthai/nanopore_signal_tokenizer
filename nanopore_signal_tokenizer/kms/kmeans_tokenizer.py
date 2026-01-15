
from .signal_normal import nanopore_normalize, nanopore_filter_signal
import faiss
import gzip
import json
from tqdm import tqdm
from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np

class KmeansTokenizer:
    """
    Nanopore RVQ Tokenizer 封装类。

    功能：
        - 加载预训练 RVQ 模型
        - tokenize 单个 read / numpy 信号 / 整个 FAST5 目录
    """

    def __init__(
        self,
        centroids_path: str,
    ):
        """
        初始化 tokenizer。
        """
        data = np.load(centroids_path, allow_pickle=True).item()
        self.window_size = data["dimension"]
        self.stride = data["stride"]
        self.index = self._init_worker(data["centroids"])

    def _init_worker(self, centroids):
        d = centroids.shape[1]
        if hasattr(faiss, 'StandardGpuResources'):
        # === GPU 模式 ===
            print("🚀 Initializing FAISS GPU index...")
            res = faiss.StandardGpuResources()  # GPU 资源管理器
            cpu_index = faiss.IndexFlatL2(d)
            cpu_index.add(centroids) # type: ignore
            # 将 CPU 索引搬到 GPU（默认 device=0）
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            # === CPU 回退模式 ===
            print("💻 Using FAISS CPU index...")
            cpu_index = faiss.IndexFlatL2(d)
            cpu_index.add(centroids) # type: ignore
            index = cpu_index
        return index
    
    def _sliding_window_chunks(self, signal):
        """
        对一维信号进行滑动窗口切片。

        Args:
            signal (np.ndarray): 一维归一化信号
            window_size (int): 窗口长度
            stride (int): 步长

        Returns:
            list of tuples: 每个元素是一个三元组 (start, end, vector)，其中：
                            - start 是切片在原始信号中的起始索引
                            - end 是切片在原始信号中的结束索引（不包含）
                            - vector 是切片本身的值
        """
        n_points = len(signal)
        if n_points < self.window_size:
            return []

        chunks_info = []
        start = 0
        while start + self.window_size <= n_points:
            end = start + self.window_size
            chunk = signal[start:end]
            chunks_info.append((start, end, chunk))
            start += self.stride
        return chunks_info

    def tokenize_data(self, signal: np.ndarray) -> str:
        # Normalize
        norm_sig_no_filter = nanopore_normalize(signal)
        norm_sig = nanopore_filter_signal(norm_sig_no_filter) # 进行去噪处理
        if norm_sig.size == 0:
            return ""
        vec_list = []
        chunks_info = self._sliding_window_chunks(norm_sig)
        for _, _, chunk in chunks_info:
            if chunk.size == 0:
                continue
            vec_list.append(chunk)
        if not vec_list:
            return ""
        try:
            X = np.stack(vec_list, axis=0).astype(np.float32)
        except Exception:
            return ""
        _, I = self.index.search(X, 1) # type: ignore
        cluster_ids = I[:, 0].tolist()

        tokens = ''.join(f"<|bwav:{int(cid)}|>" for cid in cluster_ids)

        return tokens


    def tokenize_read(self, read) -> str:
        """
        直接 tokenize 一个 ont_fast5_api read 对象，返回格式化 token 字符串。

        Args:
            read: fast5 read object
            token_type: "L1", "L2", "L3", or "L4"

        Returns:
            str: formatted token string
        """
        # --- Scale ---
        channel_info = read.handle[read.global_key + 'channel_id'].attrs
        offset = int(channel_info['offset'])
        scaling = channel_info['range'] / channel_info['digitisation']
        raw = read.handle[read.raw_dataset_name][:]
        scaled = np.array(scaling * (raw + offset), dtype=np.float32)

        return self.tokenize_data(scaled)

 
    def tokenize_fast5(self, fast5_path: str, output_path: str):
        print(f"✅ Process {fast5_path}")
        """内部方法：处理单个 FAST5 → JSONL.GZ"""
        results = []
        with get_fast5_file(fast5_path, mode="r") as f5:
            for read in tqdm(f5.get_reads()):
                try:
                    token_str = self.tokenize_read(read)
    
                    results.append({
                        "id": read.read_id,
                        "text": token_str
                    })
                except Exception as e:
                    print(f"❌ Error on read {read.read_id} in {fast5_path}: {e}")
                    continue
    
        # Save
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        print(f"✅ Wrote {len(results)} reads to {output_path}")
