import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict


class NanoporeSignalDataset(Dataset):
    def __init__(self, shards_dir, max_cache_size=32):
        """
        Args:
            shards_dir (str): Directory containing shards.npy and shards.json.
            max_cache_size (int): Max number of memmap files to keep open (per process).
        """
        meta_path = os.path.join(shards_dir, "shards.json")
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        self.shard_info = meta["shards"]  # [{"path": "...", "num_samples": N}, ...]
        self.chunk_size = meta["chunk_size"]
        self.dtype = np.dtype(meta["dtype"])
        self.shards_dir = shards_dir
        self.max_cache_size = max_cache_size

        # Build global offsets for fast indexing
        self.offsets = [0]
        for info in self.shard_info:
            self.offsets.append(self.offsets[-1] + info["num_samples"])
        self.total_samples = self.offsets[-1]

        # LRU cache for memmap objects: {shard_path: memmap_array}
        self._cache = OrderedDict()

    def _get_memmap(self, shard_path):
        """Get memmap array with LRU caching."""
        if shard_path in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(shard_path)
            return self._cache[shard_path]
        
        # Open new memmap
        memmap_arr = np.load(shard_path, mmap_mode='r')
        
        # Evict oldest if cache is full
        if len(self._cache) >= self.max_cache_size:
            self._cache.popitem(last=False)  # Remove least recently used
        
        self._cache[shard_path] = memmap_arr
        return memmap_arr

    def _find_shard(self, idx):
        """Binary search to find which shard contains global index `idx`."""
        lo, hi = 0, len(self.offsets) - 2  # valid shard indices: 0 to N-1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.offsets[mid] <= idx < self.offsets[mid + 1]:
                return mid
            elif idx < self.offsets[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        return lo  # fallback (should not happen if idx in range)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.total_samples})")

        shard_id = self._find_shard(idx)
        local_idx = idx - self.offsets[shard_id]
        shard_filename = self.shard_info[shard_id]["path"]
        shard_path = os.path.join(self.shards_dir, shard_filename)

        # Get cached or new memmap
        data = self._get_memmap(shard_path)
        chunk = data[local_idx]  # shape: (chunk_size,)
        return torch.from_numpy(chunk.copy()).unsqueeze(0)  # shape: (1, chunk_size)
