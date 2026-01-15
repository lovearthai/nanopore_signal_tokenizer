import torch
import sys

if len(sys.argv) != 2:
    print("Usage: python inspect_ckpt.py <model.pth>")
    sys.exit(1)

ckpt_path = sys.argv[1]
print(f"📂 Loading {ckpt_path}...")

ckpt = torch.load(ckpt_path, map_location='cpu')

print("\n🔍 Type of checkpoint:", type(ckpt))
print("=" * 60)

# 情况 1: checkpoint 是 dict（最常见）
if isinstance(ckpt, dict):
    print("🔑 Top-level keys:")
    for k in ckpt.keys():
        v = ckpt[k]
        if hasattr(v, 'shape'):
            print(f"  - {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        elif isinstance(v, (int, float, str, bool)):
            print(f"  - {k}: value={v} ({type(v).__name__})")
        else:
            print(f"  - {k}: type={type(v)}")

    # 可选：打印前几个张量的统计信息
    print("\n📊 Sample tensor stats (first 3 parameters):")
    count = 0
    for k, v in ckpt.items():
        if hasattr(v, 'shape') and v.numel() > 0:
            print(f"  {k}: mean={v.float().mean():.4f}, std={v.float().std():.4f}, min={v.min():.4f}, max={v.max():.4f}")
            count += 1
            if count >= 3:
                break

# 情况 2: checkpoint 是整个模型（不推荐保存方式）
elif hasattr(ckpt, 'state_dict'):
    print("⚠️ This checkpoint saved the entire model (not just state_dict).")
    print("Keys in state_dict:")
    for k, v in ckpt.state_dict().items():
        print(f"  - {k}: shape={tuple(v.shape)}")
else:
    print("❓ Unknown checkpoint format.")
