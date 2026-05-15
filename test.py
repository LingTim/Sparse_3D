import torch
from diffusers import DiffusionPipeline

print(f"PyTorch 版本: {torch.version.__version__}")
print(f"CUDA 版本: {torch.version.cuda}")

try:
    import diff_gaussian_rasterization
    print("3DGS Rasterizer: 載入成功")
except Exception as e:
    print(f"3DGS Rasterizer: 失敗 ({e})")

try:
    # 測試一下 Zero123++ 依賴的元件
    print(f"Diffusers 版本: {DiffusionPipeline.__name__} 載入成功")
except Exception as e:
    print(f"Zero123++ 組件: 失敗 ({e})")