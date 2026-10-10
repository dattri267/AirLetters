"""Run this first on your laptop:  python src/check_env.py
Confirms CUDA works, measures GPU throughput, and checks the other libraries."""
import platform
import time

print("python", platform.python_version(), platform.system())
try:
    import torch
    print("torch", torch.__version__, "| built for CUDA:", torch.version.cuda)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU OK: {p.name}, {p.total_memory/2**30:.1f} GB")
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(50):
            x @ x
        torch.cuda.synchronize()
        print(f"fp16 matmul: {50*2*4096**3/(time.time()-t)/1e12:.1f} TFLOPS")
    else:
        print("GPU NOT available to PyTorch. Likely a CPU-only wheel. Fix:\n"
              "  1) nvidia-smi   (driver must show a CUDA version)\n"
              "  2) pip uninstall -y torch torchvision\n"
              "  3) pip install torch --index-url https://download.pytorch.org/whl/cu126   (pick the cuXXX that pytorch.org suggests)")
except ImportError:
    print("torch missing: pip install torch --index-url https://download.pytorch.org/whl/cu126")
for m in ("numpy", "pandas", "sklearn", "matplotlib", "cv2", "mediapipe"):
    try:
        mod = __import__(m)
        print(f"{m} {getattr(mod, '__version__', '?')}")
    except ImportError:
        print(f"{m} MISSING -> pip install " + {"cv2": "opencv-python", "sklearn": "scikit-learn"}.get(m, m))
import os
print("CPU cores:", os.cpu_count(), "(use cores-1 workers for extraction)")
