"""Test CUDA device and PyTorch installation."""
import torch
import sys

def test_cuda():
    print(f"Python: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    assert torch.cuda.is_available(), "CUDA is not available!"
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Device: {device_name}")
    print(f"GPU VRAM: {vram_gb:.2f} GB")
    assert "RTX" in device_name or "GeForce" in device_name or "NVIDIA" in device_name
    print("CUDA Verification PASSED!")

if __name__ == "__main__":
    test_cuda()
