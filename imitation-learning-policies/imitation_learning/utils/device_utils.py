import torch


def get_default_device() -> torch.device:
    """Pick CUDA when available, then Intel XPU, then the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")
