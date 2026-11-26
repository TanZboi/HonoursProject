from __future__ import annotations


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def reset_peak_memory() -> None:
    """Start a new peak-memory measurement."""
    if _cuda_available():
        import torch

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> float | None:
    """Peak GPU memory since the last reset, in MB (None without CUDA)."""
    if not _cuda_available():
        return None
    import torch

    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20
