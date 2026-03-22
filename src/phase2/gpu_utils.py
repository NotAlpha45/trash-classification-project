"""GPU device and CUDA memory helpers for Phase 2 notebooks/scripts."""

from __future__ import annotations

import gc

import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return CUDA device when available and requested, otherwise CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_device_info(device: torch.device) -> None:
    """Print a concise summary of selected compute device."""
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(index)
        total_gb = torch.cuda.get_device_properties(index).total_memory / (1024**3)
        print(f"Using GPU: {gpu_name} ({total_gb:.2f} GB VRAM)")
    else:
        print("CUDA not available. Falling back to CPU.")


def clear_gpu_memory() -> None:
    """Run Python GC and clear CUDA allocator cache when available."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
