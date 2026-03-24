"""GPU device and CUDA memory helpers for Phase 2 notebooks/scripts."""

from __future__ import annotations

import gc

import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return CUDA device when available and requested, otherwise CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_gpu_memory_stats(device: torch.device | None = None) -> dict[str, float]:
    """Return CUDA memory stats in GB for the selected device.

    Returns zeros when CUDA is not available.
    """
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "total_gb": 0.0,
            "free_gb": 0.0,
        }

    active_device = device or torch.device("cuda")
    index = active_device.index if active_device.index is not None else 0
    allocated = torch.cuda.memory_allocated(index) / (1024**3)
    reserved = torch.cuda.memory_reserved(index) / (1024**3)
    total = torch.cuda.get_device_properties(index).total_memory / (1024**3)
    free = max(0.0, total - reserved)
    return {
        "allocated_gb": float(allocated),
        "reserved_gb": float(reserved),
        "total_gb": float(total),
        "free_gb": float(free),
    }


def print_device_info(device: torch.device) -> None:
    """Print a concise summary of selected compute device."""
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(index)
        stats = get_gpu_memory_stats(device)
        print(
            f"Using GPU: {gpu_name} ({stats['total_gb']:.2f} GB VRAM, "
            f"allocated {stats['allocated_gb']:.2f} GB, reserved {stats['reserved_gb']:.2f} GB)"
        )
    else:
        print("CUDA not available. Falling back to CPU.")


def print_gpu_memory(prefix: str = "GPU memory", device: torch.device | None = None) -> None:
    """Print current CUDA memory usage and capacity summary."""
    if not torch.cuda.is_available():
        print(f"{prefix}: CUDA unavailable")
        return

    stats = get_gpu_memory_stats(device)
    print(
        f"{prefix}: allocated={stats['allocated_gb']:.2f} GB, "
        f"reserved={stats['reserved_gb']:.2f} GB, free={stats['free_gb']:.2f} GB"
    )


def should_use_half_precision(device: torch.device, use_half_precision: bool) -> bool:
    """Return whether fp16 should be enabled for inference on the given device."""
    return bool(use_half_precision and device.type == "cuda" and torch.cuda.is_available())


def clear_gpu_memory() -> None:
    """Run Python GC and clear CUDA allocator cache when available."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def maybe_periodic_gpu_maintenance(
    step_index: int,
    cleanup_interval: int = 0,
    memory_log_interval: int = 0,
    device: torch.device | None = None,
    log_prefix: str = "GPU memory",
) -> None:
    """Run periodic memory logging/cleanup in long-running loops.

    Args:
        step_index: 1-based index of current loop step.
        cleanup_interval: If > 0, call ``clear_gpu_memory`` every N steps.
        memory_log_interval: If > 0, print memory stats every N steps.
        device: Optional target device for memory queries.
        log_prefix: Prefix used for memory log lines.
    """
    if step_index <= 0:
        return

    if memory_log_interval > 0 and step_index % memory_log_interval == 0:
        print_gpu_memory(prefix=f"{log_prefix} @ step {step_index}", device=device)

    if cleanup_interval > 0 and step_index % cleanup_interval == 0:
        clear_gpu_memory()
