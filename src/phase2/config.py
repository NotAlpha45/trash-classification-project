"""Shared configuration for Phase 2 RAC notebooks and utilities."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


PHASE2_CONFIG: dict[str, Any] = {
    "k_vote": 10,
    "K_density": 50,
    "alpha": 0.5,
    "alpha_sweep": [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
    "imbalance_ratios": [10, 50, 100],
    "batch_size": 100,
    "epsilon": 1e-6,
    "kde_bandwidth": 0.5,
    "kde_bandwidth_sweep": [0.1, 0.25, 0.5, 1.0],
    "majority_threshold": 1.15,
    "db_path": "./chroma_db",
    "figures_path": "./figures/phase2",
    "class_names": ["Black", "Blue", "Green", "TTR"],
}


def get_phase2_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a mutable copy of the shared Phase 2 config.

    Args:
        overrides: Optional key-value overrides for notebook-specific settings.

    Returns:
        Deep-copied config dictionary.
    """
    config = deepcopy(PHASE2_CONFIG)
    if overrides:
        config.update(overrides)
    return config
