"""Imbalance simulation utilities for retrieval result post-filtering."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def simulate_imbalance(
    results: dict[str, Any],
    majority_classes: list[str],
    minority_classes: list[str],
    ratio: int,
) -> dict[str, Any]:
    """Post-filter Chroma results to emulate class imbalance.

    Args:
        results: Raw Chroma query result dictionary with ``metadatas`` and ``distances``.
        majority_classes: Labels treated as majority.
        minority_classes: Labels treated as minority.
        ratio: Desired majority:minority ratio.

    Returns:
        A filtered result dictionary preserving Chroma's query structure.
    """
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    ids = (results.get("ids") or [[]])[0] if "ids" in results else [None] * len(metadatas)

    k_density = len(metadatas)
    minority_denominator = max(1, len(minority_classes))
    majority_cap = int(ratio * (k_density / minority_denominator))

    kept_indices: list[int] = []
    kept_by_class: dict[str, int] = defaultdict(int)

    for idx, meta in enumerate(metadatas):
        label = (meta or {}).get("label")
        if label in majority_classes:
            if kept_by_class[label] < majority_cap:
                kept_indices.append(idx)
                kept_by_class[label] += 1
        else:
            kept_indices.append(idx)

    return {
        "metadatas": [[metadatas[i] for i in kept_indices]],
        "distances": [[distances[i] for i in kept_indices]],
        "ids": [[ids[i] for i in kept_indices]],
    }
