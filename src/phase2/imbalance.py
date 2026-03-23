"""Imbalance simulation utilities for retrieval result post-filtering."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def infer_class_groups(
    class_counts: dict[str, int],
    threshold: float = 1.15,
) -> tuple[list[str], list[str]]:
    """Infer majority and minority classes from class-count statistics.

    A class is marked as majority when its count is at least
    ``threshold * mean(other_class_counts)``. Remaining classes are minority.

    Args:
        class_counts: Mapping of class label to count.
        threshold: Relative threshold against the mean of remaining classes.

    Returns:
        Tuple of ``(majority_classes, minority_classes)``.
    """
    if not class_counts:
        return [], []

    ordered = sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))
    labels = [label for label, _ in ordered]

    majority_classes: list[str] = []
    for label, count in ordered:
        other_counts = [value for other_label, value in ordered if other_label != label]
        if not other_counts:
            continue

        other_mean = sum(other_counts) / len(other_counts)
        if count >= threshold * other_mean:
            majority_classes.append(label)

    # Keep the split usable for imbalance simulation.
    if not majority_classes:
        majority_classes = [labels[0]]
    if len(majority_classes) == len(labels):
        majority_classes = [labels[0]]

    minority_classes = [label for label in labels if label not in majority_classes]
    return majority_classes, minority_classes


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
    ids = (
        (results.get("ids") or [[]])[0] if "ids" in results else [None] * len(metadatas)
    )

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
