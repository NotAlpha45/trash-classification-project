"""Data loading helpers for Phase 2 experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_image_path(split_dir: Path, label: str, raw_name: str) -> Path:
    """Resolve an image path with optional extension fallback.

    Args:
        split_dir: Dataset split root, e.g. ``dataset/CVPR_2024_dataset_Test``.
        label: Class folder name.
        raw_name: Filename from CSV, with or without extension.

    Returns:
        Existing image path.

    Raises:
        FileNotFoundError: If no matching file exists.
    """
    base_path = split_dir / label / str(raw_name)
    if base_path.suffix and base_path.exists():
        return base_path

    if not base_path.suffix:
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = base_path.with_suffix(ext)
            if candidate.exists():
                return candidate
        if base_path.exists():
            return base_path

    raise FileNotFoundError(f"Missing image for label={label}, name={raw_name}")


def build_records_from_csv(
    csv_path: Path,
    split_dir: Path,
    text_column: str = "text",
    label_column: str = "label",
    text_key: str = "text",
    max_missing_examples: int = 10,
) -> tuple[list[dict[str, Any]], list[str], int]:
    """Build sample records from a CSV using extension-aware image resolution.

    Args:
        csv_path: CSV file containing filename/text and class label columns.
        split_dir: Dataset split root directory.
        text_column: CSV column containing the filename/text token.
        label_column: CSV column containing class labels.
        text_key: Output key used for the text field in returned records.
        max_missing_examples: Maximum number of missing examples to collect.

    Returns:
        Tuple containing:
        - List of records with keys ``image_path``, ``label``, and ``text_key``.
        - List of missing ``label/text`` examples (truncated).
        - Total number of rows read from the CSV.

    Raises:
        ValueError: If required CSV columns are missing.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if not {text_column, label_column}.issubset(df.columns):
        raise ValueError(f"{csv_path.name} must contain columns: {text_column}, {label_column}")

    records: list[dict[str, Any]] = []
    missing_examples: list[str] = []

    for row in df.itertuples(index=False):
        label = str(getattr(row, label_column))
        text_value = str(getattr(row, text_column))

        try:
            image_path = resolve_image_path(split_dir, label, text_value)
        except FileNotFoundError:
            if len(missing_examples) < max_missing_examples:
                missing_examples.append(f"{label}/{text_value}")
            continue

        records.append(
            {
                "image_path": str(image_path),
                "label": label,
                text_key: text_value,
            }
        )

    return records, missing_examples, len(df)
