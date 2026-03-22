"""Evaluation and result persistence helpers for Phase 2 RAC notebooks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm


def load_image_as_numpy(image_path: str) -> np.ndarray:
    """Load an image file as an RGB ``np.uint8`` array.

    Args:
        image_path: Path to image.

    Returns:
        Image array of shape ``(H, W, 3)``.
    """
    image = Image.open(image_path).convert("RGB")
    return np.array(image, dtype=np.uint8)


def compute_metrics(y_true: list[str], y_pred: list[str], class_names: list[str]) -> dict[str, Any]:
    """Compute accuracy, F1, report, and confusion matrix.

    Args:
        y_true: Ground truth class labels.
        y_pred: Predicted class labels.
        class_names: Ordered class labels.

    Returns:
        Metrics dictionary suitable for JSON serialization.
    """
    per_class_f1_array = np.atleast_1d(f1_score(y_true, y_pred, average=None, labels=class_names))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class_f1": {
            label: float(score)
            for label, score in zip(class_names, per_class_f1_array.tolist(), strict=False)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_names).tolist(),
        "report": classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
    }
    return metrics


def evaluate_variant(
    score_fn: Callable[..., str],
    test_samples: list[dict[str, Any]],
    image_collection,
    text_collection,
    config: dict,
    alpha: float | None = None,
    score_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a scoring function on a full test dataset.

    Args:
        score_fn: Callable scoring variant that predicts a class label.
        test_samples: List of sample records with keys: ``image_path``, ``text``, ``label``.
        image_collection: Chroma image collection.
        text_collection: Chroma text collection.
        config: Experiment configuration dictionary.
        alpha: Optional fusion weight override.
        score_kwargs: Additional keyword args passed to ``score_fn``.

    Returns:
        Dictionary with metrics, latency, and predictions.
    """
    if score_kwargs is None:
        score_kwargs = {}

    y_true: list[str] = []
    y_pred: list[str] = []
    latencies_ms: list[float] = []

    active_alpha = config["alpha"] if alpha is None else alpha

    for sample in tqdm(test_samples, desc=f"Evaluating {score_fn.__name__}"):
        image_path = sample["image_path"]
        text = sample["text"]
        label = sample["label"]

        query_image = load_image_as_numpy(image_path)
        start = time.perf_counter()
        pred = score_fn(
            query_image=query_image,
            query_text=text,
            image_collection=image_collection,
            text_collection=text_collection,
            config=config,
            alpha=active_alpha,
            **score_kwargs,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        y_true.append(label)
        y_pred.append(pred)
        latencies_ms.append(elapsed_ms)

    metrics = compute_metrics(y_true, y_pred, config["class_names"])
    metrics["latency_ms_per_sample"] = float(np.mean(latencies_ms) if latencies_ms else 0.0)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["alpha"] = float(active_alpha)
    metrics["variant"] = score_fn.__name__
    return metrics


def save_results(results: dict[str, Any], path: str) -> None:
    """Save experiment outputs to JSON.

    Args:
        results: Serializable result dictionary.
        path: Output JSON file path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def load_results(path: str) -> dict[str, Any]:
    """Load experiment outputs from a JSON file.

    Args:
        path: Input JSON file path.

    Returns:
        Parsed dictionary.
    """
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)
