"""Evaluation and result persistence helpers for Phase 2 RAC notebooks."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm.auto import tqdm

from .gpu_utils import maybe_periodic_gpu_maintenance


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
        "per_class_precision": {
            label: float(score)
            for label, score in zip(
                class_names,
                np.atleast_1d(precision_score(y_true, y_pred, average=None, labels=class_names, zero_division=0)).tolist(),
                strict=False,
            )
        },
        "per_class_recall": {
            label: float(score)
            for label, score in zip(
                class_names,
                np.atleast_1d(recall_score(y_true, y_pred, average=None, labels=class_names, zero_division=0)).tolist(),
                strict=False,
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_names).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
    }
    metrics["report"] = metrics["classification_report"]
    return metrics


def evaluate_variant(
    score_fn: Callable[..., str],
    test_samples: list[dict[str, Any]],
    image_collection,
    text_collection,
    config: dict,
    alpha: float | None = None,
    score_kwargs: dict[str, Any] | None = None,
    cleanup_interval: int = 0,
    memory_log_interval: int = 0,
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

    model_device = None
    if score_kwargs and "image_model" in score_kwargs:
        try:
            model_device = next(score_kwargs["image_model"].parameters()).device
        except Exception:
            model_device = None

    for sample_index, sample in enumerate(
        tqdm(test_samples, desc=f"Evaluating {score_fn.__name__}"),
        start=1,
    ):
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

        maybe_periodic_gpu_maintenance(
            step_index=sample_index,
            cleanup_interval=cleanup_interval,
            memory_log_interval=memory_log_interval,
            device=model_device,
            log_prefix=score_fn.__name__,
        )

    metrics = compute_metrics(y_true, y_pred, config["class_names"])
    mean_latency_ms = float(np.mean(latencies_ms) if latencies_ms else 0.0)
    metrics["inference_time_ms"] = mean_latency_ms
    metrics["latency_ms_per_sample"] = mean_latency_ms
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


def save_metrics_summary_csv(variant_results: dict[str, dict[str, Any]], path: str) -> None:
    """Save per-variant metric summary to CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    class_names: list[str] = []
    for metrics in variant_results.values():
        keys = list((metrics.get("per_class_f1") or {}).keys())
        if keys:
            class_names = keys
            break

    headers = [
        "variant",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "inference_time_ms",
    ]
    headers.extend([f"f1_{name}" for name in class_names])
    headers.extend([f"precision_{name}" for name in class_names])
    headers.extend([f"recall_{name}" for name in class_names])

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()

        for variant_name, metrics in variant_results.items():
            row = {
                "variant": variant_name,
                "accuracy": metrics.get("accuracy", 0.0),
                "macro_f1": metrics.get("macro_f1", 0.0),
                "weighted_f1": metrics.get("weighted_f1", 0.0),
                "inference_time_ms": metrics.get("inference_time_ms", metrics.get("latency_ms_per_sample", 0.0)),
            }

            per_f1 = metrics.get("per_class_f1", {})
            per_p = metrics.get("per_class_precision", {})
            per_r = metrics.get("per_class_recall", {})
            for class_name in class_names:
                row[f"f1_{class_name}"] = per_f1.get(class_name, 0.0)
                row[f"precision_{class_name}"] = per_p.get(class_name, 0.0)
                row[f"recall_{class_name}"] = per_r.get(class_name, 0.0)

            writer.writerow(row)


def save_alpha_sweep_csv(alpha_sweep: dict[str, Any], path: str) -> None:
    """Save alpha sweep values to CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    alphas = alpha_sweep.get("alphas", [])
    macro_f1 = alpha_sweep.get("macro_f1", [])

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alpha", "macro_f1"])
        for alpha, score in zip(alphas, macro_f1):
            writer.writerow([alpha, score])


def save_predictions_csv(variant_results: dict[str, dict[str, Any]], path: str) -> None:
    """Save per-sample predictions for each variant to CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "sample_index", "y_true", "y_pred"])

        for variant_name, metrics in variant_results.items():
            y_true = metrics.get("y_true", [])
            y_pred = metrics.get("y_pred", [])
            for index, (truth, pred) in enumerate(zip(y_true, y_pred)):
                writer.writerow([variant_name, index, truth, pred])


def save_imbalance_summary_csv(imbalance_results: dict[str, Any], path: str) -> None:
    """Save imbalance experiment summary metrics to CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ratio",
                "variant",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "green_f1",
                "ttr_f1",
            ]
        )

        ratios = imbalance_results.get("ratios", [])
        variants = imbalance_results.get("variants", {})
        for ratio in ratios:
            ratio_key = str(ratio)
            for variant_name, by_ratio in variants.items():
                metrics = by_ratio.get(ratio_key, {})
                per_f1 = metrics.get("per_class_f1", {})
                writer.writerow(
                    [
                        ratio,
                        variant_name,
                        metrics.get("accuracy", 0.0),
                        metrics.get("macro_f1", 0.0),
                        metrics.get("weighted_f1", 0.0),
                        per_f1.get("Green", 0.0),
                        per_f1.get("TTR", 0.0),
                    ]
                )


def save_continual_summary_csv(continual_results: dict[str, Any], path: str) -> None:
    """Save continual-learning curve values to CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "db_size_percent",
                "db_count",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "inference_time_ms",
            ]
        )

        steps = continual_results.get("steps", {})
        for pct in continual_results.get("db_size_percent", []):
            step = steps.get(str(pct), {})
            writer.writerow(
                [
                    pct,
                    step.get("db_count", 0),
                    step.get("accuracy", 0.0),
                    step.get("macro_f1", 0.0),
                    step.get("weighted_f1", 0.0),
                    step.get("inference_time_ms", 0.0),
                ]
            )
