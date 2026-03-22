"""Visualization helpers for Phase 2 RAC experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_scoring_comparison(evaluation_results: dict[str, Any], save_path: str):
    """Plot macro F1 comparison across scoring variants."""
    _ensure_dir(str(Path(save_path).parent))
    variants = []
    macro_f1 = []
    for variant, metrics in evaluation_results.get("variants", {}).items():
        variants.append(variant)
        macro_f1.append(metrics.get("macro_f1", 0.0))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=variants, y=macro_f1, ax=ax, palette="viridis")
    ax.set_title("Macro F1 by Scoring Variant")
    ax.set_ylabel("Macro F1")
    ax.set_xlabel("Variant")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    return fig


def plot_minority_f1_vs_imbalance(imbalance_results: dict[str, Any], save_path: str):
    """Plot Green/TTR F1 vs imbalance ratio for each method."""
    _ensure_dir(str(Path(save_path).parent))
    ratios = imbalance_results.get("ratios", [])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for idx, minority_label in enumerate(["Green", "TTR"]):
        ax = axes[idx]
        for variant, per_ratio in imbalance_results.get("variants", {}).items():
            values = [per_ratio.get(str(r), {}).get("per_class_f1", {}).get(minority_label, 0.0) for r in ratios]
            ax.plot(ratios, values, marker="o", label=variant)
        ax.set_title(f"{minority_label} F1 vs Imbalance")
        ax.set_xlabel("Imbalance Ratio (majority:minority)")
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    return fig


def plot_alpha_sensitivity(alpha_results: dict[str, Any], save_path: str):
    """Plot macro F1 over alpha sweep for local DNDS."""
    _ensure_dir(str(Path(save_path).parent))
    alphas = alpha_results.get("alphas", [])
    values = alpha_results.get("macro_f1", [])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, values, marker="o", linewidth=2)
    ax.set_title("Alpha Sensitivity (local_dnds)")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    return fig


def plot_continual_learning_curve(continual_results: dict[str, Any], save_path: str):
    """Plot macro F1 as database size increases."""
    _ensure_dir(str(Path(save_path).parent))
    percents = continual_results.get("db_size_percent", [])
    macro_f1 = continual_results.get("macro_f1", [])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(percents, macro_f1, marker="o", linewidth=2)
    ax.set_title("Continual Learning Curve")
    ax.set_xlabel("DB Size (%)")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    return fig


def plot_confusion_matrices(
    evaluation_results: dict[str, Any],
    class_names: list[str],
    save_path: str,
    variants: list[str] | None = None,
):
    """Create a 2x2 confusion matrix grid for selected variants."""
    _ensure_dir(str(Path(save_path).parent))
    if variants is None:
        variants = ["majority_vote", "idw", "global_dnds", "local_dnds"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, variant in zip(axes.flatten(), variants):
        matrix = evaluation_results.get("variants", {}).get(variant, {}).get("confusion_matrix")
        if matrix is None:
            matrix = np.zeros((len(class_names), len(class_names)))
        sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(variant)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names, rotation=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    return fig


def plot_phase2_vs_phase1(best_phase2_macro_f1: float, phase1_macro_f1: float, save_path: str):
    """Plot side-by-side bar chart comparing best Phase 2 variant to Phase 1."""
    _ensure_dir(str(Path(save_path).parent))
    labels = ["Phase 1 Fusion", "Phase 2 Best RAC"]
    values = [phase1_macro_f1, best_phase2_macro_f1]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=labels, y=values, palette=["#6c757d", "#2a9d8f"], ax=ax)
    ax.set_title("Phase 2 vs Phase 1")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", va="bottom")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    return fig
