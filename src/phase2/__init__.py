"""Phase 2 retrieval-augmented classification package."""

from .db_client import get_class_counts, get_image_collection, get_persistent_client, get_text_collection
from .embedders import ImageEmbedder, TextEmbedder
from .evaluation import compute_metrics, evaluate_variant, load_results, save_results
from .imbalance import simulate_imbalance
from .scoring import global_dnds, idw, local_dnds, majority_vote, traditional
from .visualization import (
    plot_alpha_sensitivity,
    plot_confusion_matrices,
    plot_continual_learning_curve,
    plot_minority_f1_vs_imbalance,
    plot_phase2_vs_phase1,
    plot_scoring_comparison,
)

__all__ = [
    "ImageEmbedder",
    "TextEmbedder",
    "get_persistent_client",
    "get_image_collection",
    "get_text_collection",
    "get_class_counts",
    "majority_vote",
    "idw",
    "global_dnds",
    "local_dnds",
    "traditional",
    "compute_metrics",
    "evaluate_variant",
    "save_results",
    "load_results",
    "simulate_imbalance",
    "plot_scoring_comparison",
    "plot_minority_f1_vs_imbalance",
    "plot_alpha_sensitivity",
    "plot_continual_learning_curve",
    "plot_confusion_matrices",
    "plot_phase2_vs_phase1",
]
