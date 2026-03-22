"""Phase 2 retrieval-augmented classification package."""

from importlib import import_module


_SYMBOL_TO_MODULE = {
    "ImageEmbedder": ".embedders",
    "TextEmbedder": ".embedders",
    "get_persistent_client": ".db_client",
    "get_image_collection": ".db_client",
    "get_text_collection": ".db_client",
    "get_class_counts": ".db_client",
    "majority_vote": ".scoring",
    "idw": ".scoring",
    "global_dnds": ".scoring",
    "local_dnds": ".scoring",
    "traditional": ".scoring",
    "compute_metrics": ".evaluation",
    "evaluate_variant": ".evaluation",
    "save_results": ".evaluation",
    "load_results": ".evaluation",
    "simulate_imbalance": ".imbalance",
    "plot_scoring_comparison": ".visualization",
    "plot_minority_f1_vs_imbalance": ".visualization",
    "plot_alpha_sensitivity": ".visualization",
    "plot_continual_learning_curve": ".visualization",
    "plot_confusion_matrices": ".visualization",
    "plot_phase2_vs_phase1": ".visualization",
}


def __getattr__(name: str):
    """Lazily import symbols to avoid hard dependency failures at package import time."""
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value

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
