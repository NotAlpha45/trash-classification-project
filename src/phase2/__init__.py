"""Phase 2 retrieval-augmented classification package."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PHASE2_CONFIG, get_phase2_config
    from .data_utils import build_records_from_csv, resolve_image_path
    from .db_client import get_class_counts, get_image_collection, get_persistent_client, get_text_collection
    from .embedders import ImageEmbedder, TextEmbedder
    from .gpu_utils import clear_gpu_memory, get_device, print_device_info
    from .evaluation import compute_metrics, evaluate_variant, load_results, save_results
    from .evaluation import (
        save_alpha_sweep_csv,
        save_continual_summary_csv,
        save_imbalance_summary_csv,
        save_metrics_summary_csv,
        save_predictions_csv,
    )
    from .imbalance import infer_class_groups, simulate_imbalance
    from .scoring import global_dnds, idw, local_dnds, majority_vote, traditional
    from .traditional import load_phase1_traditional_components
    from .visualization import (
        plot_alpha_sensitivity,
        plot_confusion_matrices,
        plot_continual_learning_curve,
        plot_minority_f1_vs_imbalance,
        plot_phase2_vs_phase1,
        plot_scoring_comparison,
    )


_SYMBOL_TO_MODULE = {
    "PHASE2_CONFIG": ".config",
    "get_phase2_config": ".config",
    "resolve_image_path": ".data_utils",
    "build_records_from_csv": ".data_utils",
    "ImageEmbedder": ".embedders",
    "TextEmbedder": ".embedders",
    "get_device": ".gpu_utils",
    "print_device_info": ".gpu_utils",
    "clear_gpu_memory": ".gpu_utils",
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
    "save_metrics_summary_csv": ".evaluation",
    "save_alpha_sweep_csv": ".evaluation",
    "save_predictions_csv": ".evaluation",
    "save_imbalance_summary_csv": ".evaluation",
    "save_continual_summary_csv": ".evaluation",
    "infer_class_groups": ".imbalance",
    "simulate_imbalance": ".imbalance",
    "load_phase1_traditional_components": ".traditional",
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
    "PHASE2_CONFIG",
    "get_phase2_config",
    "resolve_image_path",
    "build_records_from_csv",
    "ImageEmbedder",
    "TextEmbedder",
    "get_device",
    "print_device_info",
    "clear_gpu_memory",
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
    "save_metrics_summary_csv",
    "save_alpha_sweep_csv",
    "save_predictions_csv",
    "save_imbalance_summary_csv",
    "save_continual_summary_csv",
    "infer_class_groups",
    "simulate_imbalance",
    "load_phase1_traditional_components",
    "plot_scoring_comparison",
    "plot_minority_f1_vs_imbalance",
    "plot_alpha_sensitivity",
    "plot_continual_learning_curve",
    "plot_confusion_matrices",
    "plot_phase2_vs_phase1",
]
