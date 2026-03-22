"""Scoring variants for retrieval-augmented classification (RAC)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np

from .db_client import get_class_counts


def _safe_query(
    image_collection,
    text_collection,
    query_image: np.ndarray,
    query_text: str,
    k_density: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Query both Chroma collections and return raw result payloads."""
    try:
        image_results = image_collection.query(
            query_images=[query_image],
            n_results=k_density,
            include=["metadatas", "distances"],
        )
        text_results = text_collection.query(
            query_texts=[query_text],
            n_results=k_density,
            include=["metadatas", "distances"],
        )
        return image_results, text_results
    except Exception as exc:
        raise RuntimeError(f"Failed to query Chroma collections: {exc}") from exc


def _accumulate_idw(
    results: dict[str, Any],
    class_names: list[str],
    k_vote: int,
    epsilon: float,
) -> dict[str, float]:
    """Aggregate inverse-distance scores over top-k neighbors."""
    scores = {label: 0.0 for label in class_names}
    metadatas = (results.get("metadatas") or [[]])[0][:k_vote]
    distances = (results.get("distances") or [[]])[0][:k_vote]

    for meta, distance in zip(metadatas, distances):
        label = (meta or {}).get("label")
        if label in scores:
            scores[label] += 1.0 / (float(distance) + epsilon)
    return scores


def _local_dnds_modality(
    results: dict[str, Any],
    class_names: list[str],
    k_vote: int,
    k_density: int,
    epsilon: float,
) -> dict[str, float]:
    """Compute local DNDS score for a single modality."""
    scores = {label: 0.0 for label in class_names}
    metadatas = (results.get("metadatas") or [[]])[0][:k_density]
    distances = (results.get("distances") or [[]])[0][:k_density]

    density_counts = Counter((meta or {}).get("label") for meta in metadatas)
    vote_metadatas = metadatas[:k_vote]
    vote_distances = distances[:k_vote]

    raw_by_class: dict[str, float] = defaultdict(float)
    for meta, dist in zip(vote_metadatas, vote_distances):
        label = (meta or {}).get("label")
        if label in scores:
            raw_by_class[label] += 1.0 / (float(dist) + epsilon)

    for label in class_names:
        rho_c = density_counts.get(label, 0) / float(max(1, k_density))
        if rho_c > 0:
            scores[label] = raw_by_class.get(label, 0.0) / rho_c
    return scores


def _global_dnds_modality(
    results: dict[str, Any],
    class_names: list[str],
    k_vote: int,
    epsilon: float,
    global_density: dict[str, float],
) -> dict[str, float]:
    """Compute global density-normalized scores for a modality."""
    raw = _accumulate_idw(results, class_names, k_vote, epsilon)
    out = {label: 0.0 for label in class_names}
    for label in class_names:
        rho = global_density.get(label, 0.0)
        if rho > 0:
            out[label] = raw[label] / rho
    return out


def _fuse_scores(
    image_scores: dict[str, float],
    text_scores: dict[str, float],
    class_names: list[str],
    alpha: float,
) -> str:
    """Fuse modality scores and return argmax label."""
    fused = {
        label: alpha * image_scores.get(label, 0.0)
        + (1.0 - alpha) * text_scores.get(label, 0.0)
        for label in class_names
    }
    return max(fused, key=fused.get)


def majority_vote(
    query_image: np.ndarray,
    query_text: str,
    image_collection,
    text_collection,
    config: dict,
    alpha: float = 0.5,
    **kwargs,
) -> str:
    """Predict class by majority voting over image and text top-k neighbors."""
    k_vote = int(config["k_vote"])
    k_density = int(config["K_density"])
    class_names = list(config["class_names"])

    image_results = kwargs.get("raw_image_results")
    text_results = kwargs.get("raw_text_results")
    if image_results is None or text_results is None:
        image_results, text_results = _safe_query(
            image_collection, text_collection, query_image, query_text, k_density
        )

    image_votes = Counter(
        (m or {}).get("label")
        for m in (image_results.get("metadatas") or [[]])[0][:k_vote]
    )
    text_votes = Counter(
        (m or {}).get("label")
        for m in (text_results.get("metadatas") or [[]])[0][:k_vote]
    )

    score_table = {
        label: alpha * image_votes.get(label, 0)
        + (1.0 - alpha) * text_votes.get(label, 0)
        for label in class_names
    }
    return max(score_table, key=score_table.get)


def idw(
    query_image: np.ndarray,
    query_text: str,
    image_collection,
    text_collection,
    config: dict,
    alpha: float = 0.5,
    **kwargs,
) -> str:
    """Predict class using inverse distance weighting without imbalance correction."""
    k_vote = int(config["k_vote"])
    k_density = int(config["K_density"])
    epsilon = float(config["epsilon"])
    class_names = list(config["class_names"])

    image_results = kwargs.get("raw_image_results")
    text_results = kwargs.get("raw_text_results")
    if image_results is None or text_results is None:
        image_results, text_results = _safe_query(
            image_collection, text_collection, query_image, query_text, k_density
        )

    image_scores = _accumulate_idw(image_results, class_names, k_vote, epsilon)
    text_scores = _accumulate_idw(text_results, class_names, k_vote, epsilon)
    return _fuse_scores(image_scores, text_scores, class_names, alpha)


def global_dnds(
    query_image: np.ndarray,
    query_text: str,
    image_collection,
    text_collection,
    config: dict,
    alpha: float = 0.5,
    **kwargs,
) -> str:
    """Predict class using global database priors as density correction."""
    k_vote = int(config["k_vote"])
    k_density = int(config["K_density"])
    epsilon = float(config["epsilon"])
    class_names = list(config["class_names"])

    image_results = kwargs.get("raw_image_results")
    text_results = kwargs.get("raw_text_results")
    if image_results is None or text_results is None:
        image_results, text_results = _safe_query(
            image_collection, text_collection, query_image, query_text, k_density
        )

    image_counts = kwargs.get("image_class_counts") or get_class_counts(
        image_collection
    )
    text_counts = kwargs.get("text_class_counts") or get_class_counts(text_collection)

    image_total = max(1, sum(image_counts.values()))
    text_total = max(1, sum(text_counts.values()))

    image_density = {
        label: image_counts.get(label, 0) / image_total for label in class_names
    }
    text_density = {
        label: text_counts.get(label, 0) / text_total for label in class_names
    }

    image_scores = _global_dnds_modality(
        image_results, class_names, k_vote, epsilon, image_density
    )
    text_scores = _global_dnds_modality(
        text_results, class_names, k_vote, epsilon, text_density
    )
    return _fuse_scores(image_scores, text_scores, class_names, alpha)


def local_dnds(
    query_image: np.ndarray,
    query_text: str,
    image_collection,
    text_collection,
    config: dict,
    alpha: float = 0.5,
    **kwargs,
) -> str:
    """Predict class using local density-normalized distance score (DNDS)."""
    k_vote = int(config["k_vote"])
    k_density = int(config["K_density"])
    epsilon = float(config["epsilon"])
    class_names = list(config["class_names"])

    image_results = kwargs.get("raw_image_results")
    text_results = kwargs.get("raw_text_results")
    if image_results is None or text_results is None:
        image_results, text_results = _safe_query(
            image_collection, text_collection, query_image, query_text, k_density
        )

    image_scores = _local_dnds_modality(
        image_results, class_names, k_vote, k_density, epsilon
    )
    text_scores = _local_dnds_modality(
        text_results, class_names, k_vote, k_density, epsilon
    )
    return _fuse_scores(image_scores, text_scores, class_names, alpha)


def traditional(
    query_image: np.ndarray,
    query_text: str,
    image_collection,
    text_collection,
    config: dict,
    alpha: float = 0.5,
    **kwargs,
) -> str:
    """Predict class using Phase 1 model logits and weighted late fusion.

    Expected keyword arguments include ``image_model``, ``text_model``, ``tokenizer``, and
    ``transform`` to mirror the original Phase 1 inference setup.
    """
    class_names = list(config["class_names"])
    image_model = kwargs.get("image_model")
    text_model = kwargs.get("text_model")
    tokenizer = kwargs.get("tokenizer")
    transform: Callable | None = kwargs.get("transform")
    use_half_precision = bool(kwargs.get("use_half_precision", False))

    if any(item is None for item in (image_model, text_model, tokenizer, transform)):
        raise ValueError(
            "traditional() requires image_model, text_model, tokenizer, and transform keyword arguments."
        )

    import torch
    from PIL import Image

    image_pil = Image.fromarray(query_image.astype(np.uint8)).convert("RGB")
    model_device = next(image_model.parameters()).device
    image_tensor = transform(image_pil).unsqueeze(0).to(model_device)

    tokenized = tokenizer(
        query_text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
    )
    tokenized = {key: value.to(model_device) for key, value in tokenized.items()}

    use_autocast = use_half_precision and model_device.type == "cuda"

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_autocast):
            image_out = image_model(image_tensor)
            text_out = text_model(**tokenized)

        image_logits = image_out.logits if hasattr(image_out, "logits") else image_out
        text_logits = text_out.logits if hasattr(text_out, "logits") else text_out

        fused = alpha * text_logits + (1.0 - alpha) * image_logits
        pred_idx = int(torch.argmax(fused, dim=1).item())

    if pred_idx < 0 or pred_idx >= len(class_names):
        raise IndexError(f"Predicted index {pred_idx} is outside class_names bounds.")
    return class_names[pred_idx]
