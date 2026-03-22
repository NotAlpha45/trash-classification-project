"""Phase 1-compatible model loading helpers for traditional baseline comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


def _load_state_dict(payload: dict[str, Any] | Any) -> dict[str, Any]:
    """Extract model state dict from either full checkpoint payload or raw state dict."""
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict):
        return payload
    raise TypeError("Unsupported checkpoint payload format for traditional model loading.")


def load_phase1_traditional_components(
    image_checkpoint_path: str | Path,
    text_checkpoint_path: str | Path,
    num_classes: int,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load Phase 1 image/text models, tokenizer, and image transform.

    The image architecture mirrors the Phase 1 `multimodal_fusion.ipynb` setup:
    MobileNetV3-Large with a custom 3-layer classifier head.

    Args:
        image_checkpoint_path: Path to MobileNetV3 checkpoint.
        text_checkpoint_path: Path to DistilBERT checkpoint.
        num_classes: Number of output classes.
        device: Target device for loaded models.

    Returns:
        Dictionary containing image_model, text_model, tokenizer, and transform.
    """
    image_checkpoint_path = Path(image_checkpoint_path)
    text_checkpoint_path = Path(text_checkpoint_path)

    image_model = models.mobilenet_v3_large(weights=None)
    first_layer = image_model.classifier[0]
    in_features = first_layer.in_features if isinstance(first_layer, nn.Linear) else 960

    image_model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )

    image_payload = torch.load(image_checkpoint_path, map_location=device)
    image_model.load_state_dict(_load_state_dict(image_payload))
    image_model = image_model.to(device)
    image_model.eval()

    text_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_classes,
    )
    text_payload = torch.load(text_checkpoint_path, map_location=device)
    text_model.load_state_dict(_load_state_dict(text_payload))
    text_model = text_model.to(device)
    text_model.eval()

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return {
        "image_model": image_model,
        "text_model": text_model,
        "tokenizer": tokenizer,
        "transform": image_transform,
    }
