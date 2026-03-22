"""Embedding wrappers used by Phase 2 RAC notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


class ImageEmbedder:
    """OpenCLIP image embedding wrapper with singleton model caching."""

    _model = None
    _preprocess = None

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
        """Initialize the OpenCLIP model lazily.

        Args:
            model_name: OpenCLIP vision backbone.
            pretrained: OpenCLIP pretrained weights identifier.
        """
        if ImageEmbedder._model is None or ImageEmbedder._preprocess is None:
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            model.eval()
            ImageEmbedder._model = model
            ImageEmbedder._preprocess = preprocess
        self.model_name = model_name
        self.pretrained = pretrained

    def embed(self, image_path: str) -> np.ndarray:
        """Embed a single image into a normalized vector.

        Args:
            image_path: Path to an RGB image file.

        Returns:
            A 1D numpy array containing image embedding features.
        """
        import torch

        image = Image.open(Path(image_path)).convert("RGB")
        image_tensor = ImageEmbedder._preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = ImageEmbedder._model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_batch(self, image_paths: list[str]) -> list[np.ndarray]:
        """Embed a batch of image paths.

        Args:
            image_paths: List of image paths.

        Returns:
            List of embedding vectors.
        """
        return [self.embed(path) for path in image_paths]


class TextEmbedder:
    """Sentence-transformer text embedding wrapper with singleton model caching."""

    _model: SentenceTransformer | None = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize sentence transformer lazily.

        Args:
            model_name: Sentence-transformer model identifier.
        """
        if TextEmbedder._model is None:
            TextEmbedder._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Input text.

        Returns:
            A 1D numpy float32 embedding vector.
        """
        vector = TextEmbedder._model.encode([text], normalize_embeddings=True)
        return vector[0].astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of text strings.

        Args:
            texts: List of strings.

        Returns:
            List of embedding vectors.
        """
        vectors = TextEmbedder._model.encode(texts, normalize_embeddings=True)
        return [vec.astype(np.float32) for vec in vectors]
