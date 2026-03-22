"""ChromaDB client and collection helpers for Phase 2 RAC experiments."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import (
    OpenCLIPEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)


def get_persistent_client(db_path: str) -> chromadb.PersistentClient:
    """Create a persistent ChromaDB client.

    Args:
        db_path: Filesystem path where ChromaDB persists collections.

    Returns:
        A configured ``chromadb.PersistentClient`` instance.
    """
    return chromadb.PersistentClient(path=str(Path(db_path).resolve()))


def _get_image_embedding_function() -> OpenCLIPEmbeddingFunction:
    """Create the OpenCLIP embedding function for image retrieval."""
    return OpenCLIPEmbeddingFunction()


def _get_text_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """Create the sentence-transformer embedding function for text retrieval."""
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def get_image_collection(client: chromadb.PersistentClient, name: str = "trash_image_db") -> Collection:
    """Get or create the image embedding collection.

    Args:
        client: Persistent ChromaDB client.
        name: Collection name.

    Returns:
        A Chroma collection storing image embeddings and metadata.
    """
    return client.get_or_create_collection(
        name=name,
        embedding_function=_get_image_embedding_function(),
        metadata={"description": "Image embeddings for RAC experiment"},
    )


def get_text_collection(client: chromadb.PersistentClient, name: str = "trash_text_db") -> Collection:
    """Get or create the text embedding collection.

    Args:
        client: Persistent ChromaDB client.
        name: Collection name.

    Returns:
        A Chroma collection storing text embeddings and metadata.
    """
    return client.get_or_create_collection(
        name=name,
        embedding_function=_get_text_embedding_function(),
        metadata={"description": "Text (filename) embeddings for RAC experiment"},
    )


def get_class_counts(collection: Collection) -> dict[str, int]:
    """Compute class counts from collection metadata.

    Args:
        collection: Chroma collection containing ``label`` metadata.

    Returns:
        Dictionary mapping class label to count.
    """
    payload: dict[str, Any] = collection.get(include=["metadatas"])
    metadatas = payload.get("metadatas") or []
    labels = [item.get("label") for item in metadatas if isinstance(item, dict) and item.get("label")]
    return dict(Counter(labels))
