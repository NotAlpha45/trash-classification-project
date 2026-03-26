"""Imbalance experiment utilities."""

from __future__ import annotations

import random


def infer_class_groups(
    class_counts: dict[str, int],
    threshold: float = 1.15,
) -> tuple[list[str], list[str]]:
    """Infer majority and minority classes from class-count statistics.

    A class is marked as majority when its count is at least
    ``threshold * mean(other_class_counts)``. Remaining classes are minority.

    Args:
        class_counts: Mapping of class label to count.
        threshold: Relative threshold against the mean of remaining classes.

    Returns:
        Tuple of ``(majority_classes, minority_classes)``.
    """
    if not class_counts:
        return [], []

    ordered = sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))
    labels = [label for label, _ in ordered]

    majority_classes: list[str] = []
    for label, count in ordered:
        other_counts = [value for other_label, value in ordered if other_label != label]
        if not other_counts:
            continue

        other_mean = sum(other_counts) / len(other_counts)
        if count >= threshold * other_mean:
            majority_classes.append(label)

    # Keep the split usable for imbalance simulation.
    if not majority_classes:
        majority_classes = [labels[0]]
    if len(majority_classes) == len(labels):
        majority_classes = [labels[0]]

    minority_classes = [label for label in labels if label not in majority_classes]
    return majority_classes, minority_classes


def build_imbalanced_collections(
    client,
    source_image_collection,
    source_text_collection,
    majority_classes: list[str],
    minority_classes: list[str],
    ratio: int,
    image_collection_name: str,
    text_collection_name: str,
    seed: int = 42,
) -> tuple:
    """Build subsampled ChromaDB collections simulating minority class scarcity.

    Majority class records are kept fully intact. Minority class records are
    subsampled down to floor(majority_min_count / ratio) per class, where
    majority_min_count is the count of the smallest majority class. This mirrors
    real-world imbalance where minority items were never collected in sufficient
    quantities, rather than artificially capping majority retrieval.

    The same random seed is used for all calls to ensure reproducibility across
    experiment runs.

    Args:
        client: ChromaDB persistent client for the imbalanced DB.
        source_image_collection: Fully populated source image collection (read-only).
        source_text_collection: Fully populated source text collection (read-only).
        majority_classes: Class labels whose records are kept intact.
        minority_classes: Class labels whose records are subsampled down.
        ratio: Target majority:minority ratio. Minority target per class is
            floor(min(majority_class_counts) / ratio).
        image_collection_name: Name for the new imbalanced image collection.
        text_collection_name: Name for the new imbalanced text collection.
        seed: Random seed for reproducible subsampling. Default: 42.

    Returns:
        Tuple of (imbalanced_image_collection, imbalanced_text_collection).

    Raises:
        RuntimeError: If source collection fetch fails or returns empty results
            for any class.
        ValueError: If ratio <= 0 or no majority/minority classes are provided.
    """
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if not majority_classes:
        raise ValueError("majority_classes must not be empty")
    if not minority_classes:
        raise ValueError("minority_classes must not be empty")

    all_classes = majority_classes + minority_classes
    image_records: dict[str, dict] = {}
    text_records: dict[str, dict] = {}

    for class_name in all_classes:
        result = source_image_collection.get(
            where={"label": class_name},
            include=["embeddings", "metadatas"],
        )
        image_records[class_name] = result
        if not result["ids"]:
            raise RuntimeError(
                f"Source image collection returned 0 records for class '{class_name}'. "
                f"Check that the collection is fully populated."
            )

    for class_name in all_classes:
        result = source_text_collection.get(
            where={"label": class_name},
            include=["embeddings", "metadatas"],
        )
        text_records[class_name] = result
        if not result["ids"]:
            raise RuntimeError(
                f"Source text collection returned 0 records for class '{class_name}'. "
                f"Check that the collection is fully populated."
            )

    majority_counts = {c: len(image_records[c]["ids"]) for c in majority_classes}
    majority_min_count = min(majority_counts.values())
    minority_target = max(1, majority_min_count // ratio)

    print(f"Ratio {ratio}:1 - minority target per class: {minority_target}")
    print(f"  Majority counts kept intact: {majority_counts}")
    print(f"  Minority classes will be subsampled to: {minority_target}")

    rng = random.Random(seed)
    selected_ids: dict[str, list[str]] = {}

    for cls in majority_classes:
        selected_ids[cls] = list(image_records[cls]["ids"])

    for cls in minority_classes:
        all_ids = list(image_records[cls]["ids"])
        if len(all_ids) <= minority_target:
            selected_ids[cls] = all_ids
        else:
            selected_ids[cls] = rng.sample(all_ids, minority_target)

    selected_id_set = {id_ for ids in selected_ids.values() for id_ in ids}

    for name in [image_collection_name, text_collection_name]:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    imb_image_col = client.create_collection(
        name=image_collection_name,
        embedding_function=None,
        metadata={"ratio": ratio, "type": "image", "seed": seed},
    )
    imb_text_col = client.create_collection(
        name=text_collection_name,
        embedding_function=None,
        metadata={"ratio": ratio, "type": "text", "seed": seed},
    )

    batch_size = 200

    for cls in all_classes:
        source = image_records[cls]
        indices = [i for i, id_ in enumerate(source["ids"]) if id_ in selected_id_set]
        ids_to_add = [source["ids"][i] for i in indices]
        embeddings_to_add = [source["embeddings"][i] for i in indices]
        metadatas_to_add = [source["metadatas"][i] for i in indices]

        for batch_start in range(0, len(ids_to_add), batch_size):
            batch_end = batch_start + batch_size
            imb_image_col.add(
                ids=ids_to_add[batch_start:batch_end],
                embeddings=embeddings_to_add[batch_start:batch_end],
                metadatas=metadatas_to_add[batch_start:batch_end],
            )

    for cls in all_classes:
        source = text_records[cls]
        indices = [i for i, id_ in enumerate(source["ids"]) if id_ in selected_id_set]
        ids_to_add = [source["ids"][i] for i in indices]
        embeddings_to_add = [source["embeddings"][i] for i in indices]
        metadatas_to_add = [source["metadatas"][i] for i in indices]

        for batch_start in range(0, len(ids_to_add), batch_size):
            batch_end = batch_start + batch_size
            imb_text_col.add(
                ids=ids_to_add[batch_start:batch_end],
                embeddings=embeddings_to_add[batch_start:batch_end],
                metadatas=metadatas_to_add[batch_start:batch_end],
            )

    print(f"\nBuilt imbalanced collections at ratio {ratio}:1")
    actual_counts_image: dict[str, int] = {}
    for cls in all_classes:
        img_result = imb_image_col.get(where={"label": cls}, include=[])
        actual_counts_image[cls] = len(img_result["ids"])
        print(f"  Image DB - {cls}: {actual_counts_image[cls]} records")

    for maj_cls in majority_classes:
        for min_cls in minority_classes:
            maj_n = actual_counts_image[maj_cls]
            min_n = actual_counts_image[min_cls]
            if min_n > 0:
                actual_ratio = maj_n / min_n
                print(
                    f"  Actual ratio {maj_cls}:{min_cls} = {actual_ratio:.2f}:1 "
                    f"(target {ratio}:1)"
                )

    return imb_image_col, imb_text_col


def get_imbalanced_collection_names(ratio: int) -> tuple[str, str]:
    """Return ChromaDB collection names for a given imbalance ratio.

    Args:
        ratio: Majority:minority ratio.

    Returns:
        Tuple of (image_collection_name, text_collection_name).
    """
    return (
        f"trash_image_imbalanced_{ratio}to1",
        f"trash_text_imbalanced_{ratio}to1",
    )


def teardown_imbalanced_collections(client, ratio: int) -> None:
    """Delete imbalanced collections for a given ratio to free disk space.

    Safe to call even if the collections do not exist.

    Args:
        client: ChromaDB persistent client.
        ratio: Majority:minority ratio whose collections should be deleted.
    """
    image_name, text_name = get_imbalanced_collection_names(ratio)
    for name in [image_name, text_name]:
        try:
            client.delete_collection(name)
            print(f"  Deleted collection: {name}")
        except Exception:
            print(f"  Collection not found (skipped): {name}")
