"""Model loading and validation utilities for the KNN flood-risk classifier."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATHS: tuple[Path, ...] = (
    ROOT_DIR / "models" / "knn_model.pkl",
    ROOT_DIR / "knn_model.pkl",
)


def resolve_model_path() -> Path | None:
    """Return the first existing model artifact path."""
    for candidate in DEFAULT_MODEL_PATHS:
        if candidate.exists():
            return candidate
    return None


def load_knn_bundle(target_neighbors: int = 17) -> dict[str, Any]:
    """Load the serialized KNN classifier and test partition, enforcing k=target_neighbors."""
    model_path = resolve_model_path()
    if model_path is None:
        raise FileNotFoundError(
            "Model artifact 'models/knn_model.pkl' not found. Run the training export pipeline first."
        )

    with model_path.open("rb") as f:
        bundle: dict[str, Any] = pickle.load(f)

    model = bundle.get("model")
    if model is not None and hasattr(model, "n_neighbors"):
        model.n_neighbors = target_neighbors
        bundle["n_neighbors"] = target_neighbors

    return bundle
