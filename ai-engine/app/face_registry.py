import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


Registry = Dict[str, np.ndarray]


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def save_registry(registry_path: Path, names: List[str], embeddings: np.ndarray) -> None:
    if len(names) == 0:
        raise ValueError("At least one enrolled name is required")

    if embeddings.ndim != 2:
        raise ValueError("Embeddings array must be 2D")

    if len(names) != int(embeddings.shape[0]):
        raise ValueError("Names and embeddings row count mismatch")

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(registry_path, names=np.array(names), embeddings=embeddings)


def save_registry_metadata(metadata_path: Path, payload: dict) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_registry(registry_path: Path) -> Registry:
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")

    data = np.load(registry_path, allow_pickle=False)
    names = data["names"].tolist()
    embeddings = np.asarray(data["embeddings"])

    if len(names) != len(embeddings):
        raise ValueError("Registry names and embeddings length mismatch")

    if embeddings.ndim != 2:
        raise ValueError("Registry embeddings must be a 2D array")

    return {name: l2_normalize(embedding) for name, embedding in zip(names, embeddings)}


def find_best_match(
    embedding: np.ndarray,
    registry: Registry,
    threshold: float,
) -> Tuple[str | None, float]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0")

    if not registry:
        return None, -1.0

    best_name = None
    best_score = -1.0

    for name, enrolled_embedding in registry.items():
        score = cosine_similarity(embedding, enrolled_embedding)
        if score > best_score:
            best_score = score
            best_name = name

    if best_score < threshold:
        return None, best_score

    return best_name, best_score


def list_images(folder: Path) -> Iterable[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path
