import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def save_registry(registry_path: Path, names: List[str], embeddings: np.ndarray) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(registry_path, names=np.array(names), embeddings=embeddings)


def save_registry_metadata(metadata_path: Path, payload: dict) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_registry(registry_path: Path) -> Dict[str, np.ndarray]:
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")

    data = np.load(registry_path, allow_pickle=False)
    names = data["names"].tolist()
    embeddings = data["embeddings"]

    if len(names) != len(embeddings):
        raise ValueError("Registry names and embeddings length mismatch")

    return {name: l2_normalize(embedding) for name, embedding in zip(names, embeddings)}


def find_best_match(
    embedding: np.ndarray,
    registry: Dict[str, np.ndarray],
    threshold: float,
) -> Tuple[str | None, float]:
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
