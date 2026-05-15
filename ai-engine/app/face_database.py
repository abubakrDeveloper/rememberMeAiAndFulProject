from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from numpy.linalg import norm as np_norm
from PIL import Image

from .types import PersonRecord


class FaceDatabase:
    """Roster database using FaceNet-512 embeddings (cosine similarity matching)."""

    def __init__(self, tolerance: float = 0.5) -> None:
        self.tolerance = tolerance
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mtcnn = MTCNN(image_size=160, margin=40, keep_all=False, device=device)
        self._resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        self._device = device
        self._records: Dict[str, PersonRecord] = {}
        self._encodings: Dict[str, List[np.ndarray]] = {}
        # Short-term re-ID gallery: last 10 confirmed-frame embeddings per person
        self._recent_embeddings: Dict[str, deque] = {}

    @staticmethod
    def _parse_person(file_path: Path, role: str) -> PersonRecord:
        stem = file_path.stem
        # Format: "FULL NAME - 123456789" (name first, then numeric ID after " - ")
        if " - " in stem:
            name_part, id_part = stem.rsplit(" - ", 1)
            id_part = id_part.strip()
            if id_part.isdigit():
                return PersonRecord(
                    person_id=id_part,
                    name=name_part.strip().title(),
                    role=role,
                )
        parts = stem.split("_", 1)
        # Format: "ID_Name" where ID is a pure number (e.g. "001_John_Doe")
        if parts[0].isdigit() and len(parts) > 1:
            person_id = parts[0].strip()
            name = parts[1].replace("_", " ").strip()
        else:
            name = stem.replace("_", " ").strip()
            person_id = stem.strip()
        return PersonRecord(person_id=person_id, name=name, role=role)

    def _encode(self, face_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Return a L2-normalised 512-d FaceNet embedding, or None if MTCNN finds no face."""
        img = Image.fromarray(face_rgb)
        aligned = self._mtcnn(img)  # aligned 160×160 tensor, or None
        if aligned is None:
            return None
        with torch.no_grad():
            emb = self._resnet(aligned.unsqueeze(0).to(self._device)).cpu().numpy()[0]
        norm = float(np_norm(emb))
        return emb / (norm + 1e-6)

    def _load_role_dir(self, role_dir: Path, role: str) -> int:
        if not role_dir.exists():
            return 0
        loaded = 0
        for file_path in sorted(role_dir.glob("*")):
            if file_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            # cv2.imread fails on Windows with non-ASCII paths; use buffer workaround
            img_buf = np.fromfile(str(file_path), dtype=np.uint8)
            img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
            if img is None:
                print(f"  skip {file_path.name}: could not read image")
                continue
            face_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            enc = self._encode(face_rgb)
            if enc is None:
                print(f"  skip {file_path.name}: no face found")
                continue
            person = self._parse_person(file_path, role)
            self._records[person.person_id] = person
            self._encodings.setdefault(person.person_id, []).append(enc)
            loaded += 1
        return loaded

    def load(self, students_dir: str, teachers_dir: str, **_) -> Dict[str, int]:
        student_count = self._load_role_dir(Path(students_dir), "student")
        teacher_count = self._load_role_dir(Path(teachers_dir), "teacher")
        return {
            "students": student_count,
            "teachers": teacher_count,
            "people": len(self._records),
        }

    def load_people_dir(self, people_dir: str) -> Dict[str, int]:
        """Load faces from a single directory with a generic 'person' role."""
        count = self._load_role_dir(Path(people_dir), "person")
        return {"people": count}

    def has_people(self) -> bool:
        return len(self._records) > 0

    def list_people(self, role: Optional[str] = None) -> List[PersonRecord]:
        people = list(self._records.values())
        if role:
            people = [p for p in people if p.role == role]
        return sorted(people, key=lambda p: (p.role, p.person_id))

    def encode(self, face_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Public face encoding — returns L2-normalised 512-d embedding, or None."""
        return self._encode(face_rgb)

    def match_embedding(self, embedding: np.ndarray) -> Tuple[Optional[PersonRecord], float]:
        """Match a pre-computed embedding against the enrollment roster.

        Returns (PersonRecord, score) only when the winner beats the threshold
        (1 - tolerance = 0.80) AND clears the runner-up by at least MARGIN (0.15).
        """
        if not self._encodings:
            return None, 0.0

        # Per-person best: max cosine similarity over all that person's enrollment vectors
        per_person_best: Dict[str, float] = {
            pid: max(float(np.dot(embedding, vec)) for vec in vecs)
            for pid, vecs in self._encodings.items()
        }

        ranked = sorted(per_person_best.items(), key=lambda kv: kv[1], reverse=True)
        best_id, best_sim = ranked[0]
        second_sim = ranked[1][1] if len(ranked) > 1 else -1.0

        MARGIN = 0.15
        threshold = 1.0 - self.tolerance  # 0.80 with default tolerance=0.20
        if best_sim >= threshold and (best_sim - second_sim) >= MARGIN and best_id in self._records:
            return self._records[best_id], best_sim
        return None, best_sim

    def match(self, face_rgb: np.ndarray) -> Tuple[Optional[PersonRecord], float]:
        """Encode face_rgb then match against the enrollment roster."""
        embedding = self._encode(face_rgb)
        if embedding is None:
            return None, 0.0
        return self.match_embedding(embedding)

    def update_gallery(self, person_id: str, embedding: np.ndarray) -> None:
        """Add a confirmed-frame embedding to the short-term re-ID gallery.

        Each person keeps at most 10 recent embeddings (oldest auto-evicted).
        """
        if person_id not in self._recent_embeddings:
            self._recent_embeddings[person_id] = deque(maxlen=10)
        self._recent_embeddings[person_id].append(embedding)

    def reid_match(self, embedding: np.ndarray) -> Tuple[Optional[PersonRecord], float]:
        """Re-ID: match against the recent confirmed-frame gallery.

        Used when a new track fails enrollment matching — attempts to re-attach
        a face to a previously confirmed student who re-entered after occlusion.

        Uses the L2-normalised mean of each person's recent embeddings (higher
        quality than single passport photos) with a softer threshold (0.72) and
        a smaller margin (0.08) since gallery embeddings are already aligned.
        """
        if not self._recent_embeddings:
            return None, 0.0

        per_person_best: Dict[str, float] = {}
        for person_id, embs in self._recent_embeddings.items():
            if not embs:
                continue
            avg = np.mean(np.stack(list(embs)), axis=0)
            avg = avg / (float(np_norm(avg)) + 1e-6)
            per_person_best[person_id] = float(np.dot(embedding, avg))

        if not per_person_best:
            return None, 0.0

        ranked = sorted(per_person_best.items(), key=lambda kv: kv[1], reverse=True)
        best_id, best_sim = ranked[0]
        second_sim = ranked[1][1] if len(ranked) > 1 else -1.0

        REID_THRESHOLD = 0.72
        REID_MARGIN = 0.08
        if best_sim >= REID_THRESHOLD and (best_sim - second_sim) >= REID_MARGIN and best_id in self._records:
            return self._records[best_id], best_sim
        return None, best_sim