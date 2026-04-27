from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import face_recognition
import numpy as np

from .types import PersonRecord


class FaceDatabase:
    """Roster database using dlib 128-d face encodings via face_recognition."""

    def __init__(self, tolerance: float = 0.5) -> None:
        self.tolerance = tolerance
        self._records: Dict[str, PersonRecord] = {}
        self._encodings: Dict[str, List[np.ndarray]] = {}

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

    def _load_role_dir(self, role_dir: Path, role: str) -> int:
        if not role_dir.exists():
            return 0
        loaded = 0
        for file_path in sorted(role_dir.glob("*")):
            if file_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            image = face_recognition.load_image_file(str(file_path))
            encs = face_recognition.face_encodings(image)
            if not encs:
                print(f"  skip {file_path.name}: no face found")
                continue
            person = self._parse_person(file_path, role)
            self._records[person.person_id] = person
            self._encodings.setdefault(person.person_id, []).append(encs[0])
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

    @staticmethod
    def _ensure_min_size(img: np.ndarray, min_dim: int = 150) -> np.ndarray:
        h, w = img.shape[:2]
        if h >= min_dim and w >= min_dim:
            return img
        scale = max(min_dim / h, min_dim / w)
        new_w, new_h = int(w * scale), int(h * scale)
        import cv2
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def match(self, face_rgb: np.ndarray) -> Tuple[Optional[PersonRecord], float]:
        """Match an RGB face crop (with padding) against the roster."""
        if not self._encodings:
            return None, 0.0
        face_rgb = self._ensure_min_size(face_rgb, min_dim=80)
        face_rgb = np.ascontiguousarray(face_rgb, dtype=np.uint8)
        h, w = face_rgb.shape[:2]
        # Use the entire crop as the face region — bypasses dlib HOG which fails
        # on top-down CCTV angles. The bounding box from MediaPipe is already
        # tight around the face.
        locations = [(0, w, h, 0)]
        encs = face_recognition.face_encodings(face_rgb, known_face_locations=locations)
        if not encs:
            return None, 0.0
        query = encs[0]

        best_id = ""
        best_dist = 999.0
        for person_id, vectors in self._encodings.items():
            dists = face_recognition.face_distance(vectors, query)
            min_dist = float(np.min(dists))
            if min_dist < best_dist:
                best_dist = min_dist
                best_id = person_id

        confidence = max(0.0, 1.0 - best_dist)
        if best_dist <= self.tolerance and best_id in self._records:
            return self._records[best_id], confidence
        return None, confidence