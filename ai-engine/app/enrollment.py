import argparse
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.config import DemoConfig
from app.face_registry import l2_normalize, list_images, save_registry, save_registry_metadata


def get_largest_face_embedding(face_analyzer: FaceAnalysis, image: np.ndarray) -> np.ndarray | None:
    faces = face_analyzer.get(image)
    if not faces:
        return None

    largest_face = max(
        faces,
        key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
    )
    return l2_normalize(largest_face.embedding)


def build_face_analyzer(model_name: str, det_width: int, det_height: int) -> FaceAnalysis:
    face_analyzer = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
    face_analyzer.prepare(ctx_id=0, det_size=(det_width, det_height))
    return face_analyzer


def enroll_students(
    students_dir: Path,
    output_registry: Path,
    output_metadata: Path,
    model_name: str,
    det_width: int,
    det_height: int,
    min_images: int,
) -> None:
    if not students_dir.exists():
        raise FileNotFoundError(f"Students directory not found: {students_dir}")

    analyzer = build_face_analyzer(model_name, det_width, det_height)

    enrolled_names: list[str] = []
    enrolled_embeddings: list[np.ndarray] = []
    meta_summary: dict[str, dict] = {}

    student_folders = sorted([path for path in students_dir.iterdir() if path.is_dir()])

    for student_folder in student_folders:
        student_id = student_folder.name
        embeddings: list[np.ndarray] = []
        processed_images = 0

        for image_path in list_images(student_folder):
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            processed_images += 1
            embedding = get_largest_face_embedding(analyzer, image)
            if embedding is not None:
                embeddings.append(embedding)

        if len(embeddings) < min_images:
            meta_summary[student_id] = {
                "status": "skipped",
                "processed_images": processed_images,
                "valid_embeddings": len(embeddings),
                "reason": f"Need at least {min_images} valid face images",
            }
            continue

        averaged_embedding = l2_normalize(np.mean(np.array(embeddings), axis=0))
        enrolled_names.append(student_id)
        enrolled_embeddings.append(averaged_embedding)

        meta_summary[student_id] = {
            "status": "enrolled",
            "processed_images": processed_images,
            "valid_embeddings": len(embeddings),
        }

    if not enrolled_embeddings:
        raise RuntimeError("No students were enrolled. Check image quality and folder layout.")

    embeddings_array = np.vstack(enrolled_embeddings)
    save_registry(output_registry, enrolled_names, embeddings_array)
    save_registry_metadata(
        output_metadata,
        {
            "enrolled_count": len(enrolled_names),
            "students": meta_summary,
        },
    )

    print(f"Enrollment complete. Enrolled {len(enrolled_names)} students.")
    print(f"Registry file: {output_registry}")
    print(f"Metadata file: {output_metadata}")


def parse_args() -> argparse.Namespace:
    defaults = DemoConfig()

    parser = argparse.ArgumentParser(description="Enroll students by generating face embeddings.")
    parser.add_argument("--students-dir", type=Path, default=defaults.students_dir)
    parser.add_argument("--output-registry", type=Path, default=defaults.registry_file)
    parser.add_argument("--output-metadata", type=Path, default=defaults.registry_meta_file)
    parser.add_argument("--model-name", default="buffalo_l")
    parser.add_argument("--det-width", type=int, default=640)
    parser.add_argument("--det-height", type=int, default=640)
    parser.add_argument("--min-images", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enroll_students(
        students_dir=args.students_dir,
        output_registry=args.output_registry,
        output_metadata=args.output_metadata,
        model_name=args.model_name,
        det_width=args.det_width,
        det_height=args.det_height,
        min_images=args.min_images,
    )


if __name__ == "__main__":
    main()
