from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DemoConfig:
    students_dir: Path = PROJECT_ROOT / "data" / "students"
    roster_csv: Path = PROJECT_ROOT / "data" / "students" / "roster.csv"
    registry_file: Path = PROJECT_ROOT / "data" / "registry" / "embeddings.npz"
    registry_meta_file: Path = PROJECT_ROOT / "data" / "registry" / "metadata.json"
    events_csv: Path = PROJECT_ROOT / "outputs" / "logs" / "events.csv"
    review_actions_csv: Path = PROJECT_ROOT / "outputs" / "logs" / "review_actions.csv"
    reports_dir: Path = PROJECT_ROOT / "outputs" / "reports"
    alerts_dir: Path = PROJECT_ROOT / "outputs" / "alerts"
