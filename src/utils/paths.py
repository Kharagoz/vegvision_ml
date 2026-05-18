from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_value: str | Path | None, base_dir: Path | None = None) -> Path | None:
    if path_value is None:
        return None

    path = Path(path_value)
    if path.is_absolute():
        return path

    base = base_dir if base_dir is not None else get_project_root()
    return base / path


def ensure_parent_dir(path_value: str | Path) -> Path:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path_value: str | Path) -> Path:
    path = Path(path_value)
    path.mkdir(parents=True, exist_ok=True)
    return path
