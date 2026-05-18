from __future__ import annotations

from pathlib import Path
from typing import Any

from .paths import get_project_root, resolve_path


def _normalize_candidate(candidate: Any, project_root: Path) -> str | Path | None:
    if candidate is None:
        return None

    candidate_str = str(candidate)
    candidate_path = Path(candidate_str)

    if candidate_path.is_absolute() and candidate_path.exists():
        return candidate_path

    resolved = resolve_path(candidate_str, project_root)
    if resolved is not None and resolved.exists():
        return resolved

    if candidate_str.startswith(("http://", "https://")):
        return candidate_str

    if candidate_str.endswith((".pt", ".yaml", ".yml")):
        return candidate_str

    return candidate_str


def resolve_model_source(model_config: dict[str, Any], project_root: Path | None = None) -> str | Path:
    root = project_root or get_project_root()
    model_source_cfg = model_config.get("model_source", {})
    source_type = str(model_source_cfg.get("type", "weights")).lower()

    candidates: list[Any] = [model_source_cfg.get("value")]
    if source_type == "auto":
        candidates.append(model_source_cfg.get("fallback_value"))

    for candidate in candidates:
        normalized = _normalize_candidate(candidate, root)
        if normalized is not None:
            return normalized

    raise FileNotFoundError("Не удалось определить источник модели из конфига")
