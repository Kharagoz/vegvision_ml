from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .paths import ensure_parent_dir


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")

    with path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)

    return content or {}


def save_yaml_config(data: dict[str, Any], config_path: str | Path) -> Path:

    path = ensure_parent_dir(config_path)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
    return path


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)

    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged
