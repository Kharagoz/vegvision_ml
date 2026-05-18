from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import ensure_parent_dir


def model_size_mb(model_path: str | Path) -> float | None:
    path = Path(model_path)
    if not path.exists():
        return None
    return round(path.stat().st_size / (1024 ** 2), 2)


def extract_validation_metrics(
    val_results: Any,
    model_name: str,
    model_source: str | Path,
    weights_path: str | Path,
    dataset_name: str | None = None,
    split: str = "val",
) -> dict[str, Any]:

    box_metrics = getattr(val_results, "box", None)
    speed = getattr(val_results, "speed", {}) or {}

    precision = getattr(box_metrics, "mp", None) if box_metrics is not None else None
    recall = getattr(box_metrics, "mr", None) if box_metrics is not None else None
    map50 = getattr(box_metrics, "map50", None) if box_metrics is not None else None
    map50_95 = getattr(box_metrics, "map", None) if box_metrics is not None else None

    return {
        "model_name": model_name,
        "model_source": str(model_source),
        "weights_path": str(weights_path),
        "dataset_name": dataset_name,
        "split": split,
        "precision": round(float(precision), 6) if precision is not None else None,
        "recall": round(float(recall), 6) if recall is not None else None,
        "map50": round(float(map50), 6) if map50 is not None else None,
        "map50_95": round(float(map50_95), 6) if map50_95 is not None else None,
        "inference_ms": round(float(speed.get("inference", 0.0)), 3) if isinstance(speed, dict) else None,
        "preprocess_ms": round(float(speed.get("preprocess", 0.0)), 3) if isinstance(speed, dict) else None,
        "postprocess_ms": round(float(speed.get("postprocess", 0.0)), 3) if isinstance(speed, dict) else None,
        "model_size_mb": model_size_mb(weights_path),
    }


def save_json(data: dict[str, Any], file_path: str | Path) -> Path:
    path = ensure_parent_dir(file_path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    return path


def save_dataframe_csv(dataframe: pd.DataFrame, file_path: str | Path) -> Path:
    path = ensure_parent_dir(file_path)
    dataframe.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "| empty |\n| --- |\n| no rows |"

    headers = list(dataframe.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in dataframe.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)
