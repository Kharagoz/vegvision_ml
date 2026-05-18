from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from src.utils.config import load_yaml_config
from src.utils.logging_utils import setup_logger
from src.utils.metrics import dataframe_to_markdown, save_dataframe_csv
from src.utils.paths import ensure_dir, get_project_root, resolve_path
from typing import Dict, Any
import re


def collect_validation_metrics(metrics_dir: Path) -> pd.DataFrame:
    metric_files = sorted(metrics_dir.glob("*_validation_metrics.json"))
    rows: list[dict] = []

    for metric_file in metric_files:
        data = load_yaml_config(metric_file)
        data = dict(data or {})
        data["metrics_file"] = str(metric_file)

        model_conf_path = data.get("model_config")
        role = None
        if model_conf_path:
            try:
                model_conf = load_yaml_config(model_conf_path)
                role = model_conf.get("role")
            except Exception:
                role = None

        model_name = data.get("model_name") or ""
        if role is None and isinstance(model_name, str) and model_name:
            if model_name.endswith("n"):
                role = "edge"
            elif model_name.endswith("l"):
                role = "cloud"

        data["role"] = role

        family = model_name
        if isinstance(model_name, str) and re.search(r"[nl]$", model_name):
            family = model_name[:-1]
        data["family"] = family

        rows.append(data)

    if not rows:
        raise FileNotFoundError(f"В папке нет метрик для сравнения: {metrics_dir}")

    dataframe = pd.DataFrame(rows)

    preferred_columns = [
        "model_name",
        "dataset_name",
        "split",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "inference_ms",
        "model_size_mb",
        "weights_path",
    ]

    existing_columns = [column for column in preferred_columns if column in dataframe.columns]
    remaining_columns = [column for column in dataframe.columns if column not in existing_columns]
    return dataframe[existing_columns + remaining_columns]


def compare_models(
    experiment_config_path: str | Path,
    metrics_dir: str | Path | None = None,
    output_csv: str | Path | None = None,
    output_md: str | Path | None = None,
) -> pd.DataFrame:
    project_root = get_project_root()
    logger = setup_logger("vegvision.compare")

    experiment_config = load_yaml_config(experiment_config_path)

    resolved_metrics_dir = resolve_path(metrics_dir or experiment_config.get("paths", {}).get("metrics_dir", "results/metrics"), project_root)
    if resolved_metrics_dir is None:
        raise FileNotFoundError("Не удалось определить папку с метриками.")
    resolved_metrics_dir = ensure_dir(resolved_metrics_dir)

    dataframe = collect_validation_metrics(resolved_metrics_dir)
    sort_columns = [column for column in ["map50_95", "map50", "recall", "precision"] if column in dataframe.columns]
    if sort_columns:
        dataframe = dataframe.sort_values(by=sort_columns, ascending=False, na_position="last")

    comparisons_dir_path = resolve_path(experiment_config.get("paths", {}).get("comparisons_dir", "results/comparisons"), project_root)
    if comparisons_dir_path is None:
        raise FileNotFoundError("Не удалось определить папку для сравнительных таблиц.")
    comparisons_dir = ensure_dir(comparisons_dir_path)

    csv_path = resolve_path(output_csv, project_root) if output_csv is not None else comparisons_dir / "model_comparison.csv"
    if csv_path is None:
        raise FileNotFoundError("Не удалось определить путь для CSV-таблицы.")

    md_path = resolve_path(output_md, project_root) if output_md is not None else comparisons_dir / "model_comparison.md"
    if md_path is None:
        raise FileNotFoundError("Не удалось определить путь для Markdown-таблицы.")

    save_dataframe_csv(dataframe, csv_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(dataframe_to_markdown(dataframe), encoding="utf-8")

    edge_df = dataframe[dataframe["role"] == "edge"] if "role" in dataframe.columns else pd.DataFrame()
    cloud_df = dataframe[dataframe["role"] == "cloud"] if "role" in dataframe.columns else pd.DataFrame()

    edge_csv = comparisons_dir / "edge_models_comparison.csv"
    edge_md = comparisons_dir / "edge_models_comparison.md"
    cloud_csv = comparisons_dir / "cloud_models_comparison.csv"
    cloud_md = comparisons_dir / "cloud_models_comparison.md"

    if not edge_df.empty:
        save_dataframe_csv(edge_df, edge_csv)
        edge_md.write_text(dataframe_to_markdown(edge_df), encoding="utf-8")

    if not cloud_df.empty:
        save_dataframe_csv(cloud_df, cloud_csv)
        cloud_md.write_text(dataframe_to_markdown(cloud_df), encoding="utf-8")

    families = sorted(set(dataframe.get("family", [])))
    cascade_rows: list[Dict[str, Any]] = []
    for fam in families:
        edge_row = dataframe[(dataframe.get("family") == fam) & (dataframe.get("role") == "edge")]
        cloud_row = dataframe[(dataframe.get("family") == fam) & (dataframe.get("role") == "cloud")]
        if edge_row.empty and cloud_row.empty:
            continue

        er = edge_row.iloc[0].to_dict() if not edge_row.empty else {}
        cr = cloud_row.iloc[0].to_dict() if not cloud_row.empty else {}

        cascade = {
            "family": fam,
            "edge_model": er.get("model_name"),
            "edge_recall": er.get("recall"),
            "edge_inference_ms": er.get("inference_ms"),
            "edge_model_size_mb": er.get("model_size_mb"),
            "cloud_model": cr.get("model_name"),
            "cloud_precision": cr.get("precision"),
            "cloud_map50": cr.get("map50"),
            "cloud_map50_95": cr.get("map50_95"),
            "cloud_inference_ms": cr.get("inference_ms"),
            "cloud_model_size_mb": cr.get("model_size_mb"),
        }
        cascade_rows.append(cascade)

    if cascade_rows:
        cascade_df = pd.DataFrame(cascade_rows)
        cascade_csv = comparisons_dir / "cascade_comparison.csv"
        cascade_md = comparisons_dir / "cascade_comparison.md"
        save_dataframe_csv(cascade_df, cascade_csv)
        cascade_md.write_text(dataframe_to_markdown(cascade_df), encoding="utf-8")

    logger.info("Comparison tables saved to %s", comparisons_dir)
    print(dataframe.to_string(index=False))
    return dataframe


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Collect validation metrics and build a comparison table.")
    parser.add_argument("--experiment-config", default="configs/experiment.yaml", help="Path to experiment YAML")
    parser.add_argument("--metrics-dir", default=None, help="Folder with validation metrics JSON files")
    parser.add_argument("--output-csv", default=None, help="Where to save CSV comparison table")
    parser.add_argument("--output-md", default=None, help="Where to save Markdown comparison table")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    compare_models(
        experiment_config_path=args.experiment_config,
        metrics_dir=args.metrics_dir,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )


if __name__ == "__main__":
    main()
