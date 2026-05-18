from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import os
import pandas as pd
from ultralytics.models.yolo.model import YOLO

os.environ["ULTRALYTICS_SKIP_CHECKS"] = "1"
os.environ["YOLO_SKIP_CHECKS"] = "1"

from src.utils.config import deep_merge_dicts, load_yaml_config
from src.utils.logging_utils import setup_logger
from src.utils.metrics import extract_validation_metrics, save_dataframe_csv, save_json
from src.utils.model_utils import resolve_model_source
from src.utils.paths import ensure_dir, get_project_root, resolve_path


def resolve_weights_path(
    model_config: dict,
    project_root: Path,
    weights_path: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> Path | str:
    if weights_path is not None:
        resolved_weights = resolve_path(weights_path, project_root)
        if resolved_weights is None:
            raise FileNotFoundError("Не удалось разрешить путь к весам.")
        return resolved_weights

    if run_dir is not None:
        resolved_run_dir = resolve_path(run_dir, project_root)
        if resolved_run_dir is not None:
            best_candidate = resolved_run_dir / "weights" / "best.pt"
            if best_candidate.exists():
                return best_candidate
            last_candidate = resolved_run_dir / "weights" / "last.pt"
            if last_candidate.exists():
                return last_candidate

    return resolve_model_source(model_config, project_root)


def validate_model(
    experiment_config_path: str | Path,
    model_config_path: str | Path,
    dataset_config_path: str | Path | None = None,
    weights_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    split: str = "val",
) -> dict:
    project_root = get_project_root()
    logger = setup_logger("vegvision.validate")

    experiment_config = load_yaml_config(experiment_config_path)
    model_config = load_yaml_config(model_config_path)

    default_dataset_config = experiment_config.get("data", {}).get("dataset_config")
    resolved_dataset_config = dataset_config_path or default_dataset_config
    if resolved_dataset_config is None:
        raise ValueError("Не указан dataset_config: передай его через CLI или в configs/experiment.yaml")

    resolved_dataset_config = resolve_path(resolved_dataset_config, project_root)
    if resolved_dataset_config is None or not resolved_dataset_config.exists():
        raise FileNotFoundError(f"Dataset config не найден: {resolved_dataset_config}")

    results_root = resolve_path(experiment_config.get("paths", {}).get("results_dir", "results"), project_root)
    if results_root is None:
        raise FileNotFoundError("Не удалось определить папку results из конфига.")

    resolved_weights = resolve_weights_path(
        model_config=model_config,
        project_root=project_root,
        weights_path=weights_path,
        run_dir=run_dir,
    )

    validation_kwargs = deep_merge_dicts(experiment_config.get("validation", {}), model_config.get("validation_overrides", {}))
    validation_kwargs["data"] = str(resolved_dataset_config)
    validation_kwargs["split"] = split
    validation_kwargs["verbose"] = False

    model_name = model_config.get("model_name", Path(str(resolved_weights)).stem)
    logger.info("Start validation: model=%s | weights=%s", model_name, resolved_weights)
    logger.info("Validation args: %s", validation_kwargs)

    model = YOLO(str(resolved_weights))
    val_results = model.val(**validation_kwargs)

    dataset_name = load_yaml_config(resolved_dataset_config).get("dataset_name")
    metrics = extract_validation_metrics(
        val_results=val_results,
        model_name=model_name,
        model_source=resolve_model_source(model_config, project_root),
        weights_path=resolved_weights,
        dataset_name=dataset_name,
        split=split,
    )
    metrics["experiment_config"] = str(Path(experiment_config_path))
    metrics["model_config"] = str(Path(model_config_path))
    metrics["dataset_config"] = str(resolved_dataset_config)
    metrics["weights_path"] = str(resolved_weights)

    metrics_dir = ensure_dir(results_root / "metrics")

    run_key = f"{model_name}_{split}"
    save_json(metrics, metrics_dir / f"{run_key}_validation_metrics.json")
    save_dataframe_csv(pd.DataFrame([metrics]), metrics_dir / f"{run_key}_validation_metrics.csv")

    logger.info("Validation finished. Metrics saved to %s", metrics_dir)
    return metrics


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Validate a YOLO model and save metrics.")
    parser.add_argument("--experiment-config", default="configs/experiment.yaml", help="Path to experiment YAML")
    parser.add_argument("--model-config", required=True, help="Path to model YAML")
    parser.add_argument("--dataset-config", default=None, help="Override dataset YAML path")
    parser.add_argument("--weights", default=None, help="Explicit path to model weights (.pt)")
    parser.add_argument("--run-dir", default=None, help="Training run directory containing weights/best.pt")
    parser.add_argument("--split", default="val", help="Dataset split: val/test")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    validate_model(
        experiment_config_path=args.experiment_config,
        model_config_path=args.model_config,
        dataset_config_path=args.dataset_config,
        weights_path=args.weights,
        run_dir=args.run_dir,
        split=args.split,
    )


if __name__ == "__main__":
    main()
