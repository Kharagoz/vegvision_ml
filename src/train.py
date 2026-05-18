from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import os

from ultralytics.models.yolo.model import YOLO

os.environ["ULTRALYTICS_SKIP_CHECKS"] = "1"
os.environ["YOLO_SKIP_CHECKS"] = "1"

from src.utils.config import deep_merge_dicts, load_yaml_config
from src.utils.logging_utils import setup_logger
from src.utils.metrics import save_json
from src.utils.model_utils import resolve_model_source
from src.utils.paths import ensure_dir, get_project_root, resolve_path


def build_train_config(
    experiment_config: dict,
    model_config: dict,
    dataset_config_path: Path,
    run_name: str,
    project_root: Path,
) -> dict:

    train_kwargs = deep_merge_dicts(experiment_config.get("train", {}), model_config.get("train_overrides", {}))

    runs_root = resolve_path(experiment_config.get("paths", {}).get("runs_dir", "runs"), project_root)
    if runs_root is None:
        raise FileNotFoundError("Не удалось определить папку runs из конфига.")
    train_kwargs["project"] = str(ensure_dir(runs_root / "train"))
    train_kwargs["name"] = run_name
    train_kwargs["data"] = str(dataset_config_path)
    train_kwargs["exist_ok"] = True
    train_kwargs["seed"] = experiment_config.get("project", {}).get("seed", 42)
    train_kwargs["verbose"] = True

    return train_kwargs


def train_model(
    experiment_config_path: str | Path,
    model_config_path: str | Path,
    dataset_config_path: str | Path | None = None,
    run_name: str | None = None,
) -> Path:
    project_root = get_project_root()
    logger = setup_logger("vegvision.train")

    experiment_config = load_yaml_config(experiment_config_path)
    model_config = load_yaml_config(model_config_path)

    default_dataset_config = experiment_config.get("data", {}).get("dataset_config")
    resolved_dataset_config = dataset_config_path or default_dataset_config
    if resolved_dataset_config is None:
        raise ValueError("Не указан dataset_config: передай его через CLI или в configs/experiment.yaml")

    resolved_dataset_config = resolve_path(resolved_dataset_config, project_root)
    if resolved_dataset_config is None or not resolved_dataset_config.exists():
        raise FileNotFoundError(f"Dataset config не найден: {resolved_dataset_config}")

    model_source = resolve_model_source(model_config, project_root)
    model_name = model_config.get("model_name", Path(str(model_source)).stem)
    final_run_name = run_name or f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    train_kwargs = build_train_config(
        experiment_config=experiment_config,
        model_config=model_config,
        dataset_config_path=resolved_dataset_config,
        run_name=final_run_name,
        project_root=project_root,
    )

    logger.info("Start training: model=%s | source=%s", model_name, model_source)
    logger.info("Dataset config: %s", resolved_dataset_config)
    logger.info("Train args: %s", train_kwargs)

    model = YOLO(str(model_source))
    results = model.train(**train_kwargs)

    save_dir = Path(getattr(results, "save_dir", Path(train_kwargs["project"]) / final_run_name))
    best_weights = save_dir / "weights" / "best.pt"

    results_root = resolve_path(experiment_config.get("paths", {}).get("results_dir", "results"), project_root)
    if results_root is None:
        raise FileNotFoundError("Не удалось определить папку results из конфига.")
    logs_dir = ensure_dir(results_root / "logs")

    summary = {
        "model_name": model_name,
        "model_source": str(model_source),
        "dataset_config": str(resolved_dataset_config),
        "run_name": final_run_name,
        "save_dir": str(save_dir),
        "best_weights": str(best_weights),
        "train_kwargs": train_kwargs,
        "experiment_config": str(Path(experiment_config_path)),
        "model_config": str(Path(model_config_path)),
    }
    save_json(summary, logs_dir / f"{final_run_name}_train_summary.json")

    logger.info("Training finished. Best weights: %s", best_weights)
    return best_weights


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Train one YOLO model with shared experiment settings.")
    parser.add_argument("--experiment-config", default="configs/experiment.yaml", help="Path to experiment YAML")
    parser.add_argument("--model-config", required=True, help="Path to model YAML")
    parser.add_argument("--dataset-config", default=None, help="Override dataset YAML path")
    parser.add_argument("--run-name", default=None, help="Custom run name")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    train_model(
        experiment_config_path=args.experiment_config,
        model_config_path=args.model_config,
        dataset_config_path=args.dataset_config,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
