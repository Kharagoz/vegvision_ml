from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    data_yaml = (root_dir / "dataset" / "data.yaml").resolve()
    project_dir = (root_dir / "training_runs" / "yolo11l").resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Не найден dataset/data.yaml по пути: {data_yaml}. "
            "Проверьте наличие файла dataset/data.yaml в корне проекта."
        )

    model = YOLO("yolo11l.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=70,
        imgsz=640,
        batch=28,
        patience=15,
        amp=True,
        plots=True,
        save=True,
        verbose=True,
        device=0,
        project=str(project_dir),
        name="tomato_cucumber_yolo11l",
        exist_ok=True,
    )

    save_dir = getattr(results, "save_dir", project_dir / "tomato_cucumber_yolo11l")
    print(f"Обучение YOLO11l завершено. Результаты сохранены в: {Path(save_dir).resolve()}")


if __name__ == "__main__":
    main()
