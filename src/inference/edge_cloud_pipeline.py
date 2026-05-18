from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_path(path_value: str | Path, project_root: Path) -> Path:
    path_value = Path(path_value)
    return path_value if path_value.is_absolute() else project_root / path_value


def load_model(model_path: Path, model_role: str) -> YOLO:
    print(f"[INFO] Загружаю {model_role}-модель: {model_path}")
    return YOLO(str(model_path))


def run_model_inference(model: YOLO, image_path: Path, conf_threshold: float, model_role: str):
    print(f"[INFO] Запускаю {model_role}-инференс (conf={conf_threshold})")
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        verbose=False,
    )
    return results[0]


def save_result_image(result, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotted_image = result.plot()
    is_saved = cv2.imwrite(str(output_path), plotted_image)

    if not is_saved:
        raise RuntimeError(f"Не удалось сохранить результат: {output_path}")


def is_image_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def process_one_image(
    image_path: Path,
    edge_model: YOLO,
    cloud_model: YOLO,
    output_dir: Path,
    edge_conf: float,
    cloud_conf: float,
) -> tuple[int, int, bool]:

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV не смог прочитать изображение: {image_path}")

    print(f"\n[INFO] Обрабатываю: {image_path.name}")

    edge_result = run_model_inference(edge_model, image_path, edge_conf, model_role="edge")
    edge_detections = len(edge_result.boxes) if edge_result.boxes is not None else 0
    print(f"[INFO] Edge-детекций: {edge_detections}")

    if edge_detections == 0:
        print("[INFO] Подозрительных объектов нет. Кадр в cloud не отправляется.")
        return edge_detections, 0, False


    cloud_result = run_model_inference(cloud_model, image_path, cloud_conf, model_role="cloud")
    cloud_detections = len(cloud_result.boxes) if cloud_result.boxes is not None else 0
    print(f"[INFO] Cloud-детекций: {cloud_detections}")


    output_path = output_dir / image_path.name
    save_result_image(cloud_result, output_path)
    print(f"[INFO] Финальный результат сохранён: {output_path}")

    return edge_detections, cloud_detections, True


def edge_cloud_pipeline(
    image_path: str | Path | None = None,
    edge_model_path: str | Path = "yolo11n.pt",
    cloud_model_path: str | Path = "yolo11l.pt",
    output_image_path: str | Path = Path("runs") / "edge_cloud" / "final_result.jpg",
    edge_conf: float = 0.20,
    cloud_conf: float = 0.40,
) -> int:

    project_root = Path(__file__).resolve().parents[2]


    if image_path is None:
        test_dir = project_root / "data" / "datasets" / "tomatoes_v1" / "test" / "images"
        images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        if images:
            image_path = images[0]
        else:
            raise FileNotFoundError(f"Нет изображений в {test_dir}")
    else:
        image_path = resolve_path(image_path, project_root)

    edge_model_path = resolve_path(edge_model_path, project_root)
    cloud_model_path = resolve_path(cloud_model_path, project_root)
    output_image_path = resolve_path(output_image_path, project_root)

    if not image_path.exists():
        raise FileNotFoundError(f"Входное изображение не найдено: {image_path}")

    print(f"[INFO] Входное изображение: {image_path}")

    edge_model = load_model(edge_model_path, model_role="edge")
    cloud_model = load_model(cloud_model_path, model_role="cloud")

    edge_count, cloud_count, cloud_used = process_one_image(
        image_path=image_path,
        edge_model=edge_model,
        cloud_model=cloud_model,
        output_dir=output_image_path.parent,
        edge_conf=edge_conf,
        cloud_conf=cloud_conf,
    )

    if not cloud_used:
        return 0
    return cloud_count


def edge_cloud_pipeline_batch(
    image_dir: str | Path | None = None,
    edge_model_path: str | Path = "yolo11n.pt",
    cloud_model_path: str | Path = "yolo11l.pt",
    output_dir: str | Path = Path("runs") / "edge_cloud",
    edge_conf: float = 0.20,
    cloud_conf: float = 0.40,
) -> dict[str, int]:

    project_root = Path(__file__).resolve().parents[2]

    if image_dir is None:
        image_dir = project_root / "data" / "datasets" / "tomatoes_v1" / "test" / "images"
    else:
        image_dir = resolve_path(image_dir, project_root)

    edge_model_path = resolve_path(edge_model_path, project_root)
    cloud_model_path = resolve_path(cloud_model_path, project_root)
    output_dir = resolve_path(output_dir, project_root)

    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Папка с изображениями не найдена: {image_dir}")

    image_files = sorted([p for p in image_dir.iterdir() if p.is_file() and is_image_file(p)])
    if not image_files:
        raise FileNotFoundError(f"В папке нет изображений: {image_dir}")

    print(f"[INFO] Найдено изображений: {len(image_files)}")
    print(f"[INFO] Папка входа: {image_dir}")
    print(f"[INFO] Папка выхода: {output_dir}")

    edge_model = load_model(edge_model_path, model_role="edge")
    cloud_model = load_model(cloud_model_path, model_role="cloud")

    total_images = 0
    sent_to_cloud = 0
    cloud_positive_images = 0
    total_edge_detections = 0
    total_cloud_detections = 0

    for image_path in image_files:
        total_images += 1
        edge_count, cloud_count, cloud_used = process_one_image(
            image_path=image_path,
            edge_model=edge_model,
            cloud_model=cloud_model,
            output_dir=output_dir,
            edge_conf=edge_conf,
            cloud_conf=cloud_conf,
        )

        total_edge_detections += edge_count
        total_cloud_detections += cloud_count

        if cloud_used:
            sent_to_cloud += 1
        if cloud_count > 0:
            cloud_positive_images += 1

    summary = {
        "total_images": total_images,
        "sent_to_cloud": sent_to_cloud,
        "cloud_positive_images": cloud_positive_images,
        "total_edge_detections": total_edge_detections,
        "total_cloud_detections": total_cloud_detections,
    }

    print("\n[SUMMARY] Готово")
    print(f"[SUMMARY] Всего изображений: {summary['total_images']}")
    print(f"[SUMMARY] Отправлено в cloud: {summary['sent_to_cloud']}")
    print(f"[SUMMARY] Cloud-позитивных изображений: {summary['cloud_positive_images']}")
    print(f"[SUMMARY] Всего edge-детекций: {summary['total_edge_detections']}")
    print(f"[SUMMARY] Всего cloud-детекций: {summary['total_cloud_detections']}")

    return summary


if __name__ == "__main__":
    try:
        summary = edge_cloud_pipeline_batch()
        print(f"[DONE] Batch-пайплайн завершён. Обработано: {summary['total_images']}")
    except Exception as error:
        print(f"[ERROR] {error}")
