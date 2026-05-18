from pathlib import Path

from ultralytics import YOLO


def predict_single_image(
	model_path: str | Path = "yolo11n.pt",
	image_path: str | Path | None = None,
	output_dir: str | Path = Path("runs") / "predict_single",
) -> int:

	project_root = Path(__file__).resolve().parents[2]

	model_path = Path(model_path)
	output_dir = Path(output_dir)


	if image_path is None:
		test_dir = project_root / "data" / "datasets" / "tomatoes_v1" / "test" / "images"
		images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
		if images:
			image_path = images[0]
		else:
			raise FileNotFoundError(f"Нет изображений в {test_dir}")
	else:
		image_path = Path(image_path)

	if not model_path.is_absolute():
		model_path = project_root / model_path
	if not image_path.is_absolute():
		image_path = project_root / image_path
	if not output_dir.is_absolute():
		output_dir = project_root / output_dir

	if not image_path.exists():
		raise FileNotFoundError(f"Изображение не найдено: {image_path}")

	model = YOLO(str(model_path))

	output_dir.mkdir(parents=True, exist_ok=True)

	results = model.predict(
		source=str(image_path),
		save=True,
		project=str(output_dir.parent),
		name=output_dir.name,
		exist_ok=True,
		verbose=False,
	)

	detected_count = len(results[0].boxes) if results and results[0].boxes is not None else 0

	print(f"Найдено объектов: {detected_count}")
	print(f"Результат сохранён в: {output_dir}")

	return detected_count


if __name__ == "__main__":
	predict_single_image()
