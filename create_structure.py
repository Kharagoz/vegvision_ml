from pathlib import Path


def create_vegvision_structure(base_dir: Path) -> None:
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "notebooks",
        "src/data",
        "src/train",
        "src/inference",
        "configs",
        "runs",
    ]

    files = [
        "requirements.txt",
        "README.md",
    ]

    for relative_dir in directories:
        (base_dir / relative_dir).mkdir(parents=True, exist_ok=True)

    for relative_file in files:
        (base_dir / relative_file).touch(exist_ok=True)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    create_vegvision_structure(project_root)
    print(f"Project structure is ready at: {project_root}")
