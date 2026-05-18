#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys
import os
from glob import glob

os.environ["ULTRALYTICS_SKIP_CHECKS"] = "1"
os.environ["YOLO_SKIP_CHECKS"] = "1"

def run_validation():
    project_root = Path(__file__).parent
    
    models = [
        "yolo11n",
        "yolo11l",
        "yolo26n",
        "yolo26l",
    ]

    runs_dir = project_root / "runs" / "train"
    
    results = {}
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Validating {model_name.upper()}")
        print(f"{'='*60}\n")

        pattern = str(runs_dir / f"{model_name}_*")
        matching_runs = sorted(glob(pattern), reverse=True)
        
        if not matching_runs:
            print(f"No trained run found for {model_name}")
            results[model_name] = "NOT_FOUND"
            continue
        
        latest_run = matching_runs[0]
        run_dir = f"runs/train/{Path(latest_run).name}"
        
        cmd = [
            sys.executable,
            "-m", "src.validate",
            "--experiment-config", "configs/experiment.yaml",
            "--model-config", f"configs/models/{model_name}.yaml",
            "--run-dir", run_dir,
        ]
        
        result = subprocess.run(cmd, cwd=project_root)
        results[model_name] = "SUCCESS" if result.returncode == 0 else "FAILED"
    
    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")
    for model, status in results.items():
        print(f"{model:12} | {status}")

if __name__ == "__main__":
    run_validation()
