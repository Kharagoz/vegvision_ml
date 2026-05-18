#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys
import os

os.environ["ULTRALYTICS_SKIP_CHECKS"] = "1"
os.environ["YOLO_SKIP_CHECKS"] = "1"

def run_pilot_training():
    project_root = Path(__file__).parent
    
    models = [
        "yolo11n",
        "yolo11l",
        "yolo26n",
        "yolo26l",
    ]
    
    results = {}
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Starting pilot training for {model_name.upper()}")
        print(f"{'='*60}\n")
        
        cmd = [
            sys.executable,
            "-m", "src.train",
            "--experiment-config", "configs/experiment_pilot.yaml",
            "--model-config", f"configs/models/{model_name}.yaml",
        ]
        
        result = subprocess.run(cmd, cwd=project_root)
        results[model_name] = "SUCCESS" if result.returncode == 0 else "FAILED"
    
    print(f"\n{'='*60}")
    print("Pilot Training Summary")
    print(f"{'='*60}")
    for model, status in results.items():
        print(f"{model:12} | {status}")

if __name__ == "__main__":
    run_pilot_training()
