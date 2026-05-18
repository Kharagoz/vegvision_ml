#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys
import os

os.environ["ULTRALYTICS_SKIP_CHECKS"] = "1"
os.environ["YOLO_SKIP_CHECKS"] = "1"

def run_full_training():
    project_root = Path(__file__).parent
    
    models = [
        "yolo11n",
        "yolo11l",
        "yolo26n",
        "yolo26l",
    ]
    
    results = {}
    
    print("\n" + "="*60)
    print("FULL TRAINING FOR ALL MODELS (100 EPOCHS)")
    print("WARNING: This will take a very long time on CPU!")
    print("="*60 + "\n")
    
    for idx, model_name in enumerate(models, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(models)}] Starting full training for {model_name.upper()}")
        print(f"Model: {model_name} | Epochs: 100 | Batch: 16")
        print(f"{'='*60}\n")
        
        cmd = [
            sys.executable,
            "-m", "src.train",
            "--experiment-config", "configs/experiment.yaml",
            "--model-config", f"configs/models/{model_name}.yaml",
        ]
        
        result = subprocess.run(cmd, cwd=project_root)
        results[model_name] = "SUCCESS" if result.returncode == 0 else "FAILED"
        
        if result.returncode == 0:
            print(f"\nTraining completed for {model_name}")
        else:
            print(f"\nTraining FAILED for {model_name}")
    
    print(f"\n{'='*60}")
    print("Full Training Summary")
    print(f"{'='*60}")
    for model, status in results.items():
        status_mark = "ok" if status == "SUCCESS" else "error"
        print(f"{status_mark} {model:12} | {status}")
    
    print(f"\n{'='*60}")
    print("Next steps:")
    print("1. Run validation: python run_validation.py")
    print("2. Compare models: python run_comparison.py")
    print("3. Review results in results/comparisons/")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    run_full_training()
