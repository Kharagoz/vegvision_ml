#!/usr/bin/env python3

from pathlib import Path
import subprocess
import sys
import os

os.environ["ULTRALYTICS_SKIP_CHECKS"] = "1"
os.environ["YOLO_SKIP_CHECKS"] = "1"

def run_comparison():
    project_root = Path(__file__).parent
    
    print(f"\n{'='*60}")
    print("Generating Model Comparison Tables")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable,
        "-m", "src.compare_models",
        "--experiment-config", "configs/experiment.yaml",
    ]
    
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print(f"\n{'='*60}")
        print("Comparison tables generated successfully!")
        print(f"{'='*60}")
        print("\nGenerated files:")
        comparisons_dir = project_root / "results" / "comparisons"
        if comparisons_dir.exists():
            for file in sorted(comparisons_dir.glob("*.csv")) + sorted(comparisons_dir.glob("*.md")):
                print(f"  - {file.relative_to(project_root)}")
    else:
        print(f"\n{'='*60}")
        print("Comparison generation FAILED")
        print(f"{'='*60}")

if __name__ == "__main__":
    run_comparison()
