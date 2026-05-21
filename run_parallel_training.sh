#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"

mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_L="${LOG_DIR}/yolo11l_${TS}.log"
LOG_N="${LOG_DIR}/yolo11n_${TS}.log"

cd "${ROOT_DIR}"

echo "[INFO] Root dir: ${ROOT_DIR}"
echo "[INFO] Logs:"
echo "  - ${LOG_L}"
echo "  - ${LOG_N}"

echo "[INFO] Starting YOLO11l on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python train_yolo11l.py > "${LOG_L}" 2>&1 &
PID_L=$!

echo "[INFO] Starting YOLO11n on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python train_yolo11n.py > "${LOG_N}" 2>&1 &
PID_N=$!

echo "[INFO] Started processes:"
echo "  YOLO11l PID=${PID_L}"
echo "  YOLO11n PID=${PID_N}"

echo "[INFO] To monitor:"
echo "  tail -f \"${LOG_L}\""
echo "  tail -f \"${LOG_N}\""
echo "  watch -n 1 nvidia-smi"

echo "[INFO] Waiting for both trainings to finish..."
wait "${PID_L}"
STATUS_L=$?
wait "${PID_N}"
STATUS_N=$?

echo "[INFO] Finished. Exit codes: YOLO11l=${STATUS_L}, YOLO11n=${STATUS_N}"

if [[ ${STATUS_L} -ne 0 || ${STATUS_N} -ne 0 ]]; then
  echo "[ERROR] At least one training failed. Check logs in ${LOG_DIR}" >&2
  exit 1
fi

echo "[INFO] Both trainings completed successfully."
