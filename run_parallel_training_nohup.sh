#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PID_DIR="${ROOT_DIR}/pids"

mkdir -p "${LOG_DIR}" "${PID_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_L="${LOG_DIR}/yolo11l_${TS}.log"
LOG_N="${LOG_DIR}/yolo11n_${TS}.log"
PID_L_FILE="${PID_DIR}/yolo11l_${TS}.pid"
PID_N_FILE="${PID_DIR}/yolo11n_${TS}.pid"

cd "${ROOT_DIR}"

echo "[INFO] Root dir: ${ROOT_DIR}"
echo "[INFO] Logs:"
echo "  - ${LOG_L}"
echo "  - ${LOG_N}"

echo "[INFO] Starting YOLO11l on GPU 0 with nohup..."
nohup env CUDA_VISIBLE_DEVICES=0 python train_yolo11l.py > "${LOG_L}" 2>&1 &
PID_L=$!
echo "${PID_L}" > "${PID_L_FILE}"

echo "[INFO] Starting YOLO11n on GPU 1 with nohup..."
nohup env CUDA_VISIBLE_DEVICES=1 python train_yolo11n.py > "${LOG_N}" 2>&1 &
PID_N=$!
echo "${PID_N}" > "${PID_N_FILE}"

echo "[INFO] Started in detached mode:"
echo "  YOLO11l PID=${PID_L} (saved to ${PID_L_FILE})"
echo "  YOLO11n PID=${PID_N} (saved to ${PID_N_FILE})"

echo "[INFO] Useful commands:"
echo "  tail -f \"${LOG_L}\""
echo "  tail -f \"${LOG_N}\""
echo "  ps -fp ${PID_L} ${PID_N}"
echo "  watch -n 1 nvidia-smi"
echo "  kill ${PID_L} ${PID_N}"

echo "[INFO] Script finished. Training keeps running in background."
