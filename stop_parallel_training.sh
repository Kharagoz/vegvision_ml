#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="${ROOT_DIR}/pids"

if [[ ! -d "${PID_DIR}" ]]; then
  echo "[WARN] PID directory not found: ${PID_DIR}"
  exit 0
fi

stop_by_pattern() {
  local pattern="$1"
  local latest

  latest="$(ls -1t "${PID_DIR}"/${pattern}_*.pid 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest}" ]]; then
    echo "[INFO] No PID file for ${pattern}"
    return 0
  fi

  local pid
  pid="$(cat "${latest}" 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    echo "[WARN] Empty PID in ${latest}"
    return 0
  fi

  if ps -p "${pid}" > /dev/null 2>&1; then
    echo "[INFO] Stopping ${pattern} (PID=${pid})..."
    kill "${pid}" || true
    sleep 2

    if ps -p "${pid}" > /dev/null 2>&1; then
      echo "[WARN] PID ${pid} still alive, sending SIGKILL..."
      kill -9 "${pid}" || true
      sleep 1
    fi

    if ps -p "${pid}" > /dev/null 2>&1; then
      echo "[ERROR] Failed to stop PID ${pid}"
    else
      echo "[INFO] Stopped ${pattern} (PID=${pid})"
    fi
  else
    echo "[INFO] Process ${pid} for ${pattern} is already not running"
  fi
}

stop_by_pattern "yolo11l"
stop_by_pattern "yolo11n"

echo "[INFO] Done."
