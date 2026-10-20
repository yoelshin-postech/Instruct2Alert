#!/usr/bin/env bash
set -euo pipefail

# One-command launcher:
# 1) Start vLLM (Qwen) in `vlm` environment on GPU 0.
# 2) Wait until the API is ready.
# 3) Run CV pipeline in `cv` environment.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-}"
LOG_DIR="${RUN_ROOT:+${RUN_ROOT}/logs}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/data/logs}"
mkdir -p "${LOG_DIR}"

VLM_ENV="${VLM_ENV:-vlm}"
CV_ENV="${CV_ENV:-cv}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-/path/to/Qwen3-VL-8B-Instruct}"
VLM_SERVED_MODEL="${VLM_SERVED_MODEL:-qwen-vl-8b}"
VLM_PORT="${VLM_PORT:-8000}"
VLM_GPU="${VLM_GPU:-0}"
VLM_GPU_MEM="${VLM_GPU_MEM:-0.90}"
VLM_MAX_MODEL_LEN="${VLM_MAX_MODEL_LEN:-8192}"
RUN_SEED="${RUN_SEED:-2027}"
CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
VLM_STARTUP_RETRIES="${VLM_STARTUP_RETRIES:-120}"  # 120 * 2s = 240s
VLM_LOG_FILE="${LOG_DIR}/vllm_server.log"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda command not found."
  exit 1
fi

if [ ! -f "${ROOT_DIR}/run_e2e_pipeline.py" ]; then
  echo "[ERROR] run_e2e_pipeline.py not found in: ${ROOT_DIR}"
  exit 1
fi

if [ ! -d "${VLM_MODEL_PATH}" ]; then
  echo "[ERROR] VLM model path not found: ${VLM_MODEL_PATH}"
  echo "        Set VLM_MODEL_PATH and retry."
  exit 1
fi

cleanup() {
  if [ -n "${VLLM_PID:-}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[INFO] Stopping vLLM server (pid=${VLLM_PID})..."
    # vLLM runs in its own process group (setsid below). Signal the whole group
    # so the API server and engine processes under `conda run` exit as well.
    kill -TERM -- "-${VLLM_PID}" 2>/dev/null || kill "${VLLM_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 -- "-${VLLM_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[INFO] Launching vLLM in env='${VLM_ENV}', gpu=${VLM_GPU} ..."
echo "[INFO] vLLM log: ${VLM_LOG_FILE}"
echo "[INFO] Seed: ${RUN_SEED}"

setsid conda run -n "${VLM_ENV}" env CUDA_VISIBLE_DEVICES="${VLM_GPU}" \
  PYTHONHASHSEED="${RUN_SEED}" CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG}" \
  python -m vllm.entrypoints.openai.api_server \
    --model "${VLM_MODEL_PATH}" \
    --served-model-name "${VLM_SERVED_MODEL}" \
    --port "${VLM_PORT}" \
    --gpu-memory-utilization "${VLM_GPU_MEM}" \
    --max-model-len "${VLM_MAX_MODEL_LEN}" \
    --seed "${RUN_SEED}" \
    --trust-remote-code \
    > "${VLM_LOG_FILE}" 2>&1 &
VLLM_PID=$!

echo "[INFO] Waiting for vLLM API: http://127.0.0.1:${VLM_PORT}/v1/models"
for ((i=1; i<=VLM_STARTUP_RETRIES; i++)); do
  if curl -sSf "http://127.0.0.1:${VLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[INFO] vLLM is ready."
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[ERROR] vLLM process exited before startup completed."
    echo "        Check log: ${VLM_LOG_FILE}"
    exit 1
  fi
  sleep 2
done

if ! curl -sSf "http://127.0.0.1:${VLM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "[ERROR] vLLM startup timeout after $((VLM_STARTUP_RETRIES * 2))s."
  echo "        Check log: ${VLM_LOG_FILE}"
  exit 1
fi

echo "[INFO] Running CV pipeline in env='${CV_ENV}' ..."
cd "${ROOT_DIR}"
conda run -n "${CV_ENV}" env \
  PYTHONHASHSEED="${RUN_SEED}" CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG}" \
  python run_e2e_pipeline.py --run-seed "${RUN_SEED}" "$@"
