#!/usr/bin/env bash
set -euo pipefail

# Push the Android build of vlm-rknn-cache to a device and generate an RKLLM
# prompt cache.
#
# Usage:
#   ./scripts/run-android-cache.sh <device-ip> <model> <prompt> [remote-path]
#
# Arguments:
#   device-ip    IP address (or host) of the target device, as used by `adb connect`.
#   model        Model to run: qwen2-vl-2b, qwen2-vl-7b, or gemma3-4b.
#   prompt       Fixed prompt prefix to store in the cache.
#   remote-path  Directory to push files to on the device (default /data/local/tmp).
#
# Environment:
#   CACHE_PATH   Remote cache output path
#                (default <remote-path>/models/<model>/prompt.cache).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-android}"
MODEL_CACHE_ROOT="${ROOT_DIR}/.cache"

load_model_config() {
  case "$1" in
    qwen2-vl-2b)
      MODEL_FAMILY="qwen2-vl"
      HF_REPO="3ib0n/Qwen2-VL-2B-rkllm"
      LLM_FILE="Qwen2-VL-2B-Instruct.rkllm"
      MODEL_CACHE_NAME="qwen2-vl-2b"
      DOWNLOAD_SOURCE="huggingface"
      ;;
    qwen2-vl-7b)
      MODEL_FAMILY="qwen2-vl"
      HF_REPO="3ib0n/Qwen2-VL-7B-rkllm"
      LLM_FILE="Qwen2-VL-7B-Instruct.rkllm"
      MODEL_CACHE_NAME="qwen2-vl-7b"
      DOWNLOAD_SOURCE="huggingface"
      ;;
    gemma3-4b)
      MODEL_FAMILY="gemma3"
      HF_REPO=""
      LLM_FILE="gemma3-4b-it.rkllm"
      MODEL_CACHE_NAME="gemma3-4b"
      DOWNLOAD_SOURCE="cache"
      ;;
    *)
      return 1
      ;;
  esac
}

DEVICE_IP="${1:-}"
MODEL="${2:-}"
PROMPT="${3:-}"
REMOTE_DIR="${4:-/data/local/tmp}"

if [[ -z "${DEVICE_IP}" || -z "${MODEL}" || -z "${PROMPT}" ]]; then
  cat <<USAGE
Usage: $0 <device-ip> <model> <prompt> [remote-path]

Arguments:
  device-ip    IP address (or host) of the target device, as used by adb connect.
  model        Model to run: qwen2-vl-2b, qwen2-vl-7b, or gemma3-4b.
  prompt       Fixed prompt prefix to store in the cache.
  remote-path  Directory to push files to on the device (default /data/local/tmp).

Environment:
  CACHE_PATH   Remote cache output path
               (default <remote-path>/models/<model>/prompt.cache).
USAGE
  exit 1
fi

if ! load_model_config "${MODEL}"; then
  echo "Error: unsupported model '${MODEL}'." >&2
  echo "Expected one of: qwen2-vl-2b, qwen2-vl-7b, gemma3-4b." >&2
  exit 1
fi

MODEL_CACHE="${MODEL_CACHE_ROOT}/${MODEL_CACHE_NAME}"
REMOTE_MODEL_DIR="${REMOTE_DIR}/models/${MODEL}"
CACHE_PATH="${CACHE_PATH:-${REMOTE_MODEL_DIR}/prompt.cache}"

echo "=== Check preconditions ==="

CACHE_BIN="${BUILD_DIR}/vlm-rknn-cache"
if [[ ! -x "${CACHE_BIN}" ]]; then
  echo "Error: ${CACHE_BIN} not found." >&2
  echo "Run the Android build first: ./scripts/build-android.sh docker" >&2
  exit 1
fi

REQUIRED_TOOLS=(adb)
if [[ "${DOWNLOAD_SOURCE}" == "huggingface" ]]; then
  REQUIRED_TOOLS+=(curl)
fi
for tool in "${REQUIRED_TOOLS[@]}"; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "Error: required tool '${tool}' is not installed or not on PATH." >&2
    exit 1
  fi
done

echo "Cache binary: ${CACHE_BIN}"
echo "Selected model: ${MODEL}"
echo "Model family: ${MODEL_FAMILY}"
echo "Cache output: ${CACHE_PATH}"

echo "=== Prepare model ==="

mkdir -p "${MODEL_CACHE}"

download() {
  local file="$1"
  local dest="${MODEL_CACHE}/${file}"
  if [[ -s "${dest}" ]]; then
    echo "Already cached: ${dest}"
    return
  fi
  echo "Downloading ${file} ..."
  curl -fL --progress-bar -o "${dest}.tmp" \
    "https://huggingface.co/${HF_REPO}/resolve/main/${file}"
  mv "${dest}.tmp" "${dest}"
}

if [[ "${DOWNLOAD_SOURCE}" == "huggingface" ]]; then
  download "${LLM_FILE}"
elif [[ ! -s "${MODEL_CACHE}/${LLM_FILE}" ]]; then
  echo "Error: Model artifacts are not available for automatic download." >&2
  echo "Add the required language model at: ${MODEL_CACHE}/${LLM_FILE}" >&2
  exit 1
fi

echo "=== Connect to device ==="

adb connect "${DEVICE_IP}" >/dev/null || true

SERIAL=""
if adb devices | awk '{print $1}' | grep -q "^${DEVICE_IP}:"; then
  SERIAL="$(adb devices | awk '{print $1}' | grep "^${DEVICE_IP}:" | head -n1)"
elif adb devices | awk 'NR>1 && $2=="device" {print $1}' | grep -qx "${DEVICE_IP}"; then
  SERIAL="${DEVICE_IP}"
fi

if [[ -z "${SERIAL}" ]]; then
  echo "Error: device ${DEVICE_IP} is not connected." >&2
  adb devices >&2
  exit 1
fi

echo "Using device: ${SERIAL}"
ADB=(adb -s "${SERIAL}")
REMOTE_LIB_DIR="${REMOTE_DIR}/lib"

local_size() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

sync_model() {
  local src="$1"
  local dst="$2"
  local name lsize rsize
  name="$(basename "${dst}")"
  lsize="$(local_size "${src}")"
  rsize="$("${ADB[@]}" shell "stat -c%s '${dst}' 2>/dev/null" | tr -d '\r' || true)"

  if [[ "${rsize}" == "${lsize}" ]]; then
    echo "Up to date (${lsize} bytes): ${dst}"
    return
  fi

  echo "Pushing ${name} (local ${lsize} bytes, remote ${rsize:-missing}) ..."
  "${ADB[@]}" push "${src}" "${dst}"
}

# Quote arbitrary prompt text for the device's POSIX shell.
shell_quote() {
  local value="${1//\'/\'\\\'\'}"
  printf "'%s'" "${value}"
}

echo "=== Push files to device ==="

"${ADB[@]}" shell mkdir -p "${REMOTE_MODEL_DIR}" "${REMOTE_LIB_DIR}" "$(dirname "${CACHE_PATH}")"
"${ADB[@]}" push "${CACHE_BIN}" "${REMOTE_DIR}/vlm-rknn-cache"
"${ADB[@]}" push "${ROOT_DIR}/thirdparty/rknpu2/lib-android/arm64-v8a/librknnrt.so" "${REMOTE_LIB_DIR}/"
"${ADB[@]}" push "${ROOT_DIR}/thirdparty/rkllm/lib-android/arm64-v8a/librkllmrt.so" "${REMOTE_LIB_DIR}/"
"${ADB[@]}" push "${ROOT_DIR}/thirdparty/rkllm/lib-android/arm64-v8a/libomp.so" "${REMOTE_LIB_DIR}/"
sync_model "${MODEL_CACHE}/${LLM_FILE}" "${REMOTE_MODEL_DIR}/${LLM_FILE}"
"${ADB[@]}" shell chmod 755 "${REMOTE_DIR}/vlm-rknn-cache"

echo "=== Generate prompt cache on ${SERIAL} ==="

QUOTED_PROMPT="$(shell_quote "${PROMPT}")"
QUOTED_CACHE_PATH="$(shell_quote "${CACHE_PATH}")"
"${ADB[@]}" shell -t -t "cd ${REMOTE_DIR} && LD_LIBRARY_PATH=${REMOTE_LIB_DIR} ./vlm-rknn-cache --model-family ${MODEL_FAMILY} --llm ${REMOTE_MODEL_DIR}/${LLM_FILE} --prompt ${QUOTED_PROMPT} --output ${QUOTED_CACHE_PATH} 2>&1"
