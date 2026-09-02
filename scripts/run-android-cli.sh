#!/usr/bin/env bash
set -euo pipefail

# Push the Android build of vlm-rknn to a device and start its interactive
# prompt.
#
# Usage:
#   ./scripts/run-android-cli.sh <device-ip> <model> [remote-path]
#
# Arguments:
#   device-ip    IP address (or host) of the target device, as used by `adb connect`.
#   model        Model to run: qwen2-vl-2b, qwen2-vl-7b, or gemma3-4b.
#   remote-path  Directory to push files to on the device (default /data/local/tmp).
#
# Environment:
#   IMAGE_PATH   Image to use for interactive prompts (default data/cell.png).
#   RKLLM_CACHE  Optional RKLLM prompt cache to load at startup
#                (example: <remote-path>/models/<model>/prompt.cache).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-android}"
CACHE_DIR="${ROOT_DIR}/.cache"
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/data/cell.png}"

# Set the download and runtime parameters for a supported model. Gemma3
# artifacts are cache-only because compatible converted files are not currently
# available from Hugging Face.
load_model_config() {
  case "$1" in
    qwen2-vl-2b)
      MODEL_FAMILY="qwen2-vl"
      HF_REPO="3ib0n/Qwen2-VL-2B-rkllm"
      LLM_FILE="Qwen2-VL-2B-Instruct.rkllm"
      VISION_FILE="qwen2_vl_2b_vision_rk3588.rknn"
      CACHE_NAME="qwen2-vl-2b"
      DOWNLOAD_SOURCE="huggingface"
      ;;
    qwen2-vl-7b)
      MODEL_FAMILY="qwen2-vl"
      HF_REPO="3ib0n/Qwen2-VL-7B-rkllm"
      LLM_FILE="Qwen2-VL-7B-Instruct.rkllm"
      VISION_FILE="qwen2_vl_7b_vision_rk3588.rknn"
      CACHE_NAME="qwen2-vl-7b"
      DOWNLOAD_SOURCE="huggingface"
      ;;
    gemma3-4b)
      MODEL_FAMILY="gemma3"
      HF_REPO=""
      LLM_FILE="gemma3-4b-it.rkllm"
      VISION_FILE="gemma3-vision-projector.rknn"
      TOKENIZER_FILE="tokenizer.model"
      CACHE_NAME="gemma3-4b"
      DOWNLOAD_SOURCE="cache"
      ;;
    *)
      return 1
      ;;
  esac
}

DEVICE_IP="${1:-}"
MODEL="${2:-}"
REMOTE_DIR="${3:-/data/local/tmp}"
REMOTE_MODEL_DIR="${REMOTE_DIR}/models/${MODEL}"
RKLLM_CACHE="${RKLLM_CACHE:-${REMOTE_MODEL_DIR}/prompt.cache}"

if [[ -z "${DEVICE_IP}" || -z "${MODEL}" ]]; then
  cat <<USAGE
Usage: $0 <device-ip> <model> [remote-path]

Arguments:
  device-ip    IP address (or host) of the target device, as used by adb connect.
  model        Model to run: qwen2-vl-2b, qwen2-vl-7b, or gemma3-4b.
  remote-path  Directory to push files to on the device (default /data/local/tmp).

Environment:
  IMAGE_PATH   Image to use for interactive prompts (default data/cell.png).
  RKLLM_CACHE  Optional RKLLM prompt cache path on the ADB device
               (default <remote-path>/models/<model>/prompt.cache).
USAGE
  exit 1
fi

if ! load_model_config "${MODEL}"; then
  echo "Error: unsupported model '${MODEL}'." >&2
  echo "Expected one of: qwen2-vl-2b, qwen2-vl-7b, gemma3-4b." >&2
  exit 1
fi

MODEL_CACHE="${CACHE_DIR}/${CACHE_NAME}"

# -----------------------------------------------------------------------------
# Preconditions
# -----------------------------------------------------------------------------

echo "=== Check preconditions ==="

CLI_BIN="${BUILD_DIR}/vlm-rknn"
if [[ ! -x "${CLI_BIN}" ]]; then
  echo "Error: ${CLI_BIN} not found." >&2
  echo "Run the Android build first: ./scripts/build-android.sh docker" >&2
  exit 1
fi

echo "CLI binary: ${CLI_BIN}"

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

echo "All preconditions satisfied."

# -----------------------------------------------------------------------------
# Model selection
# -----------------------------------------------------------------------------

echo "=== Model selection ==="

echo "Selected model: ${MODEL}"
echo "Model family: ${MODEL_FAMILY}"
echo "LLM file: ${LLM_FILE}"
echo "Vision file: ${VISION_FILE}"

# -----------------------------------------------------------------------------
# Download models
# -----------------------------------------------------------------------------

echo "=== Download models ==="

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
  download "${VISION_FILE}"
  USE_VISION=1
elif [[ ! -s "${MODEL_CACHE}/${LLM_FILE}" ]]; then
  echo "Error: Model artifacts are not available for automatic download." >&2
  echo "Add the required language model to the cache directory at:" >&2
  echo "  LLM: ${MODEL_CACHE}/${LLM_FILE}" >&2
  echo "The optional vision projector may be added at:" >&2
  echo "  Vision: ${MODEL_CACHE}/${VISION_FILE}" >&2
  exit 1
elif [[ -n "${VISION_FILE}" && -s "${MODEL_CACHE}/${VISION_FILE}" ]]; then
  USE_VISION=1
else
  USE_VISION=0
  echo "Optional vision projector not found; starting a text-only session."
  echo "To enable image input, add it at: ${MODEL_CACHE}/${VISION_FILE}"
fi

if [[ -n "${TOKENIZER_FILE:-}" && -s "${MODEL_CACHE}/${TOKENIZER_FILE}" ]]; then
  USE_TOKENIZER=1
  echo "Tokenizer model: ${MODEL_CACHE}/${TOKENIZER_FILE}"
else
  USE_TOKENIZER=0
fi

if [[ "${USE_VISION}" -eq 1 ]]; then
  if [[ ! -f "${IMAGE_PATH}" ]]; then
    echo "Error: image not found: ${IMAGE_PATH}" >&2
    exit 1
  fi
  echo "Image: ${IMAGE_PATH}"
fi

# -----------------------------------------------------------------------------
# ADB connect
# -----------------------------------------------------------------------------

echo "=== Connect to device ==="

adb connect "${DEVICE_IP}" >/dev/null || true

# Resolve the serial: prefer an explicit ip:port match, fall back to the device.
SERIAL=""
if adb devices | awk '{print $1}' | grep -q "^${DEVICE_IP}:"; then
  SERIAL="$(adb devices | awk '{print $1}' | grep "^${DEVICE_IP}:" | head -n1)"
elif adb devices | awk 'NR>1 && $2=="device" {print $1}' | grep -qx "${DEVICE_IP}"; then
  SERIAL="${DEVICE_IP}"
fi

if [[ -z "${SERIAL}" ]]; then
  echo "Error: device ${DEVICE_IP} is not connected." >&2
  echo "Connected devices:" >&2
  adb devices >&2
  exit 1
fi

echo "Using device: ${SERIAL}"
ADB=(adb -s "${SERIAL}")

# -----------------------------------------------------------------------------
# Remote paths
# -----------------------------------------------------------------------------

REMOTE_MODEL_DIR="${REMOTE_DIR}/models/${MODEL}"
REMOTE_LIB_DIR="${REMOTE_DIR}/lib"
REMOTE_IMAGE="${REMOTE_DIR}/input-image"

# -----------------------------------------------------------------------------
# Sync helpers
# -----------------------------------------------------------------------------

# Local file size in bytes (portable across macOS and Linux).
local_size() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

# Push a model file only if the device copy is missing or a different size.
# Large model files are expensive to transfer, so we compare byte size, which
# is instant on both ends and a strong signal for these immutable files.
sync_model() {
  local src="$1"
  local dst="$2"
  local name lsize rsize
  name="$(basename "${dst}")"

  lsize="$(local_size "${src}")"
  # A missing remote file makes stat exit non-zero; tolerate it (rsize stays
  # empty) so set -o pipefail / set -e don't abort the script on first push.
  rsize="$("${ADB[@]}" shell "stat -c%s '${dst}' 2>/dev/null" | tr -d '\r' || true)"

  if [[ "${rsize}" == "${lsize}" ]]; then
    echo "Up to date (${lsize} bytes): ${dst}"
    return
  fi

  echo "Pushing ${name} (local ${lsize} bytes, remote ${rsize:-missing}) ..."
  "${ADB[@]}" push "${src}" "${dst}"
}

# -----------------------------------------------------------------------------
# Push files
# -----------------------------------------------------------------------------

echo "=== Push files to device ==="

"${ADB[@]}" shell mkdir -p "${REMOTE_MODEL_DIR}" "${REMOTE_LIB_DIR}"

"${ADB[@]}" push "${CLI_BIN}" "${REMOTE_DIR}/vlm-rknn"
if [[ "${USE_VISION}" -eq 1 ]]; then
  "${ADB[@]}" push "${IMAGE_PATH}" "${REMOTE_IMAGE}"
fi

"${ADB[@]}" push "${ROOT_DIR}/thirdparty/rknpu2/lib-android/arm64-v8a/librknnrt.so" "${REMOTE_LIB_DIR}/"
"${ADB[@]}" push "${ROOT_DIR}/thirdparty/rkllm/lib-android/arm64-v8a/librkllmrt.so" "${REMOTE_LIB_DIR}/"
"${ADB[@]}" push "${ROOT_DIR}/thirdparty/rkllm/lib-android/arm64-v8a/libomp.so" "${REMOTE_LIB_DIR}/"

# Model files are large; only push when missing or changed on the device.
sync_model "${MODEL_CACHE}/${LLM_FILE}" "${REMOTE_MODEL_DIR}/${LLM_FILE}"
if [[ "${USE_VISION}" -eq 1 ]]; then
  sync_model "${MODEL_CACHE}/${VISION_FILE}" "${REMOTE_MODEL_DIR}/${VISION_FILE}"
fi
if [[ "${USE_TOKENIZER}" -eq 1 ]]; then
  sync_model "${MODEL_CACHE}/${TOKENIZER_FILE}" "${REMOTE_MODEL_DIR}/${TOKENIZER_FILE}"
fi

"${ADB[@]}" shell chmod 755 "${REMOTE_DIR}/vlm-rknn"

# -----------------------------------------------------------------------------
# Start the interactive CLI
# -----------------------------------------------------------------------------

echo "=== Start interactive CLI on ${SERIAL} ==="

# Omitting --prompt starts vlm-rknn's interactive REPL. Force a pseudo-terminal
# (-t -t) so prompts and generated output stream live. Merge stderr into stdout
# so both are shown.
VISION_ARGS=""
if [[ "${USE_VISION}" -eq 1 ]]; then
  VISION_ARGS=" --vision ${REMOTE_MODEL_DIR}/${VISION_FILE} --image ${REMOTE_IMAGE}"
fi
TOKENIZER_ARGS=""
if [[ "${USE_TOKENIZER}" -eq 1 ]]; then
  TOKENIZER_ARGS=" --tokenizer-model ${REMOTE_MODEL_DIR}/${TOKENIZER_FILE}"
fi
CACHE_ARGS=""
if "${ADB[@]}" shell "test -f '${RKLLM_CACHE}'"; then
  echo "Prompt cache found: ${RKLLM_CACHE}"
  CACHE_ARGS=" --cache ${RKLLM_CACHE}"
else
  echo "No cache file found at ${RKLLM_CACHE}; starting without a prompt cache."
fi
"${ADB[@]}" shell -t -t "cd ${REMOTE_DIR} && LD_LIBRARY_PATH=${REMOTE_LIB_DIR} ./vlm-rknn --model-family ${MODEL_FAMILY} --llm ${REMOTE_MODEL_DIR}/${LLM_FILE}${VISION_ARGS}${TOKENIZER_ARGS}${CACHE_ARGS} 2>&1"
