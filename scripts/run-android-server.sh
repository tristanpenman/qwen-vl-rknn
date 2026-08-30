#!/usr/bin/env bash
set -euo pipefail

# Push the Android build of vlm-rknn-server to a device and start it with two
# models configured: Qwen2-VL (the default, loaded eagerly at startup) and
# SmolVLM2-256M (a second, non-default model loaded on demand).
#
# The Qwen2-VL model files come from:
#   https://huggingface.co/3ib0n/Qwen2-VL-2B-rkllm
#
# The SmolVLM2-256M model files come from the Qengineering project:
#   https://github.com/Qengineering/SmolVLM2-256M-NPU
#
# Usage:
#   ./scripts/run-android-server.sh <device-ip> [remote-path]
#
# Arguments:
#   device-ip    IP address (or host) of the target device, as used by `adb connect`.
#   remote-path  Directory to push files to on the device (default /data/local/tmp).
#
# Environment:
#   MODEL_SIZE   Qwen2-VL model to download and run as default: 2b (default) or 7b.
#   PORT         Port the server should listen on (default 8080).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-android}"
CACHE_DIR="${ROOT_DIR}/.cache"
MODEL_SIZE="${MODEL_SIZE:-2b}"
PORT="${PORT:-8080}"

DEVICE_IP="${1:-}"
REMOTE_DIR="${2:-/data/local/tmp}"

if [[ -z "${DEVICE_IP}" ]]; then
  cat <<USAGE
Usage: $0 <device-ip> [remote-path]

Arguments:
  device-ip    IP address (or host) of the target device, as used by adb connect.
  remote-path  Directory to push files to on the device (default /data/local/tmp).

Environment:
  MODEL_SIZE   Qwen2-VL model to download and run as default: 2b (default) or 7b.
  PORT         Port the server should listen on (default 8080).
USAGE
  exit 1
fi

# -----------------------------------------------------------------------------
# Preconditions
# -----------------------------------------------------------------------------

echo "=== Check preconditions ==="

SERVER_BIN="${BUILD_DIR}/vlm-rknn-server"
if [[ ! -x "${SERVER_BIN}" ]]; then
  echo "Error: ${SERVER_BIN} not found." >&2
  echo "Run the Android build first: ./scripts/build-android.sh docker" >&2
  exit 1
fi

echo "Server binary: ${SERVER_BIN}"

for tool in adb curl; do
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

# Default model: Qwen2-VL (loaded eagerly at startup).
case "${MODEL_SIZE}" in
  2b)
    QWEN_HF_REPO="3ib0n/Qwen2-VL-2B-rkllm"
    QWEN_LLM_FILE="Qwen2-VL-2B-Instruct.rkllm"
    QWEN_VISION_FILE="qwen2_vl_2b_vision_rk3588.rknn"
    ;;
  7b)
    QWEN_HF_REPO="3ib0n/Qwen2-VL-7B-rkllm"
    QWEN_LLM_FILE="Qwen2-VL-7B-Instruct.rkllm"
    QWEN_VISION_FILE="qwen2_vl_7b_vision_rk3588.rknn"
    ;;
  *)
    echo "Error: unsupported MODEL_SIZE '${MODEL_SIZE}' (expected 2b or 7b)." >&2
    exit 1
    ;;
esac

QWEN_HF_BASE="https://huggingface.co/${QWEN_HF_REPO}/resolve/main"
QWEN_CACHE="${CACHE_DIR}/qwen2-vl-${MODEL_SIZE}"

# Second, non-default model: SmolVLM2-256M from Qengineering/SmolVLM2-256M-NPU.
SMOLVLM_HF_REPO="Qengineering/SmolVLM2-256m-rk3588"
SMOLVLM_LLM_FILE="smolvlm2-256m-instruct_w8a8_rk3588.rkllm"
SMOLVLM_VISION_FILE="smolvlm2_256m_vision_fp16_rk3588.rknn"
SMOLVLM_HF_BASE="https://huggingface.co/${SMOLVLM_HF_REPO}/resolve/main"
SMOLVLM_CACHE="${CACHE_DIR}/smolvlm2-256m"

echo "Default model:     qwen2-vl (${MODEL_SIZE})"
echo "  LLM file:    ${QWEN_LLM_FILE}"
echo "  Vision file: ${QWEN_VISION_FILE}"
echo "Second model:      smolvlm2 (256M, non-default, loaded on demand)"
echo "  LLM file:    ${SMOLVLM_LLM_FILE}"
echo "  Vision file: ${SMOLVLM_VISION_FILE}"

# -----------------------------------------------------------------------------
# Download models
# -----------------------------------------------------------------------------

echo "=== Download models ==="

mkdir -p "${QWEN_CACHE}" "${SMOLVLM_CACHE}"

# Download a file from a Hugging Face base URL into a cache directory, skipping
# the transfer when a non-empty copy already exists.
download() {
  local base="$1"
  local dir="$2"
  local file="$3"
  local dest="${dir}/${file}"
  if [[ -s "${dest}" ]]; then
    echo "Already cached: ${dest}"
    return
  fi
  echo "Downloading ${file} ..."
  curl -fL --progress-bar -o "${dest}.tmp" "${base}/${file}"
  mv "${dest}.tmp" "${dest}"
}

download "${QWEN_HF_BASE}" "${QWEN_CACHE}" "${QWEN_LLM_FILE}"
download "${QWEN_HF_BASE}" "${QWEN_CACHE}" "${QWEN_VISION_FILE}"
download "${SMOLVLM_HF_BASE}" "${SMOLVLM_CACHE}" "${SMOLVLM_LLM_FILE}"
download "${SMOLVLM_HF_BASE}" "${SMOLVLM_CACHE}" "${SMOLVLM_VISION_FILE}"

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
# Generate server configuration
# -----------------------------------------------------------------------------

echo "=== Generate server configuration ==="

REMOTE_QWEN_DIR="${REMOTE_DIR}/models/qwen2-vl"
REMOTE_SMOLVLM_DIR="${REMOTE_DIR}/models/smolvlm2"
REMOTE_LIB_DIR="${REMOTE_DIR}/lib"

# The first section is the default model loaded eagerly at startup; subsequent
# sections (smolvlm2 here) are loaded on demand when first requested.
INI_FILE="${CACHE_DIR}/server.android.multi.ini"
cat > "${INI_FILE}" <<INI
[qwen2-vl]
model_family=qwen2-vl
vision=${REMOTE_QWEN_DIR}/${QWEN_VISION_FILE}
llm=${REMOTE_QWEN_DIR}/${QWEN_LLM_FILE}
max_new_tokens=300

[smolvlm2]
model_family=smolvlm2
vision=${REMOTE_SMOLVLM_DIR}/${SMOLVLM_VISION_FILE}
llm=${REMOTE_SMOLVLM_DIR}/${SMOLVLM_LLM_FILE}
max_new_tokens=300
INI

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

"${ADB[@]}" shell mkdir -p "${REMOTE_QWEN_DIR}" "${REMOTE_SMOLVLM_DIR}" "${REMOTE_LIB_DIR}"

"${ADB[@]}" push "${SERVER_BIN}" "${REMOTE_DIR}/vlm-rknn-server"
"${ADB[@]}" push "${INI_FILE}" "${REMOTE_DIR}/server.ini"

"${ADB[@]}" push "${ROOT_DIR}/thirdparty/rknpu2/lib-android/arm64-v8a/librknnrt.so" "${REMOTE_LIB_DIR}/"
"${ADB[@]}" push "${ROOT_DIR}/thirdparty/rkllm/lib-android/arm64-v8a/librkllmrt.so" "${REMOTE_LIB_DIR}/"
"${ADB[@]}" push "${ROOT_DIR}/thirdparty/rkllm/lib-android/arm64-v8a/libomp.so" "${REMOTE_LIB_DIR}/"

# Model files are large; only push when missing or changed on the device.
sync_model "${QWEN_CACHE}/${QWEN_LLM_FILE}" "${REMOTE_QWEN_DIR}/${QWEN_LLM_FILE}"
sync_model "${QWEN_CACHE}/${QWEN_VISION_FILE}" "${REMOTE_QWEN_DIR}/${QWEN_VISION_FILE}"
sync_model "${SMOLVLM_CACHE}/${SMOLVLM_LLM_FILE}" "${REMOTE_SMOLVLM_DIR}/${SMOLVLM_LLM_FILE}"
sync_model "${SMOLVLM_CACHE}/${SMOLVLM_VISION_FILE}" "${REMOTE_SMOLVLM_DIR}/${SMOLVLM_VISION_FILE}"

"${ADB[@]}" shell chmod 755 "${REMOTE_DIR}/vlm-rknn-server"

# -----------------------------------------------------------------------------
# Start the server
# -----------------------------------------------------------------------------

echo "=== Start server on ${SERIAL} (port ${PORT}) ==="
echo "Models: qwen2-vl (default), smolvlm2 (on demand)"

# Force a pseudo-terminal (-t -t) so the server's stdout is line-buffered and
# streams live; without a tty it stays block-buffered and nothing appears until
# the process exits. Merge stderr into stdout so both are shown.
"${ADB[@]}" shell -t -t "cd ${REMOTE_DIR} && LD_LIBRARY_PATH=${REMOTE_LIB_DIR} ./vlm-rknn-server --ini-file ${REMOTE_DIR}/server.ini --port ${PORT} 2>&1"
