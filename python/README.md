# Python

The `vlm_rknn` Python package contains utilities for preparing models for use with the C++ RKNN/RKLLM runtime. The current conversion utility converts the model corresponding to Ollama's `gemma3:4b-it-q8_0` into Rockchip RKLLM format.

## How it works

Ollama stores `gemma3:4b-it-q8_0` as a Q8_0 GGUF model, but RKLLM Toolkit does not accept GGUF input. Its W8A8 and W4A16 formats also use different quantization schemes from GGUF Q8_0.

The converter therefore does not dequantize and requantize Ollama's lossy weights. Instead, it:

1. Asks Ollama which GGUF blob backs `gemma3:4b-it-q8_0`.
2. Validates that the blob contains Q8_0 tensors.
3. Downloads the matching original `google/gemma-3-4b-it` weights from Hugging Face.
4. Loads the original weights with RKLLM Toolkit.
5. Quantizes and exports them for the selected Rockchip target.

This avoids an unnecessary Q8_0 to FP16 to W8A8 conversion.

## Prerequisites

Gemma 3 model has restricted access on Hugging Face. Request access to [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) and accept Google's usage license.

From the repository root, build the CPU conversion environment, start its Ollama server, and open a shell with:

```bash
./scripts/python-shell.sh
```

For CUDA conversion, use the GPU environment:

```bash
./scripts/python-shell-gpu.sh
```

The GPU service requires an NVIDIA driver and NVIDIA Container Toolkit on the host. From the GPU container, verify CUDA access with:

```bash
python -c \
  'import torch; print(torch.version.cuda); print(torch.cuda.is_available())'
```

Inside the selected container, authenticate with the Hugging Face CLI before running the converter:

```bash
hf auth login
```

The containers bind-mount the repository at `/workspace`. Ollama models, Hugging Face downloads, and RKLLM scratch files are stored beneath `/workspace/.cache`, which corresponds to the repository's `.cache` directory.

## RKLLM Toolkit versions

Two RKLLM Toolkit wheels are included in [`python/wheels`](./wheels):

- `rkllm_toolkit-1.3.0-cp311-cp311-linux_x86_64.whl`
- `rkllm_toolkit-1.2.3-cp311-cp311-linux_x86_64.whl`

The Python images install version 1.3.0 by default. To use version 1.2.3 instead, edit [`requirements.txt`](./requirements.txt) so its RKLLM Toolkit entry points to `/opt/wheels/rkllm_toolkit-1.2.3-cp311-cp311-linux_x86_64.whl`, then rebuild the selected Python image.

## Usage

Run the converter from either Python container:

```bash
python -m vlm_rknn.convert_gemma3
```

The default invocation validates the installed Ollama model, downloads the original Hugging Face weights to `.cache/gemma-3-4b-it`, and creates `gemma-3-4b-it-w8a8-rk3588.rkllm` in the current directory. If `XDG_CACHE_HOME` is set, it is used instead of `.cache`.

The converter prompts to pull the Ollama model if it is not already installed. Use `--skip-ollama-validation` to skip the Ollama lookup and GGUF validation. This option does not skip the Hugging Face download or RKLLM conversion.

Relative paths are resolved from the current directory, and missing output directories are created automatically.

## Options

| Option                     | Default                                  | Description                                       |
|----------------------------|------------------------------------------|---------------------------------------------------|
| `--ollama-model`           | `gemma3:4b-it-q8_0`                      | Ollama model to validate                          |
| `--hf-dir`                 | `.cache/gemma-3-4b-it`                   | Hugging Face model directory                      |
| `--output` or `-o`         | `gemma-3-4b-it-w8a8-rk3588.rkllm`        | Destination RKLLM file                            |
| `--target`                 | `rk3588`                                 | Target platform: `rk3588` or `rk3576`             |
| `--quantization`           | `w8a8`                                   | `w8a8`, `w8a8_gx`, `w4a16`, or `w4a16_gx`         |
| `--npu-cores`              | `3`                                      | Number of target NPU cores                        |
| `--max-context`            | `4096`                                   | Maximum context length                            |
| `--hybrid-rate`            | `0.25`                                   | Mixed-block quantization ratio                    |
| `--device`                 | `cpu`                                    | Conversion device: `cpu` or `cuda`                |
| `--skip-ollama-validation` | disabled                                 | Skip Ollama GGUF validation                       |

W8A8 variants use RKLLM Toolkit's `normal` algorithm. W4A16 variants use group-wise refined quantization (`grq`), which divides weights into small groups to reduce sensitivity to outliers.

For example, this command creates a W4A16 model for RK3576 with an 8192-token context using CUDA:

```bash
python -m vlm_rknn.convert_gemma3 \
  --target rk3576 \
  --quantization w4a16 \
  --max-context 8192 \
  --device cuda \
  --output ./models/gemma-3-4b-it-w4a16-rk3576.rkllm
```

Use `--device cuda` only in the GPU container with working CUDA access. Model weights and generated RKLLM files remain subject to their respective model and toolkit license terms.
