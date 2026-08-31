
"""
Convert the model corresponding to Ollama's:

    gemma3:4b-it-q8_0

to Rockchip RKLLM format.

Important:
    Ollama stores this model as a Q8_0 GGUF. RKLLM does not accept GGUF as
    input, and its W8A8/W4A16 quantization is different from GGUF Q8_0.

    Consequently this script:

        1. Finds and validates the GGUF used by Ollama.
        2. Downloads the matching original Hugging Face Gemma 3 weights.
        3. Converts those weights using rkllm-toolkit.

    This avoids:
        Q8_0 -> FP16 -> W8A8

    which would unnecessarily requantize already-lossy weights.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


OLLAMA_MODEL = "gemma3:4b-it-q8_0"
HF_MODEL = "google/gemma-3-4b-it"
HF_MODEL_URL = f"https://huggingface.co/{HF_MODEL}"
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", ".cache"))
RKLLM_CACHE_DIR = CACHE_DIR / "rkllm-toolkit"


def require_huggingface_login() -> None:
    """Stop with setup instructions when the Hugging Face CLI is logged out."""

    try:
        result = subprocess.run(
            ["hf", "auth", "whoami"],
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        result = None

    if result is not None and result.returncode == 0:
        return

    print(
        "Hugging Face authentication is required to download the original "
        "Gemma 3 weights.\n\n"
        "Log in with the Hugging Face CLI:\n\n"
        "    hf auth login\n\n"
        "You must also request access to Gemma 3 by visiting the model page "
        "and accepting Google's usage license:\n\n"
        f"    {HF_MODEL_URL}\n\n"
        "After completing both steps, run this script again.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def confirm_ollama_pull(model: str) -> bool:
    """Ask whether a missing Ollama model should be pulled."""

    while True:
        answer = input(
            f"Ollama model {model!r} is not installed. Pull it now? [y/N] "
        ).strip().lower()

        if answer in {"y", "yes"}:
            return True

        if answer in {"", "n", "no"}:
            return False

        print("Please answer yes or no.")


def ollama_show_modelfile(model: str) -> str:
    """Return Ollama's modelfile, optionally pulling a missing model first."""

    command = ["ollama", "show", "--modelfile", model]

    try:
        result = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        output = "\n".join(
            value for value in (error.stdout, error.stderr) if value
        )
        not_found = f"Error: model {model!r} not found"

        if not_found not in output or not confirm_ollama_pull(model):
            raise

        subprocess.run(["ollama", "pull", model], check=True)
        result = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    return result.stdout


def find_ollama_gguf(model: str) -> Path:
    """
    Ask Ollama which GGUF blob backs the model.

    For GGUF-backed models, `ollama show --modelfile` emits something like:

        FROM /home/user/.ollama/models/blobs/sha256-....
    """

    modelfile = ollama_show_modelfile(model)

    match = re.search(r"^FROM\s+(.+)$", modelfile, re.MULTILINE)

    if not match:
        raise RuntimeError(
            f"Could not determine GGUF blob for {model!r}.\n\n"
            f"ollama output:\n{modelfile}"
        )

    value = match.group(1).strip().strip('"')

    path = Path(value).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Ollama reported model blob:\n"
            f"    {path}\n"
            f"but that file does not exist."
        )

    return path


def validate_gguf(path: Path) -> None:
    """
    Perform a light validation of the Ollama blob.

    Requires gguf-py, which is available from llama.cpp.
    """

    try:
        from gguf import GGUFReader, GGMLQuantizationType
    except ImportError:
        print(
            "warning: Python 'gguf' package not installed; "
            "skipping detailed GGUF validation.",
            file=sys.stderr,
        )
        return

    reader = GGUFReader(str(path))

    fields = reader.fields

    architecture = None

    if "general.architecture" in fields:
        field = fields["general.architecture"]

        try:
            architecture = field.contents()
        except Exception:
            # gguf-py API has changed a little between releases.
            architecture = str(field)

    print(f"Ollama GGUF: {path}")

    if architecture is not None:
        print(f"GGUF architecture: {architecture}")

    q8_count = 0
    total = 0

    for tensor in reader.tensors:
        total += 1

        try:
            qtype = GGMLQuantizationType(tensor.tensor_type)
        except (ValueError, TypeError):
            continue

        if qtype == GGMLQuantizationType.Q8_0:
            q8_count += 1

    print(f"GGUF tensors: {total}")
    print(f"Q8_0 tensors: {q8_count}")

    if q8_count == 0:
        raise RuntimeError(
            "The Ollama model contains no Q8_0 tensors. "
            "Are you sure this is gemma3:4b-it-q8_0?"
        )


def download_hf_model(output_dir: Path) -> Path:
    """
    Download the corresponding original Gemma 3 model.

    Gemma is gated on Hugging Face, so HF_TOKEN must normally be configured
    and the Gemma license accepted on the Hugging Face account.
    """

    print(f"Downloading {HF_MODEL}...")

    model_path = snapshot_download(
        repo_id=HF_MODEL,
        local_dir=str(output_dir),
    )

    return Path(model_path)


def configure_rkllm_cache(cache_dir: Path) -> Path:
    """Redirect rkllm-toolkit's hard-coded scratch files to a writable cache."""

    import rkllm.base.opfbs as opfbs

    source_lib_dir = Path(opfbs.root_dir) / "lib"
    cache_dir = cache_dir.resolve()
    cache_lib_dir = cache_dir / "lib"
    cache_lib_dir.mkdir(parents=True, exist_ok=True)

    # opfbs loads these libraries relative to root_dir, where it also creates
    # large tmp1_*.rkllm and tmp2_*.rkllm intermediate files. Keep the
    # libraries in the image and expose them in the writable cache via links.
    for source in source_lib_dir.iterdir():
        if not source.is_file():
            continue

        link = cache_lib_dir / source.name

        if link.is_symlink() and link.resolve() == source.resolve():
            continue

        if link.exists() or link.is_symlink():
            raise RuntimeError(
                f"RKLLM cache library path already exists and is not the "
                f"expected link: {link}"
            )

        link.symlink_to(source.resolve())

    opfbs.root_dir = str(cache_dir)
    return cache_dir


def convert_rkllm(
    model_path: Path,
    output_path: Path,
    *,
    target: str,
    quantization: str,
    npu_cores: int,
    max_context: int,
    hybrid_rate: float,
    device: str,
) -> None:
    from rkllm.api import RKLLM

    rkllm_cache_dir = configure_rkllm_cache(RKLLM_CACHE_DIR)

    if quantization in {"w8a8", "w8a8_gx"}:
        algorithm = "normal"
    elif quantization in {"w4a16", "w4a16_gx"}:
        algorithm = "grq"
    else:
        raise ValueError(
            f"Unsupported quantization {quantization!r}"
        )

    print()
    print("RKLLM conversion")
    print("----------------")
    print(f"Source:       {model_path}")
    print(f"Output:       {output_path}")
    print(f"Platform:     {target}")
    print(f"Quantization: {quantization}")
    print(f"Algorithm:    {algorithm}")
    print(f"NPU cores:    {npu_cores}")
    print(f"Max context:  {max_context}")
    print(f"Hybrid rate:  {hybrid_rate}")
    print(f"Converter:    {device}")
    print(f"Scratch:      {rkllm_cache_dir}")
    print()

    llm = RKLLM()

    ret = llm.load_huggingface(
        model=str(model_path),
        device=device,
    )

    if ret != 0:
        raise RuntimeError(
            f"RKLLM load_huggingface() failed: {ret}"
        )

    ret = llm.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype=quantization,
        quantized_algorithm=algorithm,
        target_platform=target,
        num_npu_core=npu_cores,
        max_context=max_context,
        extra_qparams=None,
        dataset=None,
        hybrid_rate=hybrid_rate,
    )

    if ret != 0:
        raise RuntimeError(
            f"RKLLM build() failed: {ret}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ret = llm.export_rkllm(str(output_path))

    if ret != 0:
        raise RuntimeError(
            f"RKLLM export_rkllm() failed: {ret}"
        )

    print()
    print(f"Created: {output_path}")


def main() -> None:
    require_huggingface_login()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ollama-model",
        default=OLLAMA_MODEL,
        help=f"Ollama model to validate (default: {OLLAMA_MODEL})",
    )

    parser.add_argument(
        "--hf-dir",
        type=Path,
        default=CACHE_DIR / "gemma-3-4b-it",
        help="Directory for the original Hugging Face model",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./gemma-3-4b-it-w8a8-rk3588.rkllm"),
    )

    parser.add_argument(
        "--target",
        default="rk3588",
        choices=["rk3588", "rk3576"],
    )

    parser.add_argument(
        "--quantization",
        default="w8a8",
        choices=["w8a8", "w8a8_gx", "w4a16", "w4a16_gx"],
    )

    parser.add_argument(
        "--npu-cores",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-context",
        type=int,
        default=4096,
        help="RKLLM maximum context length",
    )

    parser.add_argument(
        "--hybrid-rate",
        type=float,
        default=0.25,
        help="Mixed block quantization ratio (default: 0.25)",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used by rkllm-toolkit during conversion",
    )

    parser.add_argument(
        "--skip-ollama-validation",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.skip_ollama_validation:
        print(f"Inspecting Ollama model {args.ollama_model!r}...")

        gguf_path = find_ollama_gguf(args.ollama_model)

        print(f"Ollama model blob: {gguf_path}")

        validate_gguf(gguf_path)

        print()

    model_path = download_hf_model(args.hf_dir)

    convert_rkllm(
        model_path=model_path,
        output_path=args.output,
        target=args.target,
        quantization=args.quantization,
        npu_cores=args.npu_cores,
        max_context=args.max_context,
        hybrid_rate=args.hybrid_rate,
        device=args.device,
    )


if __name__ == "__main__":
    main()
