# Plan

- Build and tooling
  + ~~Project scaffolding and CMake layout~~
  + ~~Wire up RKNN and RKLLM libraries from `thirdparty/`~~
  + ~~Add OpenCV via FetchContent (trimmed module set)~~
  + ~~Docker images and helper scripts for native and Android builds~~
  + ~~GitHub Actions for ShellCheck~~
  + ~~Strip symbols from release builds~~
  + ~~Add a CI job that runs the native build end-to-end~~
  + Publish prebuilt binaries from tagged releases

- Vision encoder
  + ~~Load RKNN model from disk~~
  + ~~Honour `--cores` and set the NPU core mask~~
  + ~~Query model input/output info and populate `VisionEncoder`~~
  + ~~Free `input_attrs` / `output_attrs` in `cleanup_vision_encoder`~~
  + ~~Implement `Session::encode()` with `rknn_inputs_set`, `rknn_run`, `rknn_outputs_get`, and `rknn_outputs_release`~~
  + Validate tensor layout assumptions for RKNN model metadata (`NCHW` vs `NHWC`, output shape ordering)

- Image preprocessing
  + ~~Load image with OpenCV, BGR → RGB~~
  + ~~Pad to square with neutral background (`expand2square`)~~
  + ~~Resize to model input dimensions~~
  + Confirm preprocessing matches the Qwen-VL reference (mean/std normalisation, pad colour)
  + Handle image sources other than files on disk

- Text decoder (RKLLM)
  + ~~Configure `RKLLMParam` with Qwen-VL `<|vision_start|>` / `<|vision_end|>` / `<|image_pad|>` tokens~~
  + ~~Initialise the RKLLM handle and register the streaming callback~~
  + ~~Invoke `rkllm_run` with encoded image vector~~
  + ~~Update debug logging to use existing logger rather than `printf`~~
  + Handle `rkllm_clear_kv_cache` for a `clear` REPL command

- CLI and UX
  + ~~Argument parsing for `--verbose`, `--cores`, etc~~
  + ~~Run the supplied prompt once and exit when `[prompt]` is provided~~
  + ~~Drop into an interactive REPL when no prompt is given~~
  + ~~Expose `--max-new-tokens` and `--max-context-len` flags that flow through to `ModelConfig`~~
  + ~~Optional Android frontend~~
    + ~~Allow models to be switched dynamically~~
  + Install a `SIGINT` handler that destroys the RKLLM handle cleanly

- Tests
  + ~~Fix the broken reference to `qwen_vl_rknn::target_device()`~~
  + ~~Replace the placeholder readiness check for `is_ready()` with checks that do not require real RKNN/RKLLM model files~~
  + ~~Add a host-side preprocessing unit test covering square, portrait, and landscape inputs~~
  + ~~Remove the dead `expand2square` helper~~
  + Run the host tests in CI
  + Add a smoke test that runs the binary against a tiny fixture model

- Conversion
  + Bring in scripts for model conversion

- On-device verification
  + Source or convert a Qwen-VL vision encoder (`.rknn`) and language model (`.rkllm`) suitable for RK3588
  + Document the conversion recipe
  + Capture a sample run (image + prompt + transcript) in the README
  + Measure load and per-token latency for the default model and record baseline numbers

- Documentation
  + ~~Top-level README with build instructions and project layout~~
  + ~~AGENTS.md describing the project for assistants~~
  + Add a "Running" section to the README with concrete CLI examples once `Session::encode()` is functional
  + Document the expected `thirdparty/` layout and where to download the runtime libraries
  + Document SmolVLM2 model source in the README "Models" section (currently only Qwen2-VL is covered)

- R&D
  + Investigate changes required to support Qwen3-VL (**in progress**)
  + Investigate changes required to support Llama 3.2 Vision models (**in progress**)
  + Design RAII wrappers for RKNN primitives

- ~~Cleanup~~
  + ~~Fix CI artifact paths and `build-native.sh`'s undefined `-DVLM_RKNN_ENABLE_RKNN` flag~~
  + ~~Decide whether the unconditional vision warm-up decode in `main.cc` should be silenced or removed~~
  + ~~Map `--cores 1` to a single core~~
