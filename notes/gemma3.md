# Gemma 3

This document describes the work required to add image-and-text inference for the vision-capable Gemma 3 models. It targets the Gemma 3 4B, 12B, and 27B variants. It excludes the smaller Gemma 3 1B model, since it is text-only.

The first target is `google/gemma-3-4b-it` with one image per request and pan-and-scan disabled. Gemma 3 normally converts one 896 x 896 image into 256 language-model embeddings, which fits the repository's existing single-image multimodal interface. Google documents the input resolution and token count in the [Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3).

Pan-and-scan, multiple images, and Gemma 3n should be treated as later work. Gemma 3n uses a different vision encoder and model architecture.

## Architectural Fit

Gemma 3 uses a data flow that is similar to the models already supported by this project:

```text
image
  -> SigLIP vision tower
  -> Gemma multimodal projector
  -> 256 embeddings in the text model's hidden dimension
  -> replace 256 <image_soft_token> positions in the prompt
  -> normal Gemma decoder self-attention
```

In the Hugging Face implementation, image features are computed and placed into `inputs_embeds` at image-placeholder positions before calling the language model. See the [Transformers Gemma 3 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modular_gemma3.py).

This is conceptually similar to the existing Qwen-VL and SmolVLM path:

```text
image
  -> RKNN vision model
  -> flat image-embedding buffer
  -> RKLLM_INPUT_MULTIMODAL
  -> RKLLM decoder
```

Gemma 3 therefore does not need `RKLLMCrossAttnParam`, `rkllm_set_cross_attn_params`, or access to RKLLM's decoder loop. This was a concern when evaluating Llama 3.2 variants.

## Required Model Artifacts

Support depends on two compatible model files produced from the same Gemma 3 checkpoint and conversion recipe:

1. A vision-plus-projector `.rknn` model for RK3588.
2. A Gemma 3 text decoder `.rkllm` model for RK3588.

### RKNN Artifact Contract

For the initial single-image implementation, the preferred contract is:

```text
input:
    one 896 x 896 RGB image

output:
    [1, 256, text_hidden_size]
```

The RKNN graph should include both the SigLIP vision tower and `Gemma3MultiModalProjector`. Exporting only the raw SigLIP hidden states will not work because their dimension and normalization do not match the text decoder's placeholder embeddings.

The projector includes spatial pooling, normalization, and projection into the Gemma text hidden size. Its behavior must be reproduced exactly by the exported model.

The conversion process must define whether the RKNN model accepts uint8 pixels with model-side normalization or normalized floating-point pixels. The current C++ path always submits a uint8 NHWC input with `pass_through = 0` in [`cpp/src/vlm_rknn.cc`](cpp/src/vlm_rknn.cc). Baking rescaling and normalization into the RKNN conversion is the smallest integration change. If the exported model requires float input, the C++ input abstraction must be extended instead.

### RKLLM Artifact Contract

The decoder must be converted as Gemma 3 rather than as a generic Gemma or Gemma 2 model. It must preserve:

- Gemma 3's alternating local sliding-window and global attention layers.
- The tokenizer and special-token IDs from the selected checkpoint.
- The image-placeholder embedding interface.
- The special prefill attention behavior for the block of image tokens.
- The instruction-tuned chat template when using an `-it` checkpoint.

Rockchip's RKLLM project lists Gemma 3 as a supported model family. Use a Toolkit/runtime pair known to support the converted artifact and keep the Toolkit version no newer than the runtime accepted by the device. The vendored header and libraries in this repository should be tested with a known-good Gemma 3 model before application changes are debugged.

The important Gemma tokenizer strings are normally:

```text
image start:   <start_of_image>
image end:     <end_of_image>
image content: <image_soft_token>
```

Confirm these values against the exact checkpoint's tokenizer files instead of relying only on the names above.

## Implementation

### Prove the Artifacts Independently

Before changing this repository, validate both converted artifacts with the vendor's reference tools.

For the RKNN artifact:

1. Run the Hugging Face vision tower and multimodal projector for a fixed image.
2. Run the exported RKNN model for the same preprocessed image.
3. Confirm that both produce 256 vectors in the decoder hidden dimension.
4. Compare tensor ordering and numerical error after accounting for quantization.
5. Test at least square, landscape, and portrait images.

For the RKLLM artifact:

1. Run a text-only prompt to verify tokenizer and chat-template behavior.
2. Run a known-good projected image embedding through Rockchip's multimodal demo.
3. Confirm that image-placeholder expansion consumes exactly 256 embeddings.
4. Record the required `base_domain_id`, image marker strings, chat template, maximum context length, and runtime version.

If the decoder cannot consume externally supplied projected embeddings through `RKLLM_INPUT_MULTIMODAL`, the blocker is the conversion/runtime combination rather than this repository.

### Register Gemma 3 as a Model Family

Add `kGemma3` to `ModelFamily` in [`cpp/src/vlm_rknn.h`](cpp/src/vlm_rknn.h).

Update the model-family functions in [`cpp/src/vlm_rknn.cc`](cpp/src/vlm_rknn.cc):

- Accept `gemma3`, `gemma-3`, and optionally `gemma_3` in `parseModelFamily`.
- Return a canonical name such as `gemma3` from `modelFamilyName`.
- Add the family to every exhaustive `switch`.
- Register it as requiring a vision encoder and supporting multimodal input.

Add a `ModelProfile` with values verified from the converted artifact. A likely starting point is:

```cpp
static constexpr ModelProfile kGemma3 {
    true,                   // usesVisionEncoder
    true,                   // supportsMultimodal
    verifiedBaseDomainId,
    true,                   // useChatTemplate, for an instruction model
    "<image>",              // repository-facing placeholder
    "<start_of_image>",
    "<end_of_image>",
    "<image_soft_token>",
    kGemma3ImagePreprocess,
};
```

Do not guess `base_domain_id`. Obtain it from the Rockchip conversion instructions or a working reference demo.

The repository-facing placeholder can remain `<image>`. RKLLM uses `img_start`, `img_end`, and `img_content` to translate the generic image location into the decoder's model-specific token sequence.

### Implement Gemma Image Preprocessing

Use the processor saved with the selected checkpoint as the reference implementation. The host and RKNN conversion together must reproduce its:

- RGB channel order.
- Resize and resampling behavior.
- 896 x 896 input shape.
- Pixel rescaling and normalization.
- Image padding behavior, if any.

Do not assume that the existing Qwen square-padding profile is correct for Gemma. Add the required resize behavior to `ResizeMode`, or add a dedicated Gemma preprocessing function if the processor cannot be represented accurately by the existing three modes.

There are two viable input strategies:

1. Keep the current uint8 NHWC C++ input and configure RKNN to perform rescaling and normalization.
2. Extend `VisionEncoder` and `Session::encode` to use the tensor type and layout reported by `rknn_tensor_attr`, allowing a normalized float input.

The first strategy is simpler, but the numerical comparison against the Hugging Face reference must demonstrate that it is correct.

Add unit tests for color conversion, resizing, padding or cropping, and output continuity. Test non-square images so an accidental stretch or crop is detected.

### Enforce the Vision Output Contract

The current implementation infers image token count and embedding size by selecting the first pair of adjacent dimensions greater than one. It also treats multiple RKNN outputs as equivalent embedding streams and interleaves them.

For Gemma 3, validate explicitly that:

- The vision model has the expected input count.
- Its output represents exactly 256 image tokens when pan-and-scan is disabled.
- Its final dimension equals the Gemma decoder hidden size.
- The output data type can be retrieved correctly as float embeddings.
- The tensor is ordered as token-major contiguous embeddings expected by RKLLM.
- Unexpected additional outputs are rejected rather than silently interleaved.

Prefer storing the expected token count and embedding size in model metadata or configuration and checking the queried RKNN attributes against them. The generic dimension heuristic is useful for existing artifacts but should not be the sole validation for a new model family.

### Configure Prompt and Decoder Behavior

Continue using `RKLLM_INPUT_MULTIMODAL` in `Session::decode`. For the first implementation:

```cpp
rkllmInput.multimodal_input.image_embed = imageEmbedding;
rkllmInput.multimodal_input.n_image_tokens = 256;
rkllmInput.multimodal_input.n_image = 1;
```

The prompt must contain exactly one repository-facing `<image>` placeholder when one image is supplied. Existing CLI and server behavior already uses this convention.

Configure the Gemma instruction template from the selected checkpoint. The semantic form is normally:

```text
<bos><start_of_turn>user
...<end_of_turn>
<start_of_turn>model
```

Verify whether the converted RKLLM artifact already contains its chat template before calling `rkllm_set_chat_template`. Adding a second template would corrupt the prompt. If the application must set it, use the exact checkpoint template rather than the Qwen template currently hard-coded for SmolVLM.

Gemma 3's image tokens receive special bidirectional attention during prefill. This repository does not provide a token-type mask to RKLLM, so the converted Gemma model/runtime must implement that behavior internally. This is a required artifact-level validation.

### Update CLI, Server, and Documentation

The generic CLI and server paths should require few structural changes once the model profile is registered, but verify all of the following:

- `--model-family gemma3` is accepted by the CLI.
- `--vision`, `--image`, and `--llm` are required appropriately.
- INI configuration accepts `model_family=gemma3`.
- `Session::describe` reports the correct family and vision requirement.
- A prompt containing `<image>` requires uploaded image data on the server.
- A text-only prompt can run without invoking the vision encoder.
- Supplying image data without a corresponding placeholder is either rejected or documented as ignored.

Add a configuration example such as:

```ini
[gemma3-4b]
model_family=gemma3
vision=/models/gemma3/gemma3-vision-projector.rknn
llm=/models/gemma3/gemma3-4b-it.rkllm
max_new_tokens=256
max_context_len=4096
cores=3
```

Update the recognized-family list and model documentation in `README.md`. If downloadable artifacts are published, document their checkpoint, Toolkit version, RKNN Toolkit version, quantization settings, expected driver version, and checksums.

The helper [`scripts/send-query.py`](scripts/send-query.py) already inserts the generic `<image>` placeholder and should continue to work without a Gemma-specific branch.

### Add Tests

Extend [`cpp/tests/vlm_rknn_test.cc`](cpp/tests/vlm_rknn_test.cc) to cover:

- All accepted Gemma family aliases.
- Canonical `gemma3` family naming.
- `usesVisionEncoder == true`.
- `supportsMultimodal == true`.
- The `<image>` repository placeholder.
- Gemma preprocessing behavior.
- Successful parsing of a Gemma INI section with both `vision` and `llm`.
- Rejection of a Gemma INI section without `vision`.
- Session description output.

Where practical, factor tensor-shape validation into testable helpers and add cases for `[1, 256, hidden_size]`, wrong token counts, wrong embedding sizes, and unexpected output counts.

On an RK3588 device, add an integration test set containing:

- A text-only prompt.
- A simple object-description image.
- An OCR image.
- Landscape and portrait images.
- A request missing `image_data`.
- Repeated requests to detect stale embeddings or KV-cache contamination.
- A comparison with the Hugging Face model using the same prompts and images.

## Pan-and-Scan and Multiple Images

Gemma 3 can optionally divide an image into additional crops. Each crop produces another 256-token block. Supporting that correctly requires more than changing preprocessing:

- The processor must reproduce Gemma's crop-selection rules.
- The RKNN encoder must run once per crop or accept a batch.
- Embeddings must be concatenated in processor-defined order.
- `n_image` and the prompt's image blocks must match the resulting embeddings.
- Context-length accounting must include every 256-token block.
- The server request format may need to accept multiple images.

Defer this until single-image, no-pan-and-scan inference matches the reference model.

## Resource Considerations on RK3588

The 4B model should be the first target. The 12B and 27B variants require substantially more memory for weights and KV cache and may not be practical on many RK3588 boards, especially with large context lengths.

Initial validation should use a modest `max_context_len`, such as 4096, and measure actual high-water memory usage reported by RKLLM. Increase it only after the vision encoder, decoder weights, image embeddings, and KV cache fit reliably within the device's available memory.

Quantization affects both memory use and output quality. Record the exact quantization mode used for the `.rkllm` decoder and compare representative image tasks against the unquantized reference.

## Build and Device Validation

After every C++ or build-system change, iterate with the native Docker build as needed. Before finalizing Gemma support, attempt both required builds:

```bash
./scripts/build-native.sh docker
./scripts/build-android.sh docker
```

If `ANDROID_HOME` is set, also attempt:

```bash
./gradlew :app:assembleDebug
```

The binaries cannot be validated locally because they target RK3588. Final acceptance therefore requires running the paired `.rknn` and `.rkllm` artifacts on a device with compatible RKNN, RKLLM, and RKNPU driver versions.

Record at least:

- RKLLM Toolkit and runtime versions.
- RKNN Toolkit and runtime versions.
- RKNPU driver version.
- Board model and installed RAM.
- Vision encoder latency.
- Prompt-prefill and generation latency.
- Peak memory usage.
- Output for a fixed regression set of images and prompts.
