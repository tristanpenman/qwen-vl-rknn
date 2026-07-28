# Llama 3.2 Vision 11B

This note tracks the changes required to support `meta-llama/Llama-3.2-11B-Vision-Instruct` on RK3588. It is an initial design and feasibility assessment, not a claim that the model can currently be converted or run with RKLLM.

## Summary

Llama 3.2 Vision must be represented as a separate multimodal model family, probably `mllama` or `llama3.2-vision`. It must not reuse the current text-only `llama` profile.

The main difference from the current Qwen-VL path is architectural. Llama 3.2 Vision does not replace one image marker with a sequence of ordinary language model embeddings. Its vision adapter feeds image representations into eight dedicated cross-attention layers in the language model. Supporting it therefore requires all of the following:

- an RKNN-compatible Mllama vision model and multimodal projector;
- cross-attention K/V cache generation in the layout expected by RKLLM;
- an RKLLM model that retains Mllama's cross-attention decoder layers;
- Mllama-specific tiled image preprocessing and attention masks; and
- the Llama 3 chat template with the `<|image|>` marker in the correct place.

The first task is a conversion spike. The current Rockchip `rknn-llm` README lists text-only Llama among supported models, but its explicit multimodal list does not include Llama 3.2 Vision. Its public multimodal export example also has no Mllama vision adapter. A successful call to `RKLLM.load_huggingface()` for the 11B Vision checkpoint must not be assumed to mean that its cross-attention layers were preserved.

## 11B Architecture

Meta describes the 11B model as a 10.6-billion-parameter, 128K-context model built on Llama 3.1. A separately trained vision adapter injects image encoder representations into the language model through cross-attention layers. Image plus text inference is officially supported in English; the wider set of supported languages applies to text-only use.

Transformers implements the architecture as `MllamaForConditionalGeneration`. Its configuration defaults, documented against the 11B checkpoint, provide a useful conversion reference:

- text hidden size: 4096;
- text layers: 40;
- attention heads: 32;
- K/V heads: 8;
- cross-attention layers: 3, 8, 13, 18, 23, 28, 33, and 38;
- vision hidden size: 1280;
- local vision layers: 32;
- global vision layers: 8;
- projected vision output dimension before the multimodal projector: 7680;
- tile size: 448 by 448 with 14 by 14 patches; and
- maximum tiles per image: 4.

Each 448-pixel tile produces 32 by 32 patch tokens plus a class token, or 1025 vision tokens. The reference implementation pads internally for the global vision transformer, removes that padding at its output, concatenates selected intermediate vision-layer states with the final state, and projects the resulting 7680-wide features to the language model's 4096-wide hidden size. With four tiles, the vision model's documented output shape is `[1, 1, 4, 1025, 7680]` before that projection and flattening.

These values must be checked against the exact downloaded checkpoint revision before fixing RKNN tensor shapes. The Hugging Face checkpoint is gated, so the conversion record should include its revision and copies of the relevant configuration values rather than relying indefinitely on library defaults.

## Why the Existing Multimodal Path Is Insufficient

The current implementation assumes that an RKNN model:

1. accepts exactly one uint8 NHWC image tensor;
2. emits one or more tensors that can be concatenated into a flat sequence of image embeddings;
3. supplies one `n_image_tokens` value; and
4. passes those embeddings through `RKLLM_INPUT_MULTIMODAL`, using start, end, and content marker strings from `RKLLMParam`.

That matches the Rockchip Qwen-style demo, but not native Mllama semantics. Mllama uses the single `<|image|>` token to determine which following text positions may cross-attend to an image. The image features enter the decoder through cross-attention rather than replacing that marker with ordinary text embeddings.

The bundled RKLLM header contains a more promising interface:

- `RKLLMExtendParam::use_cross_attn` enables cross-attention;
- `RKLLMCrossAttnParam` accepts encoder K and V caches, an encoder mask, and encoder positions; and
- `rkllm_set_cross_attn_params()` attaches those values to a decoder handle.

Its documented cache layouts are:

```text
K: [num_layers][num_tokens][num_kv_heads][head_dim]
V: [num_layers][num_kv_heads][head_dim][num_tokens]
```

For the 11B model, `num_layers` is expected to mean the eight cross-attention layers, not all 40 text layers. That interpretation, the cache dtype, ownership, lifetime, masking convention, and whether `num_kv_heads` is eight all require validation against an exported model and the matching runtime.

## Conversion Feasibility Gate

Before changing the application API, establish that a compatible model pair can be produced.

1. Pin matching releases of RKLLM-Toolkit, `librkllmrt.so`, RKNN-Toolkit2, `librknnrt.so`, and the NPU driver.
2. Download an approved revision of `meta-llama/Llama-3.2-11B-Vision-Instruct` and record its license and configuration metadata.
3. Attempt `RKLLM.load_huggingface()` and conversion of the Mllama decoder.
4. Inspect the exported model or toolkit logs to prove that the eight cross-attention layers remain present and that `use_cross_attn` is enabled.
5. Build a CPU reference with Transformers and capture deterministic outputs for text-only and single-image prompts.
6. Determine the required boundary between the RKNN vision graph and RKLLM.

The likely boundary is not merely a `[tokens, 4096]` projected feature tensor. Because `rkllm_set_cross_attn_params()` accepts already projected K/V caches, the RKNN side may need to include the multimodal projector and the K/V projections belonging to all eight decoder cross-attention layers. If so, it will emit at least one K and one V tensor per cross-attention layer, or two packed tensors with equivalent content.

Do not begin broad CLI or Android work until a minimal single-tile image can be converted and its cross-attention cache boundary is known. If the current toolkit cannot export Mllama, the alternatives are to wait for Rockchip support or implement a custom decoder/backend; prompt-profile changes alone cannot make the model work.

## Image Preprocessing

The current square-padding profile cannot represent Mllama preprocessing. Mllama chooses a tiled canvas that approximately preserves the source aspect ratio, resizes the image to fit that canvas, pads it, and splits it into as many as four 448 by 448 tiles. The supported tile arrangements for four tiles are:

```text
1x1, 1x2, 1x3, 1x4, 2x1, 2x2, 3x1, 4x1
```

The processor returns more than pixels:

- `pixel_values`, padded to four tiles;
- `aspect_ratio_ids`, selecting learned aspect-ratio embeddings;
- `aspect_ratio_mask`, identifying valid rather than padded tiles; and
- `num_tiles`, used when constructing the text-to-image cross-attention mask.

Images are converted to RGB, rescaled, and normalized according to the exact checkpoint processor configuration. The host must read these settings from a model manifest or reproduce a pinned reference processor. It should not copy the current Qwen grey-padding values.

There are two plausible RKNN designs:

- one static graph with four image tiles plus aspect-ratio ID and mask inputs; or
- separate static graphs for the eight supported tile arrangements.

The first is easier to package but requires RKNN support for the additional integer/mask inputs and wastes work on padded tiles. The second can reduce work but complicates model packaging and session selection. Start with a fixed single-image, four-tile graph unless conversion shows that it is impractical.

The application's `VisionEncoder` and preprocessing APIs must stop assuming one RKNN input. Tensor metadata should be matched by manifest name and validated for dtype, rank, shape, and layout. Do not infer Mllama cache dimensions by looking for the first two dimensions greater than one.

## Prompt and Cross-Attention Semantics

The Instruct prompt uses the Llama 3 chat format. A minimal image turn is:

```text
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

<|image|>Describe this image.<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```

The `<|image|>` marker belongs inside the user message body, after its role header. It is a single special token, with configuration index 128256 in the Transformers Mllama configuration. It is not equivalent to Qwen's `<|vision_start|>`, `<|vision_end|>`, and `<|image_pad|>` triplet.

Image position is semantically important. An image may only be visible to subsequent text tokens. The reference processor builds a cross-attention mask from the location of each `<|image|>` token and the number of active tiles. The first milestone should support one image in one user turn, which also matches Meta's recommendation. Multi-image and multi-turn cache behaviour should remain explicitly unsupported until the RKLLM mask and cache semantics are understood.

Prompt rendering should be owned by the model profile. Raw prompts can remain useful for debugging, but normal CLI, server, and Android requests should accept message content and generate the special-token framing internally. This avoids asking callers to reproduce a fragile chat template.

## Runtime Changes

Add a distinct model family such as:

```cpp
enum class ModelFamily
{
    // Existing families...
    kLlama,
    kLlama3_2Vision,
};
```

The new profile should declare a cross-attention integration strategy rather than pretending that every vision model produces replacement embeddings. A possible direction is:

```cpp
enum class VisionIntegration
{
    kNone,
    kTokenEmbeddings,
    kCrossAttentionKv,
};
```

Replace the single float image-embedding buffer with a typed result:

```cpp
struct CrossAttentionImage
{
    std::vector<float> keyCache;
    std::vector<float> valueCache;
    std::vector<float> mask;
    std::vector<std::int32_t> positions;
    int layers;
    int tokens;
    int kvHeads;
    int headDim;
};
```

The exact storage type may need to differ if RKNN emits FP16 while the RKLLM API truly requires FP32. Avoid committing to conversion copies until runtime tests establish what the API accepts.

`Session::initTextDecoder()` will need to set `params.extend_param.use_cross_attn` for this family. Before generation, the session should populate an `RKLLMCrossAttnParam`, call `rkllm_set_cross_attn_params()`, and keep all backing buffers alive until generation finishes. Text-only runs should bypass the vision encoder and clear or disable stale cross-attention state.

The implementation should also:

- make vision input and output schemas profile-specific;
- include the multimodal projector in the documented model boundary;
- validate the eight K/V cache layers and all cache strides;
- generate mask and position arrays from the rendered prompt;
- clear cross-attention state together with the text KV cache;
- reject unsupported multi-image or history combinations; and
- ensure failure paths cannot reuse caches from a previous image.

## Configuration and User Interfaces

The model manifest or INI section needs more information than the existing `vision` and `llm` paths. At minimum, record:

- family `llama3.2-vision`;
- the RKLLM decoder path;
- the RKNN vision/cross-attention adapter path or paths;
- checkpoint name and immutable revision;
- toolkit and runtime versions;
- tile size, maximum tiles, and supported aspect ratios;
- cross-attention layer count and cache dimensions; and
- quantization used for each artifact.

Keep the external request format model-neutral. The server and Android app can continue to supply an image and prompt, while the selected session owns tiling, special tokens, and cross-attention setup. Advertise capability metadata so clients can distinguish text-only Llama from Llama 3.2 Vision.

The initial scope should be:

- Llama 3.2 11B Vision Instruct;
- one image;
- one user turn;
- English image prompts;
- deterministic greedy decoding for validation; and
- RK3588, preferably a board with enough RAM for the selected quantization and runtime overhead.

Do not imply that the full 128K trained context is practical on RK3588. The deployed maximum context must be chosen from measured model size, KV-cache size, cross-attention cache size, and available device memory. Even raw 8-bit storage for 10.6 billion parameters is roughly 10.6 GB before runtime buffers and caches; actual `.rkllm` size and peak resident memory must be measured.

## Validation Plan

Build a small reference corpus containing a square photograph, portrait, landscape, document, and diagram. For each input, save the pinned Transformers processor tensors and model results.

Validate the port in stages:

1. Compare canvas selection, resized pixels, padding, tiles, aspect-ratio ID, tile mask, token IDs, and cross-attention mask with Transformers.
2. Compare final and selected intermediate vision states.
3. Compare the 7680-to-4096 multimodal projector output.
4. Compare each cross-attention layer's K and V cache independently.
5. Compare text-prefill logits with an image attached.
6. Compare greedy generation token by token.
7. Repeat text-only inference to prove the Mllama decoder remains usable without invoking the vision graph.
8. Measure model load, image preprocessing, vision encoding, prefill, per-token generation, and peak memory on the RK3588.

Add unit tests for prompt rendering, image-marker placement, aspect-ratio selection, masks, cache-size arithmetic, invalid tensor metadata, and stale cache clearing. On-device tests should cover repeated prompts with different images because incorrect cache reuse may still produce superficially plausible text.

## Open Questions

- Does the current RKLLM-Toolkit recognize `MllamaForConditionalGeneration`, or only ordinary Llama causal language models?
- Can it export the eight cross-attention decoder layers into `.rkllm`?
- Which toolkit/runtime release introduced `use_cross_attn`, and is the bundled library ABI-compatible with the bundled header?
- Must RKNN emit raw projected vision states or per-layer cross-attention K/V caches?
- What dtype, alignment, layer order, and head layout does `rkllm_set_cross_attn_params()` actually accept?
- How should `encoder_mask` and `encoder_pos` encode tile padding and the point in the prompt after `<|image|>`?
- Can unused padded tiles be skipped without changing learned aspect-ratio embeddings or cache layout?
- What quantization fits the complete 11B pipeline on the target board with a useful context length?
- Does RKLLM apply the checkpoint's Llama 3 chat template correctly, or should this project always render it itself?
- How are cross-attention caches cleared or replaced between turns, especially when `keep_history` is enabled?

## Sources

- [Meta Llama 3.2 Vision model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD_VISION.md)
- [Meta Llama 3.2 Vision prompt format](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/vision_prompt_format.md)
- [Transformers Mllama documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/mllama.md)
- [Transformers Mllama configuration](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/configuration_mllama.py)
- [Transformers Mllama image processor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/image_processing_mllama.py)
- [Transformers Mllama processor and cross-attention mask construction](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/processing_mllama.py)
- [Transformers Mllama implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/modeling_mllama.py)
- [Rockchip RKLLM supported models and multimodal quickstart](https://github.com/airockchip/rknn-llm)
- [Rockchip multimodal RKLLM exporter](https://github.com/airockchip/rknn-llm/blob/main/examples/multimodal_model_demo/export/export_rkllm.py)
- [Rockchip multimodal vision exporter](https://github.com/airockchip/rknn-llm/blob/main/examples/multimodal_model_demo/export/export_vision.py)
