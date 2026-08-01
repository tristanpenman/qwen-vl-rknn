# Cross Attention

These notes relate to supporting [Llama 3.2 Vision 11B](./llama.md), which would require cross-attention with RKLLM.

RKLLM can support a cross-attention decoder loop. It does not expose the loop itself, but the API provides a mechanism for supplying precomputed encoder K/V caches to the hidden loop.

The important qualification is that Llama 3.2 Vision would require a specially converted RKLLM decoder and a substantially different vision-side pipeline. It cannot be supported by simply adding another `ModelProfile` and passing ordinary image embeddings through `RKLLM_INPUT_MULTIMODAL`.

## How the Existing Models Work

The current Qwen-VL and SmolVLM path is effectively:

```text
image
  -> RKNN vision encoder
  -> [image tokens, language hidden size]
  -> RKLLM_INPUT_MULTIMODAL
  -> RKLLM inserts those embeddings at image placeholder tokens
```

The implementation assumes:

- One uint8 NHWC image input: [`cpp/src/vlm_rknn.cc:768`](cpp/src/vlm_rknn.cc#L768)
- One or more similarly shaped embedding outputs: [`cpp/src/vlm_rknn.cc:809`](cpp/src/vlm_rknn.cc#L809)
- A flat image-embedding buffer passed through `RKLLM_INPUT_MULTIMODAL`: [`cpp/src/vlm_rknn.cc:885`](cpp/src/vlm_rknn.cc#L885)
- Image placeholder strings configured on the decoder: [`cpp/src/vlm_rknn.cc:611`](cpp/src/vlm_rknn.cc#L611)

That works for architectures where visual representations become additional inputs in the decoder's token stream.

## Why Mllama Is Different

Llama 3.2 11B Vision uses the `mllama` architecture. Its visual representations are not simply substituted for text-token embeddings. Instead, selected decoder layers perform cross-attention over them. The 11B configuration has cross-attention layers interleaved through the text model.

For every cross-attention layer, the image representation is transformed by that layer's own K and V projections. Hugging Face's implementation shows the per-layer `k_proj`, `v_proj`, K normalization, and reuse of those projected states during autoregressive decoding. See the [Transformers Mllama implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mllama/modeling_mllama.py).

Therefore, one shared `[image_tokens, hidden_size]` buffer is insufficient.

## RKLLM Cross-Attention

RKLLM's headers explicitly support providing a precomputed encoder K/V cache:

- `RKLLMExtendParam::use_cross_attn`: [`thirdparty/rkllm/include/rkllm.h:50`](thirdparty/rkllm/include/rkllm.h#L50)
- `RKLLMCrossAttnParam`: [`thirdparty/rkllm/include/rkllm.h:170`](thirdparty/rkllm/include/rkllm.h#L170)
- `rkllm_set_cross_attn_params`: [`thirdparty/rkllm/include/rkllm.h:394`](thirdparty/rkllm/include/rkllm.h#L394)

It expects:

```text
encoder_k_cache:
    [num_layers][num_tokens][num_kv_heads][head_dim]

encoder_v_cache:
    [num_layers][num_kv_heads][head_dim][num_tokens]

encoder_mask:
    [num_tokens]

encoder_pos:
    [num_tokens]
```

RKLLM still runs the opaque prefill/decode loop, but its internal cross-attention operations consume K/V caches supplied by the application.

## Implementation Challenges

This is not a simple task.

### Compatible Custom-Converted `.rkllm` Decoder

An ordinary text-only Llama 3.2 `.rkllm` cannot be used.

The converted model must preserve Mllama's cross-attention decoder layers, including their:

- Query and output projections
- Cross-attention normalization
- Attention and MLP gates
- Layer placement
- Expected number of K/V heads and head dimension

The conversion also needs to agree with RKLLM about which per-layer K/V projections are performed externally. The runtime library contains cross-attention tensor names, but Rockchip's public supported-model list currently includes Qwen-VL, InternVL, and SmolVLM, not Llama 3.2 Vision. See the [Rockchip RKLLM repository](https://github.com/airockchip/rknn-llm). **This may be a show-stopper**.

A useful result of this is that the first proof of feasibility is not a C++ change. It is whether the installed RKLLM Toolkit can custom-convert Mllama into a model accepted by this API. If it cannot express Mllama's decoder structure, then we're out of luck.

### New Mllama Vision/KV Encoder

The RKNN side would also need to perform more than the current vision encoder. Conceptually:

```text
Mllama image preprocessing and tiling
  -> Mllama vision tower
  -> multimodal projector
  -> each cross-attention layer's K/V projections
  -> K normalization
  -> RKLLM-compatible K/V layouts
```

This could be exported as one RKNN graph with many outputs or as a vision graph followed by one or more projection graphs. A single graph would usually reduce transfers.

### Mllama-Specific Preprocessing

The current square-padding path is not sufficient. Mllama processing includes:

- 560 x 560 tiles
- Aspect-ratio-dependent tiling, up to the model's configured maximum
- `aspect_ratio_ids`
- `aspect_ratio_mask`
- Model-specific rescaling and normalization
- Padding and masking of unused tiles

The current encoder rejects anything other than one RKNN input, so it cannot presently feed pixel data plus aspect-ratio metadata: [`cpp/src/vlm_rknn.cc:768`](cpp/src/vlm_rknn.cc#L768).

This likely calls for a separate encoder abstraction rather than adding more exceptions to `VisionEncoder`.

### Cross-Attention Buffers and Metadata

The session would need owned buffers for:

- K cache
- Transposed V cache
- Encoder mask
- Encoder positions
- Token and tile metadata

Their dimensions should come from explicit model configuration or validated tensor attributes. The existing heuristic that takes the first adjacent dimensions greater than one is not robust enough for these tensors: [`cpp/src/vlm_rknn.cc:167`](cpp/src/vlm_rknn.cc#L167).

These buffers can also be large because they contain a separate K and V representation for every image token and cross-attention layer.

### A Distinct Decode Path

Initialization would need:

```cpp
params.extend_param.use_cross_attn = 1;
rkllm_init(...);
```

Then, before generation:

```cpp
RKLLMCrossAttnParam cross{};
cross.encoder_k_cache = ...;
cross.encoder_v_cache = ...;
cross.encoder_mask = ...;
cross.encoder_pos = ...;
cross.num_tokens = ...;

rkllm_set_cross_attn_params(handle, &cross);
rkllm_run(handle, ...);
```

The prompt would use Mllama's tokenizer, chat template, and image-token semantics. It should not go through the present Qwen-style `RKLLM_INPUT_MULTIMODAL.image_embed` path.

Mask and position behavior would need to be verified carefully against Rockchip's custom-conversion contract. Mllama's logical cross-attention mask depends on where images become visible to text tokens, while the RKLLM structure exposes only one-dimensional encoder mask and position arrays.

## Vision Integration Strategy

Assume that we declare a new cross-attention integration strategy rather than pretending that every vision model produces replacement embeddings. A possible direction is:

```cpp
enum class VisionIntegration
{
    kNone,
    kTokenEmbeddings,
    kCrossAttentionKv,
};
```

## Verdict

It is possible in principle with this RKLLM version. The decoder loop being hidden is not the blocker; `rkllm_set_cross_attn_params` is explicitly the interface into that loop.

Support is conditional on obtaining a custom-converted Mllama RKLLM decoder whose internal architecture matches the supplied caches. Without that artifact, or Toolkit support capable of producing it, this cannot be implemented purely in this repository.

The practical assessment is:

- Application-side integration: feasible, but substantial.
- RKNN vision/KV export: probably feasible, subject to supported operators and output size.
- RKLLM decoder conversion: currently unproven and the highest-risk dependency.
- Reusing the normal Llama `.rkllm` or Qwen-style multimodal route: not viable.

The current uncommitted `kMllama` block in `cpp/src/vlm_rknn.cc` is incomplete, unregistered in `ModelFamily`, and presently syntactically invalid. Even once completed, a profile alone would cover only configuration; it would not implement the cross-attention data path described above.
