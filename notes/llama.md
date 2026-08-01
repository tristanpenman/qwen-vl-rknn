# Llama 3.2 Vision 11B

This note introduces concepts required to support `meta-llama/Llama-3.2-11B-Vision-Instruct` on RK3588. It is an exploratory assessment, not a claim that the model can currently be converted or run with RKLLM.

See [Cross-Attention](./cross-attention.md) for notes on supporting cross-attention on RK3588.

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

## Prompt and Cross-Attention Semantics

The Instruct prompt uses the Llama 3 chat format. A minimal image turn is:

```text
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

<|image|>Describe this image.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

The `<|image|>` marker belongs inside the user message body, after its role header. It is a single special token, with configuration index 128256 in the Transformers Mllama configuration. It is not equivalent to Qwen's `<|vision_start|>`, `<|vision_end|>`, and `<|image_pad|>` triplet.

Image position is semantically important. An image may only be visible to subsequent text tokens. The reference processor builds a cross-attention mask from the location of each `<|image|>` token and the number of active tiles. The first milestone should support one image in one user turn, which also matches Meta's recommendation. Multi-image and multi-turn cache behaviour should remain explicitly unsupported until the RKLLM mask and cache semantics are understood.

Prompt rendering should be owned by the model profile. Raw prompts can remain useful for debugging, but normal CLI, server, and Android requests should accept message content and generate the special-token framing internally. This avoids asking callers to reproduce a fragile chat template.
