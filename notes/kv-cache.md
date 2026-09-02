# KV-Cache

_Reusing a prompt prefix with RKLLM KV cache_

RKLLM's prompt-cache API is intended for applications that repeatedly use the same prompt prefix. Generate the cache once from the fixed prefix, save it to a file, load it after model initialization, and then submit the changing user input.

For example, the RKLLM API exposes the following types and functions:

```cpp
typedef struct {
    int save_prompt_cache;
    const char* prompt_cache_path;
} RKLLMPromptCacheParam;

int rkllm_load_prompt_cache(
    LLMHandle handle,
    const char* prompt_cache_path);

int rkllm_release_prompt_cache(LLMHandle handle);
```

These APIs are present in Rockchip's [official RKLLM header][rkllm-header], and the official [API demo][rkllm-demo] shows the same configuration.

## Create the cache once

Initialize the model normally, then run the fixed prefix with `save_prompt_cache` enabled:

```cpp
const char* cache_path = "/data/local/tmp/my-prefix.cache";
const std::string fixed_prefix =
    "Your approximately 250-token application prompt ...";

RKLLMInput input{};
input.input_type = RKLLM_INPUT_PROMPT;
input.role = "user";
input.enable_thinking = false;
input.prompt_input = fixed_prefix.c_str();

RKLLMPromptCacheParam cache_param{};
cache_param.save_prompt_cache = 1;
cache_param.prompt_cache_path = cache_path;

RKLLMInferParam infer{};
infer.mode = RKLLM_INFER_GENERATE;
infer.prompt_cache_params = &cache_param;
infer.keep_history = 0;

int ret = rkllm_run(handle, &input, &infer, nullptr);
if (ret != 0) {
    // Cache creation failed.
}
```

The cache is written during this `rkllm_run()` operation. Because `rkllm_run()` is a generation API, it may also produce an answer while building the cache; discard that callback output.

If the RKLLM version provides per-run `max_new_tokens`, set it to the minimum supported value for this cache-building call. The header currently vendored in this repository does not have that field, so its generation limit must be controlled through the `RKLLMParam` used at initialization.

After the call completes, confirm that the cache file exists and is non-empty before treating creation as successful.

## Load the cache when the application starts

Load the cache after `rkllm_init()` and before serving requests:

```cpp
int ret = rkllm_load_prompt_cache(handle, cache_path);
if (ret != 0) {
    // The file may be incompatible, corrupt, or unsupported by this model.
}
```

Loading is per `LLMHandle`. If the application has several model handles or worker processes, each one needs to load the cache.

## Submit the changing continuation

Once the cache is loaded, the input is the text that logically follows the cached prefix:

```cpp
std::string user_suffix = user_input;

RKLLMInput input{};
input.input_type = RKLLM_INPUT_PROMPT;
input.role = "user";
input.enable_thinking = false;
input.prompt_input = user_suffix.c_str();

RKLLMInferParam infer{};
infer.mode = RKLLM_INFER_GENERATE;
infer.prompt_cache_params = nullptr;
infer.keep_history = 0;

int ret = rkllm_run(handle, &input, &infer, request_context);
```

Do not enable `save_prompt_cache` on normal requests, because doing so may repeatedly overwrite the cache.

Use `keep_history = 0` for independent requests that share the same prefix. `keep_history = 1` is for an accumulating conversation and can cause generated tokens from one request to become context for subsequent calls.

## Prompt formatting

The cache must end at a valid continuation boundary. For an instruct or chat model, the effective sequence should resemble:

```text
[cached system/application prefix][user input][assistant-generation marker]
```

There are two common ways to arrange this:

- Put the fixed content in RKLLM's system prompt or chat template, cache that system portion, and pass each request using `role = "user"`.
- Construct the exact model-specific token sequence, cache everything up to the point where the variable user text begins, and ensure that the suffix supplies the remaining user and assistant delimiters.

Avoid accidentally caching the start of a completed user turn if the later input is supposed to continue inside that turn. Tokenization at the boundary must be identical to tokenizing the combined prompt. Spaces, newlines, and chat control tokens can all affect that boundary.

## Compatibility constraints

Treat a prompt-cache file as tied to:

- The exact `.rkllm` model.
- The RKLLM runtime and toolkit version.
- The model's chat template and prefix text.
- The LoRA configuration, if applicable.
- Thinking mode and other formatting choices that affect the input tokens.

Regenerate the cache when any of these change. Do not assume that a cache produced for one model or runtime release is portable to another.

Model support is version-dependent. Rockchip's [changelog][rkllm-changelog] says prompt-cache storage and preloading were introduced in RKLLM 1.1. Some architectures or runtime versions may nevertheless reject saving a prompt cache. Check the API return code and runtime log on the target device.

For a 250-token prefix, compare `prefill_time_ms` with and without the loaded cache. This verifies actual reuse rather than merely confirming that the file loaded. Setting `RKLLM_LOG_LEVEL=1` is also useful for observing prompt token counts and the active KV-cache size.

[rkllm-header]: https://github.com/airockchip/rknn-llm/blob/main/rkllm-runtime/Linux/librkllm_api/include/rkllm.h
[rkllm-demo]: https://github.com/airockchip/rknn-llm/blob/main/examples/rkllm_api_demo/deploy/src/llm_demo.cpp
[rkllm-changelog]: https://github.com/airockchip/rknn-llm/blob/main/CHANGELOG.md
