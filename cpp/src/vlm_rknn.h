#pragma once

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>
#include <rknn_api.h>
#include <rkllm.h>

namespace vlm_rknn {

struct TextDecoder
{
    LLMHandle handle = nullptr;
};

struct VisionEncoder
{
    rknn_context rknnContext = 0;
    rknn_input_output_num ioNum {};
    rknn_tensor_attr* inputAttrs = nullptr;
    rknn_tensor_attr* outputAttrs = nullptr;

    int modelChannel = 0;
    int modelWidth = 0;
    int modelHeight = 0;

    int modelImageToken = 0;
    int modelEmbedSize = 0;
};

enum class ModelFamily
{
    kQwenVL2,
    kQwenVL2_5,
    kQwenVL3,
    kLlama,
    kSmolVLM2,
    kGemma3,
};

enum class ResizeMode
{
    kPadToSquare,
    kStretch,
    kCenterCrop,
};

struct ImagePreprocessProfile
{
    ResizeMode resizeMode;
    bool rgb;
    bool normalizeInHost;
    float padR;
    float padG;
    float padB;
    std::array<float, 3> mean;
    std::array<float, 3> std;
};

struct ModelConfig
{
    ModelFamily modelFamily = ModelFamily::kQwenVL2;

    std::optional<std::string> visionEncoderPath;
    std::string languageModelPath;

    int maxNewTokens = 128;
    int maxContextLen = 2048;

    std::optional<int> numCores;
};

// A ModelConfig together with the identifier it was defined under (the INI
// section name). The server uses the identifier to route requests.
struct NamedModelConfig
{
    std::string modelId;
    ModelConfig config;
};

// Parse an INI document into an ordered list of named model configurations.
//
// Each [section] defines one model, keyed by the section name. The recognised
// keys mirror the command-line options, with underscores instead of hyphens:
//   model_family, vision, llm, max_new_tokens, max_context_len, cores
// `llm` is required, and `vision` is required for model families that use a
// vision encoder (and rejected for those that do not). The order of the result
// matches the order of the sections, so the first entry is the default model.
//
// On failure, returns false and populates `error` with a human-readable message.
bool parseModelConfigsFromIni(
    const std::string& iniText,
    std::vector<NamedModelConfig>& out,
    std::string& error);

bool parseModelFamily(std::string_view value, ModelFamily& family);
const char* modelFamilyName(ModelFamily family);
const char* modelFamilyImagePlaceholder(ModelFamily family);
const ImagePreprocessProfile& modelFamilyImagePreprocessProfile(ModelFamily family);
bool modelFamilyUsesVisionEncoder(ModelFamily family);
bool modelFamilySupportsMultimodal(ModelFamily family);
int preprocessImageForVisionEncoder(
    ModelFamily family,
    const cv::Mat& imageBgr,
    cv::Size targetSize,
    cv::Mat& output);

class Session
{
public:
    using OutputCallback = std::function<void(const char* text, LLMCallState state)>;

private:
    int initVisionEncoder();
    int initTextDecoder();

    void cleanupVisionEncoder();
    void cleanupTextDecoder();

public:
    explicit Session(ModelConfig config);

    ~Session();

    int init();
    const ModelConfig& config() const noexcept;
    bool isReady() const noexcept;
    std::string describe() const;
    bool promptContainsImage(const std::string& prompt) const;
    void setOutputCallback(OutputCallback callback);

    const VisionEncoder& visionEncoder() const
    {
        return encoder_;
    }

    const TextDecoder& textDecoder() const
    {
        return decoder_;
    }

    // Run the vision encoder and fill the output buffer with the resulting embedding.
    int encode(void* imgData, float* outResult);

    // Run the text decoder; results are delivered via the RKLLM callback.
    int decode(const std::string& prompt, float* imgVec);

private:
    static int callback(RKLLMResult* result, void* userdata, LLMCallState state);

    ModelConfig config_;
    VisionEncoder encoder_;
    TextDecoder decoder_;

    std::string lastDecodedText_;
    OutputCallback outputCallback_;
};

}  // namespace vlm_rknn
