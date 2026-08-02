#include "vlm_rknn.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "ini.h"
#include "logger.h"
#include "rknn_utils.h"

namespace vlm_rknn {

namespace {

struct ModelProfile
{
    bool usesVisionEncoder;
    bool supportsMultimodal;
    std::int32_t baseDomainId;
    bool useChatTemplate;
    const char* imagePlaceholder;
    const char* imgStart;
    const char* imgEnd;
    const char* imgContent;
    int expectedImageTokens;
    ImagePreprocessProfile imagePreprocess;
};

const ModelProfile& modelProfileFor(ModelFamily family)
{
    static constexpr ImagePreprocessProfile kQwenVlImagePreprocess {
        ResizeMode::kPadToSquare,
        true,
        false,
        127.5f,
        127.5f,
        127.5f,
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 1.0f},
    };

    static constexpr ImagePreprocessProfile kLlamaImagePreprocess {
        ResizeMode::kPadToSquare,
        true,
        false,
        0.0f,
        0.0f,
        0.0f,
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 1.0f},
    };

    static constexpr ModelProfile kQwen2Vl {
        true,
        true,
        0,
        false,
        "<image>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        0,
        kQwenVlImagePreprocess,
    };

    // TODO: Change if necessary when the actual RKLLM models are available
    static constexpr ModelProfile kQwen25Vl {
        true,
        true,
        0,
        false,
        "<image>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        0,
        kQwenVlImagePreprocess,
    };

    // TODO: Change if necessary
    static constexpr ModelProfile kQwen3Vl {
        true,
        true,
        0,
        false,
        "<image>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        0,
        kQwenVlImagePreprocess,
    };

    static constexpr ModelProfile kLlama {
        false,
        false,
        0,
        false,
        "",
        "",
        "",
        "",
        0,
        kLlamaImagePreprocess,
    };

    static constexpr ImagePreprocessProfile kSmolvlm2ImagePreprocess {
        ResizeMode::kPadToSquare,
        true,
        false,
        127.5f,
        127.5f,
        127.5f,
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 1.0f},
    };

    // The SmolVLM2-256M-NPU reference currently uses the same RKLLM multimodal
    // marker strings as the Qwen-VL path.
    static constexpr ModelProfile kSmolvlm2 {
        true,
        true,
        1,
        true,
        "<image>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        0,
        kSmolvlm2ImagePreprocess,
    };

    // Gemma 3's processor produces a fixed 896x896 image. Rescaling and
    // normalization are expected to be baked into the RKNN artifact so the
    // existing uint8 NHWC input path remains usable.
    static constexpr ImagePreprocessProfile kGemma3ImagePreprocess {
        ResizeMode::kStretch,
        true,
        false,
        0.0f,
        0.0f,
        0.0f,
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 1.0f},
    };

    // base_domain_id and chat-template handling must be verified against the
    // converted RKLLM artifact. Keep the runtime default domain and do not
    // install an application-side template in this initial integration.
    static constexpr ModelProfile kGemma3 {
        true,
        true,
        0,
        false,
        "<image>",
        "<start_of_image>",
        "<end_of_image>",
        "<image_soft_token>",
        256,
        kGemma3ImagePreprocess,
    };

    switch (family) {
    case ModelFamily::kQwenVL2:
        return kQwen2Vl;
    case ModelFamily::kQwenVL2_5:
        return kQwen25Vl;
    case ModelFamily::kQwenVL3:
        return kQwen3Vl;
    case ModelFamily::kLlama:
        return kLlama;
    case ModelFamily::kSmolVLM2:
        return kSmolvlm2;
    case ModelFamily::kGemma3:
        return kGemma3;
    }

    return kQwen2Vl;
}

bool inferEmbeddingShapeFromAttr(
    const rknn_tensor_attr& attr,
    int& imageTokens,
    int& embedSize)
{
    for (std::uint32_t i = 0; i + 1 < attr.n_dims && i + 1 < RKNN_MAX_DIMS; ++i) {
        if (attr.dims[i] > 1 && attr.dims[i + 1] > 1) {
            imageTokens = attr.dims[i];
            embedSize = attr.dims[i + 1];
            return true;
        }
    }
    return false;
}

cv::Mat padToSquare(const cv::Mat& image, const cv::Scalar& backgroundColor)
{
    const int width = image.cols;
    const int height = image.rows;
    if (width == height) {
        return image.clone();
    }

    const int size = std::max(width, height);
    cv::Mat square(size, size, image.type(), backgroundColor);
    const int xOffset = (size - width) / 2;
    const int yOffset = (size - height) / 2;
    image.copyTo(square(cv::Rect(xOffset, yOffset, width, height)));
    return square;
}

cv::Mat centerCropToAspect(const cv::Mat& image, double targetAspect)
{
    const double sourceAspect = static_cast<double>(image.cols) / static_cast<double>(image.rows);
    if (sourceAspect > targetAspect) {
        const int cropWidth = static_cast<int>(image.rows * targetAspect);
        const int xOffset = (image.cols - cropWidth) / 2;
        return image(cv::Rect(xOffset, 0, cropWidth, image.rows)).clone();
    }

    const int cropHeight = static_cast<int>(image.cols / targetAspect);
    const int yOffset = (image.rows - cropHeight) / 2;
    return image(cv::Rect(0, yOffset, image.cols, cropHeight)).clone();
}

}  // namespace

bool parseModelFamily(std::string_view value, ModelFamily& family)
{
    if (value == "qwen2-vl" || value == "qwen2_vl" || value == "qwen2") {
        family = ModelFamily::kQwenVL2;
        return true;
    }

    if (value == "qwen2.5-vl" || value == "qwen2_5_vl" ||
        value == "qwen25-vl" || value == "qwen2.5") {
        family = ModelFamily::kQwenVL2_5;
        return true;
    }

    if (value == "qwen3-vl" || value == "qwen3_vl" || value == "qwen3") {
        family = ModelFamily::kQwenVL3;
        return true;
    }

    if (value == "llama" || value == "llama3" || value == "llama3.2") {
        family = ModelFamily::kLlama;
        return true;
    }

    if (value == "smolvlm2" || value == "smol-vlm2" || value == "smol_vlm2") {
        family = ModelFamily::kSmolVLM2;
        return true;
    }

    if (value == "gemma3" || value == "gemma-3" || value == "gemma_3") {
        family = ModelFamily::kGemma3;
        return true;
    }

    return false;
}

namespace {

// Recognised INI keys for a model section. Mirrors the command-line options,
// with underscores instead of hyphens.
bool isRecognisedModelKey(const std::string& key)
{
    return key == "model_family" || key == "vision" || key == "llm"
        || key == "max_new_tokens" || key == "max_context_len" || key == "cores";
}

bool parseIntValue(
    const std::string& key,
    const std::string& value,
    int minValue,
    int maxValue,
    int& parsed,
    std::string& error)
{
    errno = 0;
    char* end = nullptr;
    const long result = std::strtol(value.c_str(), &end, 10);
    if (value.empty() || *end != '\0' || errno == ERANGE
        || result < minValue || result > maxValue) {
        error = "invalid value for '" + key + "': " + value
            + " (expected " + std::to_string(minValue) + "-" + std::to_string(maxValue) + ")";
        return false;
    }

    parsed = static_cast<int>(result);
    return true;
}

}  // namespace

bool parseModelConfigsFromIni(
    const std::string& iniText,
    std::vector<NamedModelConfig>& out,
    std::string& error)
{
    out.clear();

    ini::Document document;
    if (!ini::parse(iniText, document, error)) {
        return false;
    }

    if (document.sections.empty()) {
        error = "no models defined; expected at least one [model_id] section";
        return false;
    }

    for (const auto& section : document.sections) {
        ModelConfig config;

        for (const auto& entry : section.entries) {
            if (!isRecognisedModelKey(entry.first)) {
                error = "[" + section.name + "]: unknown key '" + entry.first + "'";
                return false;
            }
        }

        if (const auto family = section.get("model_family"); family.has_value()) {
            if (!parseModelFamily(*family, config.modelFamily)) {
                error = "[" + section.name + "]: invalid model_family: " + *family;
                return false;
            }
        }

        if (const auto value = section.get("max_new_tokens"); value.has_value()) {
            if (!parseIntValue("max_new_tokens", *value, 1, INT_MAX, config.maxNewTokens, error)) {
                error = "[" + section.name + "]: " + error;
                return false;
            }
        }

        if (const auto value = section.get("max_context_len"); value.has_value()) {
            if (!parseIntValue("max_context_len", *value, 1, INT_MAX, config.maxContextLen, error)) {
                error = "[" + section.name + "]: " + error;
                return false;
            }
        }

        if (const auto value = section.get("cores"); value.has_value()) {
            int cores = 0;
            if (!parseIntValue("cores", *value, 1, 3, cores, error)) {
                error = "[" + section.name + "]: " + error;
                return false;
            }
            config.numCores = cores;
        }

        const auto llm = section.get("llm");
        if (!llm.has_value() || llm->empty()) {
            error = "[" + section.name + "]: missing required key 'llm'";
            return false;
        }
        config.languageModelPath = *llm;

        const auto vision = section.get("vision");
        const bool hasVision = vision.has_value() && !vision->empty();
        if (modelFamilyUsesVisionEncoder(config.modelFamily)) {
            if (!hasVision) {
                error = "[" + section.name + "]: missing required key 'vision' for "
                    + modelFamilyName(config.modelFamily);
                return false;
            }
            config.visionEncoderPath = *vision;
        } else if (hasVision) {
            error = "[" + section.name + "]: 'vision' is not supported for "
                + modelFamilyName(config.modelFamily);
            return false;
        }

        out.push_back(NamedModelConfig{section.name, std::move(config)});
    }

    return true;
}

const char* modelFamilyName(ModelFamily family)
{
    switch (family) {
    case ModelFamily::kQwenVL2:
        return "qwen2-vl";
    case ModelFamily::kQwenVL2_5:
        return "qwen2.5-vl";
    case ModelFamily::kQwenVL3:
        return "qwen3-vl";
    case ModelFamily::kLlama:
        return "llama";
    case ModelFamily::kSmolVLM2:
        return "smolvlm2";
    case ModelFamily::kGemma3:
        return "gemma3";
    }

    return "qwen2-vl";
}

const char* modelFamilyImagePlaceholder(ModelFamily family)
{
    return modelProfileFor(family).imagePlaceholder;
}

const ImagePreprocessProfile& modelFamilyImagePreprocessProfile(ModelFamily family)
{
    return modelProfileFor(family).imagePreprocess;
}

bool modelFamilyUsesVisionEncoder(ModelFamily family)
{
    return modelProfileFor(family).usesVisionEncoder;
}

bool modelFamilySupportsMultimodal(ModelFamily family)
{
    return modelProfileFor(family).supportsMultimodal;
}

int preprocessImageForVisionEncoder(
    ModelFamily family,
    const cv::Mat& imageBgr,
    cv::Size targetSize,
    cv::Mat& output)
{
    if (imageBgr.empty()) {
        LOG(ERROR) << "Image input is empty";
        return -1;
    }
    if (targetSize.width <= 0 || targetSize.height <= 0) {
        LOG(ERROR) << "Invalid target image size: " << targetSize.width << "x" << targetSize.height;
        return -1;
    }

    const auto& profile = modelFamilyImagePreprocessProfile(family);
    if (profile.normalizeInHost) {
        LOG(ERROR) << "Host-side image normalization is not supported by the current uint8 RKNN input path";
        return -1;
    }

    cv::Mat color;
    if (profile.rgb) {
        if (imageBgr.channels() == 3) {
            cv::cvtColor(imageBgr, color, cv::COLOR_BGR2RGB);
        } else if (imageBgr.channels() == 4) {
            cv::cvtColor(imageBgr, color, cv::COLOR_BGRA2RGB);
        } else {
            LOG(ERROR) << "Unsupported image channel count for RGB conversion: " << imageBgr.channels();
            return -1;
        }
    } else {
        if (imageBgr.channels() != 3) {
            LOG(ERROR) << "Unsupported image channel count: " << imageBgr.channels();
            return -1;
        }
        color = imageBgr.clone();
    }

    cv::Mat resized;
    switch (profile.resizeMode) {
    case ResizeMode::kPadToSquare: {
        cv::Mat square = padToSquare(color, cv::Scalar(profile.padR, profile.padG, profile.padB));
        cv::resize(square, resized, targetSize, 0, 0, cv::INTER_LINEAR);
        break;
    }
    case ResizeMode::kStretch:
        cv::resize(color, resized, targetSize, 0, 0, cv::INTER_LINEAR);
        break;
    case ResizeMode::kCenterCrop: {
        const double targetAspect = static_cast<double>(targetSize.width) / static_cast<double>(targetSize.height);
        cv::Mat cropped = centerCropToAspect(color, targetAspect);
        cv::resize(cropped, resized, targetSize, 0, 0, cv::INTER_LINEAR);
        break;
    }
    }

    output = resized.isContinuous() ? resized : resized.clone();
    return 0;
}

int Session::initVisionEncoder()
{
    memset(&encoder_, 0, sizeof(encoder_));

    // Initialize vision encoder
    rknn_context ctx = 0;

    // Define a cleanup lambda to ensure resources are released on failure
    const auto cleanupOnFailure = [&]() {
        cleanupVisionEncoder();
        if (ctx != 0) {
            rknn_destroy(ctx);
            ctx = 0;
        }
    };

    if (!config_.visionEncoderPath.has_value() || config_.visionEncoderPath->empty()) {
        LOG(ERROR) << "Vision encoder path is required for " << modelFamilyName(config_.modelFamily);
        return -1;
    }
    int ret = rknn_init(&ctx, static_cast<void*>(const_cast<char*>(config_.visionEncoderPath->c_str())), 0, 0, nullptr);
    if (ret < 0) {
        LOG(ERROR) << "Failed to initialize RKNN, error=" << rknn_utils::rknnErrorMessage(ret);
        return -1;
    }

    rknn_utils::logRknnVersion(ctx);

    // Set RKNN core mask if specified in the configuration.
    if (config_.numCores.has_value()) {
        int coreNum = config_.numCores.value();
        if (coreNum == 1) {
            ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0);
        } else if (coreNum == 2) {
            ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1);
        } else if (coreNum == 3) {
            ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2);
        } else {
            LOG(ERROR) << "Invalid RKNN core count: " << coreNum;
            cleanupOnFailure();
            return -1;
        }
        if (ret != 0) {
            LOG(ERROR) << "Failed to set RKNN core mask, error=" << rknn_utils::rknnErrorMessage(ret);
            cleanupOnFailure();
            return -1;
        }
    }

    // Query model I/O information.
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &encoder_.ioNum, sizeof(encoder_.ioNum));
    if (ret != RKNN_SUCC) {
        LOG(ERROR) << "Failed to query RKNN input/output count, error=" << rknn_utils::rknnErrorMessage(ret);
        cleanupOnFailure();
        return -1;
    }

    if (encoder_.ioNum.n_input <= 0 || encoder_.ioNum.n_output <= 0) {
        LOG(ERROR) << "Invalid RKNN I/O count: n_input=" << encoder_.ioNum.n_input
                   << " n_output=" << encoder_.ioNum.n_output;
        cleanupOnFailure();
        return -1;
    }
    if (Logger::verbose()) {
        LOG(VERBOSE) << "RKNN model I/O count: inputs=" << encoder_.ioNum.n_input
                     << " outputs=" << encoder_.ioNum.n_output;
    }

    encoder_.inputAttrs = static_cast<rknn_tensor_attr*>(calloc(
        encoder_.ioNum.n_input,
        sizeof(rknn_tensor_attr)));
    if (encoder_.inputAttrs == nullptr) {
        LOG(ERROR) << "Failed to allocate input tensor attribute buffers";
        cleanupOnFailure();
        return -1;
    }

    encoder_.outputAttrs = static_cast<rknn_tensor_attr*>(calloc(
        encoder_.ioNum.n_output,
        sizeof(rknn_tensor_attr)));
    if (encoder_.outputAttrs == nullptr) {
        LOG(ERROR) << "Failed to allocate output tensor attribute buffers";
        cleanupOnFailure();
        return -1;
    }

    for (auto i = 0; i < encoder_.ioNum.n_input; ++i) {
        encoder_.inputAttrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &encoder_.inputAttrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOG(ERROR) << "Failed to query RKNN input attr for index " << i
                       << ", error=" << rknn_utils::rknnErrorMessage(ret);
            cleanupOnFailure();
            return -1;
        }
        if (Logger::verbose()) {
            LOG(VERBOSE) << "RKNN input tensor: " << rknn_utils::tensorAttrToString(encoder_.inputAttrs[i]);
        }
    }

    for (auto i = 0; i < encoder_.ioNum.n_output; ++i) {
        encoder_.outputAttrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &encoder_.outputAttrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOG(ERROR) << "Failed to query RKNN output attr for index " << i
                       << ", error=" << rknn_utils::rknnErrorMessage(ret);
            cleanupOnFailure();
            return -1;
        }
        if (Logger::verbose()) {
            LOG(VERBOSE) << "RKNN output tensor: " << rknn_utils::tensorAttrToString(encoder_.outputAttrs[i]);
        }
    }

    if (!inferEmbeddingShapeFromAttr(
            encoder_.outputAttrs[0],
            encoder_.modelImageToken,
            encoder_.modelEmbedSize)) {
        LOG(ERROR) << "Could not infer image token and embedding dimensions from first RKNN output tensor";
        cleanupOnFailure();
        return -1;
    }

    const auto& profile = modelProfileFor(config_.modelFamily);
    if (profile.expectedImageTokens > 0) {
        if (encoder_.ioNum.n_input != 1 || encoder_.ioNum.n_output != 1) {
            LOG(ERROR) << modelFamilyName(config_.modelFamily)
                       << " requires exactly one RKNN input and one output; got inputs="
                       << encoder_.ioNum.n_input << " outputs=" << encoder_.ioNum.n_output;
            cleanupOnFailure();
            return -1;
        }
        if (encoder_.modelImageToken != profile.expectedImageTokens) {
            LOG(ERROR) << modelFamilyName(config_.modelFamily) << " requires "
                       << profile.expectedImageTokens << " image tokens; RKNN output has "
                       << encoder_.modelImageToken;
            cleanupOnFailure();
            return -1;
        }
    }

    if (encoder_.inputAttrs[0].fmt == RKNN_TENSOR_NCHW) {
        encoder_.modelChannel = encoder_.inputAttrs[0].dims[1];
        encoder_.modelHeight = encoder_.inputAttrs[0].dims[2];
        encoder_.modelWidth = encoder_.inputAttrs[0].dims[3];
    } else {
        encoder_.modelHeight = encoder_.inputAttrs[0].dims[1];
        encoder_.modelWidth = encoder_.inputAttrs[0].dims[2];
        encoder_.modelChannel = encoder_.inputAttrs[0].dims[3];
    }
    if (config_.modelFamily == ModelFamily::kGemma3 &&
        (encoder_.modelWidth != 896 || encoder_.modelHeight != 896 || encoder_.modelChannel != 3)) {
        LOG(ERROR) << "Gemma 3 RKNN input must be 896x896 RGB; got "
                   << encoder_.modelWidth << "x" << encoder_.modelHeight
                   << " channels=" << encoder_.modelChannel;
        cleanupOnFailure();
        return -1;
    }
    if (Logger::verbose()) {
        LOG(VERBOSE) << "RKNN vision input shape: height=" << encoder_.modelHeight
                     << " width=" << encoder_.modelWidth
                     << " channel=" << encoder_.modelChannel;
        LOG(VERBOSE) << "RKNN image embedding shape: tokens=" << encoder_.modelImageToken
                     << " embedSize=" << encoder_.modelEmbedSize
                     << " output_tensors=" << encoder_.ioNum.n_output;
    }

    // Save context for later use.
    encoder_.rknnContext = ctx;
    ctx = 0;

    return 0;
}

int Session::initTextDecoder()
{
    // Prepare RKLLM parameters based on the configuration.
    const auto& profile = modelProfileFor(config_.modelFamily);
    RKLLMParam params = rkllm_createDefaultParam();
    params.model_path = config_.languageModelPath.c_str();
    params.top_k = 1;
    params.max_new_tokens = config_.maxNewTokens;
    params.max_context_len = config_.maxContextLen;
    params.skip_special_token = true;
    params.extend_param.base_domain_id = profile.baseDomainId;
    params.img_start = profile.imgStart;
    params.img_end = profile.imgEnd;
    params.img_content = profile.imgContent;

    // Initialize RKLLM session.
    int ret = rkllm_init(&decoder_.handle, &params, &callback);
    if (ret != 0) {
        LOG(ERROR) << "Failed to initialize RKLLM, error=" << ret;
        return -1;
    }

    if (profile.useChatTemplate) {
        ret = rkllm_set_chat_template(
            decoder_.handle,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
            "<|im_start|>user\n",
            "<|im_end|>\n<|im_start|>assistant\n");
        if (ret != 0) {
            LOG(ERROR) << "Failed to set RKLLM chat template, error=" << ret;
            return -1;
        }
    }

    rknn_utils::logRkllmVersion();

    return 0;
}

void Session::cleanupVisionEncoder()
{
    if (encoder_.inputAttrs) {
        free(encoder_.inputAttrs);
        encoder_.inputAttrs = nullptr;
    }

    if (encoder_.outputAttrs) {
        free(encoder_.outputAttrs);
        encoder_.outputAttrs = nullptr;
    }

    if (encoder_.rknnContext) {
        rknn_destroy(encoder_.rknnContext);
        encoder_.rknnContext = 0;
    }
}

void Session::cleanupTextDecoder()
{
    if (decoder_.handle) {
        rkllm_destroy(decoder_.handle);
        decoder_.handle = nullptr;
    }
}

Session::Session(ModelConfig config)
    : config_(std::move(config))
{
}

Session::~Session()
{
    cleanupVisionEncoder();
    cleanupTextDecoder();
}

int Session::init()
{
    if (initTextDecoder() != 0) {
        return -1;
    }

    if (modelFamilyUsesVisionEncoder(config_.modelFamily)) {
        if (initVisionEncoder() != 0) {
            return -1;
        }
    }

    return 0;
}

const ModelConfig& Session::config() const noexcept
{
    return config_;
}

bool Session::isReady() const noexcept
{
    if (decoder_.handle == nullptr) {
        return false;
    }

    if (modelFamilyUsesVisionEncoder(config_.modelFamily)) {
        return encoder_.rknnContext != 0;
    }

    return true;
}

std::string Session::describe() const
{
    const auto& profile = modelProfileFor(config_.modelFamily);

    std::ostringstream stream;
    stream << "VLM RKNN session";
    stream << " target=" << VLM_RKNN_TARGET;
    stream << " model_family=" << modelFamilyName(config_.modelFamily);
    stream << " requires_vision_encoder=" << (profile.usesVisionEncoder ? "yes" : "no");
    stream << " vision_encoder_path="
           << (config_.visionEncoderPath.has_value() && !config_.visionEncoderPath->empty()
                   ? *config_.visionEncoderPath
                   : "<unset>");
    stream << " language_model_path="
           << (config_.languageModelPath.empty() ? "<unset>" : config_.languageModelPath);
    return stream.str();
}

bool Session::promptContainsImage(const std::string& prompt) const
{
    const auto& profile = modelProfileFor(config_.modelFamily);
    return profile.supportsMultimodal &&
        profile.imagePlaceholder[0] != '\0' &&
        prompt.find(profile.imagePlaceholder) != std::string::npos;
}

void Session::setOutputCallback(OutputCallback callback)
{
    outputCallback_ = std::move(callback);
}

int Session::encode(void* imgData, float* outResult)
{
    if (encoder_.rknnContext == 0) {
        LOG(ERROR) << "Vision encoder has not been initialized";
        return -1;
    }

    if (imgData == nullptr) {
        LOG(ERROR) << "Image input buffer is null";
        return -1;
    }

    if (outResult == nullptr) {
        LOG(ERROR) << "Image embedding output buffer is null";
        return -1;
    }

    if (encoder_.ioNum.n_input != 1) {
        LOG(ERROR) << "Unsupported RKNN input count: " << encoder_.ioNum.n_input;
        return -1;
    }

    const auto& inputAttr = encoder_.inputAttrs[0];
    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = 0;
    input.buf = imgData;
    input.size = inputAttr.n_elems;
    input.pass_through = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;

    int ret = rknn_inputs_set(encoder_.rknnContext, 1, &input);
    if (ret != RKNN_SUCC) {
        LOG(ERROR) << "Failed to set RKNN input, error=" << rknn_utils::rknnErrorMessage(ret);
        return ret;
    }

    ret = rknn_run(encoder_.rknnContext, nullptr);
    if (ret != RKNN_SUCC) {
        LOG(ERROR) << "Failed to run RKNN encoder, error=" << rknn_utils::rknnErrorMessage(ret);
        return ret;
    }

    std::vector<rknn_output> outputs(encoder_.ioNum.n_output);
    for (std::uint32_t i = 0; i < encoder_.ioNum.n_output; ++i) {
        memset(&outputs[i], 0, sizeof(rknn_output));
        outputs[i].index = i;
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 0;
    }

    ret = rknn_outputs_get(encoder_.rknnContext, encoder_.ioNum.n_output, outputs.data(), nullptr);
    if (ret != RKNN_SUCC) {
        LOG(ERROR) << "Failed to get RKNN output, error=" << rknn_utils::rknnErrorMessage(ret);
        return ret;
    }

    if (encoder_.ioNum.n_output == 1) {
        if (outputs[0].buf == nullptr) {
            LOG(ERROR) << "RKNN output buffer is null for index 0";
            ret = -1;
        } else if (encoder_.modelImageToken <= 0 || encoder_.modelEmbedSize <= 0) {
            LOG(ERROR) << "Invalid inferred RKNN output shape: tokens=" << encoder_.modelImageToken
                       << " embedSize=" << encoder_.modelEmbedSize;
            ret = -1;
        } else {
            const std::size_t expectedBytes = static_cast<std::size_t>(encoder_.modelImageToken)
                * static_cast<std::size_t>(encoder_.modelEmbedSize)
                * sizeof(float);
            if (outputs[0].size > expectedBytes) {
                LOG(ERROR) << "RKNN output buffer is larger than the expected embedding size: output_bytes="
                           << outputs[0].size
                           << " expected_bytes=" << expectedBytes;
            ret = -1;
        } else {
            memcpy(outResult, outputs[0].buf, outputs[0].size);
            }
        }
    } else {
        for (std::uint32_t i = 0; i < encoder_.ioNum.n_output; ++i) {
            int outputTokens = 0;
            int outputEmbedSize = 0;
            if (!inferEmbeddingShapeFromAttr(encoder_.outputAttrs[i], outputTokens, outputEmbedSize)) {
                LOG(ERROR) << "Could not infer embedding shape for RKNN output index " << i;
                ret = -1;
                break;
            }
            if (outputTokens != encoder_.modelImageToken ||
                outputEmbedSize != encoder_.modelEmbedSize) {
                LOG(ERROR) << "Unsupported RKNN output shape at index " << i
                           << ": expected tokens=" << encoder_.modelImageToken
                           << " embedSize=" << encoder_.modelEmbedSize
                           << " but got tokens=" << outputTokens
                           << " embedSize=" << outputEmbedSize;
                ret = -1;
                break;
            }
            if (outputs[i].buf == nullptr) {
                LOG(ERROR) << "RKNN output buffer is null for index " << i;
                ret = -1;
                break;
            }
        }

        if (ret == RKNN_SUCC) {
            const std::size_t outputCount = encoder_.ioNum.n_output;
            const std::size_t imageTokens = encoder_.modelImageToken;
            const std::size_t embedSize = encoder_.modelEmbedSize;
            for (std::size_t token = 0; token < imageTokens; ++token) {
                for (std::size_t outputIndex = 0; outputIndex < outputCount; ++outputIndex) {
                    const auto* output = static_cast<const float*>(outputs[outputIndex].buf);
                    memcpy(
                        outResult + token * outputCount * embedSize + outputIndex * embedSize,
                        output + token * embedSize,
                        embedSize * sizeof(float));
                }
            }
        }
    }

    int releaseRet = rknn_outputs_release(encoder_.rknnContext, encoder_.ioNum.n_output, outputs.data());
    if (releaseRet != RKNN_SUCC) {
        LOG(ERROR) << "Failed to release RKNN output buffers, error=" << rknn_utils::rknnErrorMessage(releaseRet);
        return releaseRet;
    }

    if (ret != RKNN_SUCC) {
        return ret;
    }

    return 0;
}

int Session::decode(const std::string& prompt, float* imgVec)
{
    const auto& profile = modelProfileFor(config_.modelFamily);

    lastDecodedText_.clear();

    RKLLMInferParam rkllmInferParams;
    memset(&rkllmInferParams, 0, sizeof(RKLLMInferParam));
    rkllmInferParams.mode = RKLLM_INFER_GENERATE;
    rkllmInferParams.keep_history = profile.useChatTemplate ? 1 : 0;

    RKLLMInput rkllmInput;
    memset(&rkllmInput, 0, sizeof(RKLLMInput));
    if (!promptContainsImage(prompt)) {
        rkllmInput.input_type = RKLLM_INPUT_PROMPT;
        rkllmInput.role = "user";
        rkllmInput.prompt_input = const_cast<char*>(prompt.c_str());
    } else {
        if (imgVec == nullptr) {
            LOG(ERROR) << "Prompt references " << profile.imagePlaceholder << " but no image embedding was provided";
            return -1;
        }
        const std::size_t imageTokens = encoder_.modelImageToken;
        const std::size_t imageHeight = encoder_.modelHeight;
        const std::size_t imageWidth = encoder_.modelWidth;
        if (imageTokens == 0 || imageHeight == 0 || imageWidth == 0) {
            LOG(ERROR) << "Vision encoder metadata is not available for multimodal decoding";
            return -1;
        }
        rkllmInput.input_type = RKLLM_INPUT_MULTIMODAL;
        rkllmInput.role = "user";
        rkllmInput.multimodal_input.prompt = const_cast<char*>(prompt.c_str());
        rkllmInput.multimodal_input.image_embed = imgVec;
        rkllmInput.multimodal_input.n_image_tokens = imageTokens;
        rkllmInput.multimodal_input.n_image = 1;
        rkllmInput.multimodal_input.image_height = imageHeight;
        rkllmInput.multimodal_input.image_width = imageWidth;
    }

    int ret = rkllm_run(decoder_.handle, &rkllmInput, &rkllmInferParams, this);
    if (ret != 0) {
        LOG(ERROR) << "Failed to run RKLLM, error=" << ret;
        return ret;
    }

    return 0;
}

int Session::callback(RKLLMResult* result, void* userdata, LLMCallState state)
{
    Session* session = static_cast<Session*>(userdata);

    if (state == RKLLM_RUN_FINISH) {
        LOG(INFO) << "RKLLM run finished";
        if (session->outputCallback_) {
            session->outputCallback_(nullptr, state);
        }
    } else if (state == RKLLM_RUN_ERROR) {
        LOG(ERROR) << "RKLLM run error";
        if (session->outputCallback_) {
            session->outputCallback_(nullptr, state);
        }
    } else if (state == RKLLM_RUN_NORMAL) {
        session->lastDecodedText_ += result->text;
        LOG(INFO) << session->lastDecodedText_;
        if (session->outputCallback_) {
            session->outputCallback_(result->text, state);
        }
    }

    return 0;
}

}  // namespace vlm_rknn
