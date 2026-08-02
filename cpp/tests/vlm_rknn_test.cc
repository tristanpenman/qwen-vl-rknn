#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "vlm_rknn.h"

namespace {

void expect(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main()
{
    vlm_rknn::Session emptySession({});
    expect(!emptySession.isReady(), "empty session should not be ready");

    vlm_rknn::ModelConfig config;
    expect(config.modelFamily == vlm_rknn::ModelFamily::kQwenVL2,
           "default model family should be Qwen2-VL");
    config.visionEncoderPath = "/models/qwen-vl/vision_encoder.rknn";
    config.languageModelPath = "/models/qwen-vl/language_model.rkllm";
    vlm_rknn::Session session(config);

    expect(!session.isReady(), "session should not be ready before init");
    expect(session.config().visionEncoderPath.has_value(), "vision encoder path should be set");
    expect(
        *session.config().visionEncoderPath == "/models/qwen-vl/vision_encoder.rknn",
        "vision encoder path should be retained");
    expect(
        session.config().languageModelPath == "/models/qwen-vl/language_model.rkllm",
        "language model path should be retained");
    expect(session.describe().find("VLM RKNN session") != std::string::npos,
           "description should contain project name");
    expect(session.describe().find("target=") != std::string::npos, "description should contain target device");
    expect(session.describe().find("model_family=qwen2-vl") != std::string::npos,
           "description should contain model family");
    expect(session.describe().find("requires_vision_encoder=yes") != std::string::npos,
           "description should contain vision encoder requirement");

    vlm_rknn::ModelFamily family;
    expect(vlm_rknn::parseModelFamily("llama", family), "llama should parse");
    expect(family == vlm_rknn::ModelFamily::kLlama, "llama should map to Llama family");
    expect(vlm_rknn::parseModelFamily("smol-vlm2", family), "smol-vlm2 alias should parse");
    expect(family == vlm_rknn::ModelFamily::kSmolVLM2, "smol-vlm2 should map to SmolVLM2 family");
    expect(vlm_rknn::parseModelFamily("gemma3", family), "gemma3 should parse");
    expect(family == vlm_rknn::ModelFamily::kGemma3, "gemma3 should map to Gemma 3 family");
    expect(vlm_rknn::parseModelFamily("gemma-3", family), "gemma-3 alias should parse");
    expect(vlm_rknn::parseModelFamily("gemma_3", family), "gemma_3 alias should parse");
    expect(std::string(vlm_rknn::modelFamilyName(family)) == "gemma3",
           "Gemma 3 should have a canonical family name");
    expect(!vlm_rknn::parseModelFamily("not-a-family", family), "unknown family should not parse");

    expect(!vlm_rknn::modelFamilyUsesVisionEncoder(vlm_rknn::ModelFamily::kLlama),
           "llama should not require a vision encoder");
    expect(!vlm_rknn::modelFamilySupportsMultimodal(vlm_rknn::ModelFamily::kLlama),
           "llama should not support multimodal input");
    expect(vlm_rknn::modelFamilyUsesVisionEncoder(vlm_rknn::ModelFamily::kQwenVL2),
           "Qwen2-VL should require a vision encoder");
    expect(vlm_rknn::modelFamilyUsesVisionEncoder(vlm_rknn::ModelFamily::kSmolVLM2),
           "SmolVLM2 should require a vision encoder");
    expect(vlm_rknn::modelFamilySupportsMultimodal(vlm_rknn::ModelFamily::kSmolVLM2),
           "SmolVLM2 should be registered as multimodal");
    expect(vlm_rknn::modelFamilyUsesVisionEncoder(vlm_rknn::ModelFamily::kGemma3),
           "Gemma 3 should require a vision encoder");
    expect(vlm_rknn::modelFamilySupportsMultimodal(vlm_rknn::ModelFamily::kGemma3),
           "Gemma 3 should be registered as multimodal");
    expect(std::string(vlm_rknn::modelFamilyImagePlaceholder(vlm_rknn::ModelFamily::kGemma3)) == "<image>",
           "Gemma 3 should expose the repository image placeholder");
    expect(std::string(vlm_rknn::modelFamilyImagePlaceholder(vlm_rknn::ModelFamily::kQwenVL2)) == "<image>",
           "Qwen2-VL image placeholder should be exposed");
    expect(std::string(vlm_rknn::modelFamilyImagePlaceholder(vlm_rknn::ModelFamily::kLlama)).empty(),
           "llama image placeholder should be empty");

    expect(session.promptContainsImage("<image>Describe this."), "Qwen2-VL should detect its image placeholder");
    expect(!session.promptContainsImage("Describe this."), "Qwen2-VL should allow plain text prompts");

    const auto& qwenPreprocess =
        vlm_rknn::modelFamilyImagePreprocessProfile(vlm_rknn::ModelFamily::kQwenVL2);
    expect(qwenPreprocess.resizeMode == vlm_rknn::ResizeMode::kPadToSquare,
           "Qwen2-VL should pad images to square");
    expect(qwenPreprocess.rgb, "Qwen2-VL should request RGB encoder input");
    expect(!qwenPreprocess.normalizeInHost, "Qwen2-VL should not normalize in host code");

    const auto& smolPreprocess =
        vlm_rknn::modelFamilyImagePreprocessProfile(vlm_rknn::ModelFamily::kSmolVLM2);
    expect(smolPreprocess.resizeMode == vlm_rknn::ResizeMode::kPadToSquare,
           "SmolVLM2 profile should have an explicit resize mode");

    const auto& gemmaPreprocess =
        vlm_rknn::modelFamilyImagePreprocessProfile(vlm_rknn::ModelFamily::kGemma3);
    expect(gemmaPreprocess.resizeMode == vlm_rknn::ResizeMode::kStretch,
           "Gemma 3 should resize to the fixed encoder dimensions");
    expect(gemmaPreprocess.rgb, "Gemma 3 should request RGB encoder input");
    expect(!gemmaPreprocess.normalizeInHost,
           "Gemma 3 normalization should be included in the RKNN artifact");

    cv::Mat bgrPixel(1, 1, CV_8UC3, cv::Scalar(10, 20, 30));
    cv::Mat preprocessedPixel;
    expect(vlm_rknn::preprocessImageForVisionEncoder(
               vlm_rknn::ModelFamily::kQwenVL2,
               bgrPixel,
               cv::Size(1, 1),
               preprocessedPixel) == 0,
           "Qwen2-VL preprocessing should succeed");
    expect(preprocessedPixel.cols == 1 && preprocessedPixel.rows == 1,
           "preprocessed image should match requested dimensions");
    const cv::Vec3b rgbPixel = preprocessedPixel.at<cv::Vec3b>(0, 0);
    expect(rgbPixel[0] == 30 && rgbPixel[1] == 20 && rgbPixel[2] == 10,
           "Qwen2-VL preprocessing should convert BGR to RGB");

    cv::Mat wideBgr(1, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat padded;
    expect(vlm_rknn::preprocessImageForVisionEncoder(
               vlm_rknn::ModelFamily::kQwenVL2,
               wideBgr,
               cv::Size(2, 2),
               padded) == 0,
           "Qwen2-VL preprocessing should pad non-square images");
    expect(padded.cols == 2 && padded.rows == 2, "padded image should match requested dimensions");
    expect(padded.at<cv::Vec3b>(1, 0)[0] >= 127 && padded.at<cv::Vec3b>(1, 0)[0] <= 128,
           "Qwen2-VL padding should use the profile background color");

    vlm_rknn::ModelConfig llamaConfig;
    llamaConfig.modelFamily = vlm_rknn::ModelFamily::kLlama;
    llamaConfig.languageModelPath = "/models/llama.rkllm";
    vlm_rknn::Session llamaSession(llamaConfig);
    expect(!llamaSession.isReady(), "llama session should not be ready before init");
    expect(llamaSession.describe().find("model_family=llama") != std::string::npos,
           "description should contain llama model family");
    expect(llamaSession.describe().find("requires_vision_encoder=no") != std::string::npos,
           "description should report llama does not require a vision encoder");
    expect(!llamaSession.promptContainsImage("<image>Describe this."),
           "llama should not treat image placeholders as multimodal prompts");

    // INI-based model configuration.
    {
        const std::string ini =
            "[gemma3-4b]\n"
            "model_family=gemma3\n"
            "vision=/models/gemma3/gemma3-vision-projector.rknn\n"
            "llm=/models/gemma3/gemma3-4b-it.rkllm\n";
        std::vector<vlm_rknn::NamedModelConfig> configs;
        std::string error;
        expect(vlm_rknn::parseModelConfigsFromIni(ini, configs, error),
               "Gemma 3 INI should parse: " + error);
        expect(configs.size() == 1 && configs[0].config.modelFamily == vlm_rknn::ModelFamily::kGemma3,
               "Gemma 3 INI should retain its model family");
        vlm_rknn::Session gemmaSession(configs[0].config);
        expect(gemmaSession.describe().find("model_family=gemma3") != std::string::npos,
               "Gemma 3 session description should name the family");
        expect(gemmaSession.promptContainsImage("<image>Describe this."),
               "Gemma 3 should detect its image placeholder");
    }

    {
        const std::string ini =
            "[qwen]\n"
            "model_family=qwen2-vl\n"
            "vision=/models/qwen/vision.rknn\n"
            "llm=/models/qwen/language.rkllm\n"
            "max_new_tokens=300\n"
            "\n"
            "; a text-only model\n"
            "[chat]\n"
            "model_family=llama\n"
            "llm=/models/llama.rkllm\n"
            "cores=2\n";

        std::vector<vlm_rknn::NamedModelConfig> configs;
        std::string error;
        expect(vlm_rknn::parseModelConfigsFromIni(ini, configs, error),
               "valid INI should parse: " + error);
        expect(configs.size() == 2, "two models should be parsed");
        expect(configs[0].modelId == "qwen", "first model should be the default 'qwen'");
        expect(configs[0].config.modelFamily == vlm_rknn::ModelFamily::kQwenVL2,
               "qwen model family should parse");
        expect(configs[0].config.visionEncoderPath.has_value() &&
                   *configs[0].config.visionEncoderPath == "/models/qwen/vision.rknn",
               "qwen vision path should be retained");
        expect(configs[0].config.languageModelPath == "/models/qwen/language.rkllm",
               "qwen llm path should be retained");
        expect(configs[0].config.maxNewTokens == 300, "qwen max new tokens should be 300");
        expect(configs[1].modelId == "chat", "second model should be 'chat'");
        expect(configs[1].config.modelFamily == vlm_rknn::ModelFamily::kLlama,
               "chat model family should be llama");
        expect(!configs[1].config.visionEncoderPath.has_value(),
               "llama model should have no vision path");
        expect(configs[1].config.numCores.has_value() && *configs[1].config.numCores == 2,
               "chat cores should be 2");
    }

    {
        std::vector<vlm_rknn::NamedModelConfig> configs;
        std::string error;
        expect(!vlm_rknn::parseModelConfigsFromIni("", configs, error),
               "empty INI should fail");
        expect(!vlm_rknn::parseModelConfigsFromIni(
                   "[m]\nmodel_family=qwen2-vl\nvision=/v.rknn\n", configs, error),
               "missing llm should fail");
        expect(!vlm_rknn::parseModelConfigsFromIni(
                   "[m]\nmodel_family=gemma3\nllm=/m.rkllm\n", configs, error),
               "Gemma 3 without a vision model should fail");
        expect(!vlm_rknn::parseModelConfigsFromIni(
                   "[m]\nllm=/m.rkllm\n", configs, error),
               "vision model without vision encoder should fail");
        expect(!vlm_rknn::parseModelConfigsFromIni(
                   "[m]\nmodel_family=llama\nllm=/m.rkllm\nvision=/v.rknn\n", configs, error),
               "vision key for llama should fail");
        expect(!vlm_rknn::parseModelConfigsFromIni(
                   "[m]\nmodel_family=llama\nllm=/m.rkllm\nbogus=1\n", configs, error),
               "unknown key should fail");
        expect(!vlm_rknn::parseModelConfigsFromIni(
                   "[m]\nmodel_family=llama\nllm=/m.rkllm\ncores=5\n", configs, error),
               "out-of-range cores should fail");
        expect(!vlm_rknn::parseModelConfigsFromIni(
                   "[m]\nllm=/a.rkllm\nmodel_family=llama\n[m]\nllm=/b.rkllm\nmodel_family=llama\n",
                   configs, error),
               "duplicate section should fail");
    }

    {
        // A text-only single-model INI should parse and default correctly.
        std::vector<vlm_rknn::NamedModelConfig> configs;
        std::string error;
        expect(vlm_rknn::parseModelConfigsFromIni(
                   "[only]\nmodel_family=llama\nllm=/m.rkllm\n", configs, error),
               "single llama model should parse: " + error);
        expect(configs.size() == 1 && configs[0].modelId == "only",
               "single model should be parsed and named");
        expect(configs[0].config.maxNewTokens == 128 && configs[0].config.maxContextLen == 2048,
               "defaults should apply when keys are omitted");
    }

    std::cout << "placeholder tests passed" << '\n';
    return 0;
}
