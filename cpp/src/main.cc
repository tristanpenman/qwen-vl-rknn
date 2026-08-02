// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// Copyright (c) 2026 Tristan Penman
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "logger.h"
#include "vlm_rknn.h"

namespace {

void printUsage(const char* program)
{
    std::cout << "Usage: " << program
              << " [-v|--verbose] [--cores <num_cores>]"
              << " [--model-family <qwen2-vl|qwen2.5-vl|qwen3-vl|llama|smolvlm2|gemma3>]"
              << " [--max-new-tokens <tokens>] [--max-context-len <tokens>]"
              << " --llm <language_model_path>"
              << " [--vision <vision_encoder_path> --image <image_path>]"
              << " [--prompt <prompt>]\n";
    std::cout << "--vision and --image are required for vision model families and unsupported for llama.\n";
    std::cout << "If --prompt is omitted, an interactive REPL is started.\n";
}

bool getOptionValue(int argc, char** argv, int& index, const char* option, const char*& value)
{
    if (index + 1 >= argc) {
        std::cout << option << " option requires an argument\n";
        return false;
    }
    value = argv[++index];
    return true;
}

bool parseIntOption(const char* option, const char* value, int minValue, int maxValue, int& parsed)
{
    errno = 0;
    char* end = nullptr;
    const long result = std::strtol(value, &end, 10);
    if (value == end || *end != '\0' || errno == ERANGE || result < minValue || result > maxValue) {
        std::cout << "Invalid value for " << option << ": " << value
                  << " (expected " << minValue << "-" << maxValue << ")\n";
        return false;
    }

    parsed = static_cast<int>(result);
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    Logger::configure(std::cout);

    std::optional<int> numCores;
    std::optional<int> maxNewTokens;
    std::optional<int> maxContextLen;
    std::optional<vlm_rknn::ModelFamily> modelFamily;
    std::optional<std::string> visionEncoderPath;
    std::optional<std::string> languageModelPath;
    std::optional<std::string> imagePath;
    std::optional<std::string> prompt;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            Logger::configure(std::cout, Logger::Level::kVerbose);
            continue;
        }
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "--cores") == 0) {
            const char* value = nullptr;
            if (!getOptionValue(argc, argv, i, "--cores", value)) {
                return -1;
            }
            int parsed = 0;
            if (!parseIntOption("--cores", value, 1, 3, parsed)) {
                return -1;
            }
            numCores = parsed;
            continue;
        }
        if (strcmp(argv[i], "--model-family") == 0) {
            const char* value = nullptr;
            if (!getOptionValue(argc, argv, i, "--model-family", value)) {
                std::cout << "--model-family option requires one of: "
                          << "qwen2-vl, qwen2.5-vl, qwen3-vl, llama, smolvlm2, gemma3\n";
                return -1;
            }
            vlm_rknn::ModelFamily parsedFamily;
            if (!vlm_rknn::parseModelFamily(value, parsedFamily)) {
                std::cout << "Invalid model family specified: " << value << "\n";
                std::cout << "Expected one of: qwen2-vl, qwen2.5-vl, qwen3-vl, llama, smolvlm2, gemma3\n";
                return -1;
            }
            modelFamily = parsedFamily;
            continue;
        }
        if (strcmp(argv[i], "--max-new-tokens") == 0) {
            const char* value = nullptr;
            if (!getOptionValue(argc, argv, i, "--max-new-tokens", value)) {
                return -1;
            }
            int parsed = 0;
            if (!parseIntOption("--max-new-tokens", value, 1, INT_MAX, parsed)) {
                return -1;
            }
            maxNewTokens = parsed;
            continue;
        }
        if (strcmp(argv[i], "--max-context-len") == 0) {
            const char* value = nullptr;
            if (!getOptionValue(argc, argv, i, "--max-context-len", value)) {
                return -1;
            }
            int parsed = 0;
            if (!parseIntOption("--max-context-len", value, 1, INT_MAX, parsed)) {
                return -1;
            }
            maxContextLen = parsed;
            continue;
        }
        if (strcmp(argv[i], "--vision") == 0) {
            const char* value = nullptr;
            if (!getOptionValue(argc, argv, i, "--vision", value)) {
                return -1;
            }
            visionEncoderPath = value;
            continue;
        }
        if (strcmp(argv[i], "--llm") == 0) {
            const char* value = nullptr;
            if (!getOptionValue(argc, argv, i, "--llm", value)) {
                return -1;
            }
            languageModelPath = value;
            continue;
        }
        if (strcmp(argv[i], "--image") == 0) {
            const char* value = nullptr;
            if (!getOptionValue(argc, argv, i, "--image", value)) {
                return -1;
            }
            imagePath = value;
            continue;
        }
        if (strcmp(argv[i], "--prompt") == 0) {
            const char* value = nullptr;
            if (!getOptionValue(argc, argv, i, "--prompt", value)) {
                return -1;
            }
            prompt = value;
            continue;
        }

        std::cout << "Unexpected positional argument or unknown option: " << argv[i] << "\n";
        printUsage(argv[0]);
        return -1;
    }

    if (!languageModelPath.has_value() || languageModelPath->empty()) {
        std::cout << "Missing required --llm <language_model_path> argument\n";
        printUsage(argv[0]);
        return 1;
    }

    vlm_rknn::ModelConfig config;
    config.languageModelPath = *languageModelPath;
    if (modelFamily.has_value()) {
        config.modelFamily = modelFamily.value();
    }
    const bool usesVisionEncoder = vlm_rknn::modelFamilyUsesVisionEncoder(config.modelFamily);
    if (usesVisionEncoder) {
        if (!visionEncoderPath.has_value() || visionEncoderPath->empty()) {
            std::cout << "Missing required --vision <vision_encoder_path> argument for "
                      << vlm_rknn::modelFamilyName(config.modelFamily) << "\n";
            printUsage(argv[0]);
            return 1;
        }
        if (!imagePath.has_value() || imagePath->empty()) {
            std::cout << "Missing required --image <image_path> argument for "
                      << vlm_rknn::modelFamilyName(config.modelFamily) << "\n";
            printUsage(argv[0]);
            return 1;
        }
        config.visionEncoderPath = *visionEncoderPath;
    } else {
        if (visionEncoderPath.has_value() && !visionEncoderPath->empty()) {
            std::cout << "--vision is not supported for "
                      << vlm_rknn::modelFamilyName(config.modelFamily) << "\n";
            printUsage(argv[0]);
            return 1;
        }
        if (imagePath.has_value() && !imagePath->empty()) {
            std::cout << "--image is not supported for "
                      << vlm_rknn::modelFamilyName(config.modelFamily) << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }
    if (numCores.has_value()) {
        config.numCores = numCores.value();
    }
    if (maxNewTokens.has_value()) {
        config.maxNewTokens = maxNewTokens.value();
    }
    if (maxContextLen.has_value()) {
        config.maxContextLen = maxContextLen.value();
    }

    vlm_rknn::Session session(config);

    if (session.init() != 0) {
        LOG(ERROR) << "Session initialization failed.";
        return -1;
    }

    if (!session.isReady()) {
        LOG(ERROR) << "Session is not ready. Something went wrong.";
        return -1;
    }

    LOG(INFO) << "Session initialized successfully.";
    LOG(INFO) << session.describe();

    if (!usesVisionEncoder) {
        int ret = 0;
        if (prompt.has_value()) {
            LOG(INFO) << "Running decoder with prompt: " << *prompt;
            ret = session.decode(*prompt, nullptr);
            if (ret != 0) {
                LOG(ERROR) << "Failed to run decoder with prompt, error=" << ret;
                return -1;
            }
        } else {
            LOG(INFO) << "No prompt provided; starting interactive REPL.";
            LOG(INFO) << "Type ':quit' or ':exit' to end the session.";
            std::string replPrompt;
            while (true) {
                std::cout << "> " << std::flush;
                if (!std::getline(std::cin, replPrompt)) {
                    LOG(INFO) << "EOF received, exiting REPL.";
                    break;
                }
                if (replPrompt == ":quit" || replPrompt == ":exit") {
                    LOG(INFO) << "Exiting REPL.";
                    break;
                }
                if (replPrompt.empty()) {
                    continue;
                }
                ret = session.decode(replPrompt, nullptr);
                if (ret != 0) {
                    LOG(ERROR) << "Failed to run decoder with prompt, error=" << ret;
                    return -1;
                }
            }
        }
        return 0;
    }

    // Load the image using OpenCV.
    cv::Mat image = cv::imread(*imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        LOG(ERROR) << "Could not open or find the image: " << *imagePath;
        return -1;
    }

    // Get image dimensions from the model configuration.
    const auto& encoder = session.visionEncoder();
    auto imageWidth = encoder.modelWidth;
    auto imageHeight = encoder.modelHeight;

    cv::Mat resizedImg;
    LOG(INFO) << "Preprocessing image to " << imageWidth << "x" << imageHeight;
    int ret = vlm_rknn::preprocessImageForVisionEncoder(
        config.modelFamily,
        image,
        cv::Size(imageWidth, imageHeight),
        resizedImg);
    if (ret != 0) {
        LOG(ERROR) << "Failed to preprocess image, error=" << ret;
        return -1;
    }

    // Determine size of the embedding that will be passed to the decoder.
    auto imageTokens = encoder.modelImageToken;
    auto imageEmbedLen = encoder.modelEmbedSize;
    auto embedOutputCount = encoder.ioNum.n_output;
    auto rkllmImageEmbedLen = imageTokens * imageEmbedLen * embedOutputCount;
    LOG(INFO) << "Image embedding size: " << rkllmImageEmbedLen;

    // Allocate memory for the image embedding output.
    std::vector<float> imgVec;
    imgVec.resize(rkllmImageEmbedLen);

    LOG(INFO) << "Running encoder...";
    ret = session.encode(resizedImg.data, imgVec.data());
    if (ret != 0) {
        LOG(ERROR) << "Failed to run model, error=" << ret;
        return -1;
    }

    LOG(INFO) << "Encoder ran successfully!";
    if (prompt.has_value()) {
        LOG(INFO) << "Running decoder with prompt: " << *prompt;
        ret = session.decode(*prompt, imgVec.data());
        if (ret != 0) {
            LOG(ERROR) << "Failed to run decoder with prompt, error=" << ret;
            return -1;
        }
    } else {
        LOG(INFO) << "No prompt provided; starting interactive REPL.";
        LOG(INFO) << "Type ':quit' or ':exit' to end the session.";
        std::string prompt;
        while (true) {
            std::cout << "> " << std::flush;
            if (!std::getline(std::cin, prompt)) {
                LOG(INFO) << "EOF received, exiting REPL.";
                break;
            }
            if (prompt == ":quit" || prompt == ":exit") {
                LOG(INFO) << "Exiting REPL.";
                break;
            }
            if (prompt.empty()) {
                continue;
            }
            ret = session.decode(prompt, imgVec.data());
            if (ret != 0) {
                LOG(ERROR) << "Failed to run decoder with prompt, error=" << ret;
                return -1;
            }
        }
    }

    return 0;
}
