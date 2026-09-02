// Copyright (c) 2026 Tristan Penman
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

#include "logger.h"
#include "vlm_rknn.h"

namespace {

void printUsage(const char* program)
{
    std::cout << "Usage: " << program
              << " [-v|--verbose] [--cores <num_cores>]"
              << " [--model-family <qwen2-vl|qwen2.5-vl|qwen3-vl|llama|smolvlm2|gemma3>]"
              << " --llm <language_model_path> --prompt <prompt> --output <cache_path>\n";
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

bool parseCores(const char* value, int& parsed)
{
    errno = 0;
    char* end = nullptr;
    const long result = std::strtol(value, &end, 10);
    if (value == end || *end != '\0' || errno == ERANGE || result < 1 || result > 3) {
        std::cout << "Invalid value for --cores: " << value << " (expected 1-3)\n";
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
    std::optional<vlm_rknn::ModelFamily> modelFamily;
    std::optional<std::string> languageModelPath;
    std::optional<std::string> prompt;
    std::optional<std::string> outputPath;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            Logger::configure(std::cout, Logger::Level::kVerbose);
            continue;
        }

        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }

        const char* value = nullptr;
        if (strcmp(argv[i], "--cores") == 0) {
            if (!getOptionValue(argc, argv, i, "--cores", value)) {
                return 1;
            }

            int parsed = 0;
            if (!parseCores(value, parsed)) {
                return 1;
            }

            numCores = parsed;
            continue;
        }

        if (strcmp(argv[i], "--model-family") == 0) {
            if (!getOptionValue(argc, argv, i, "--model-family", value)) {
                return 1;
            }

            vlm_rknn::ModelFamily parsed;
            if (!vlm_rknn::parseModelFamily(value, parsed)) {
                std::cout << "Invalid model family specified: " << value << "\n";
                printUsage(argv[0]);
                return 1;
            }
            modelFamily = parsed;
            continue;
        }

        if (strcmp(argv[i], "--llm") == 0) {
            if (!getOptionValue(argc, argv, i, "--llm", value)) {
                return 1;
            }

            languageModelPath = value;
            continue;
        }

        if (strcmp(argv[i], "--prompt") == 0) {
            if (!getOptionValue(argc, argv, i, "--prompt", value)) {
                return 1;
            }

            prompt = value;
            continue;
        }

        if (strcmp(argv[i], "--output") == 0) {
            if (!getOptionValue(argc, argv, i, "--output", value)) {
                return 1;
            }

            outputPath = value;
            continue;
        }

        std::cout << "Unexpected positional argument or unknown option: " << argv[i] << "\n";
        printUsage(argv[0]);
        return 1;
    }

    if (!languageModelPath.has_value() || languageModelPath->empty()) {
        std::cout << "Missing required --llm <language_model_path> argument\n";
        printUsage(argv[0]);
        return 1;
    }

    if (!prompt.has_value() || prompt->empty()) {
        std::cout << "Missing required non-empty --prompt <prompt> argument\n";
        printUsage(argv[0]);
        return 1;
    }

    if (!outputPath.has_value() || outputPath->empty()) {
        std::cout << "Missing required non-empty --output <cache_path> argument\n";
        printUsage(argv[0]);
        return 1;
    }

    vlm_rknn::ModelConfig config;
    config.languageModelPath = *languageModelPath;
    if (modelFamily.has_value()) {
        config.modelFamily = *modelFamily;
    }

    if (numCores.has_value()) {
        config.numCores = *numCores;
    }

    // Cache creation still invokes generation; keep the discarded response as
    // short as the currently vendored RKLLM API permits.
    config.maxNewTokens = 1;

    vlm_rknn::Session session(config);
    if (session.init() != 0 || !session.isReady()) {
        LOG(ERROR) << "Session initialization failed.";
        return 1;
    }

    LOG(INFO) << "Generating prompt cache at " << *outputPath;
    const int ret = session.generatePromptCache(*prompt, *outputPath);
    if (ret != 0) {
        return 1;
    }

    std::error_code fileError;
    const auto cacheSize = std::filesystem::file_size(*outputPath, fileError);
    if (fileError || cacheSize == 0) {
        LOG(ERROR) << "RKLLM returned success but did not create a non-empty cache at "
                   << *outputPath;
        return 1;
    }

    LOG(INFO) << "Prompt cache generated successfully (" << cacheSize << " bytes).";
    return 0;
}
