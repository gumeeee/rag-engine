#include "embedding/cpu_embedding.h"
#include "common/cuda_utils.h"
#include <spdlog/spdlog.h>
#include <cstring>
#include <cmath>

namespace rag_engine {

CPUEmbeddingService::CPUEmbeddingService(int32_t dimension) : dimension_(dimension) {
    SPDLOG_INFO("Initializing CPU embedding service with dimension {}", dimension_);
}

Embedding CPUEmbeddingService::encode(const std::string& text) {
    // Simple deterministic embedding based on text hash
    // In production, use proper tokenizer and model inference
    Embedding emb(dimension_);

    // Use text content to seed pseudo-random but deterministic values
    size_t seed = 0;
    for (unsigned char c : text) {
        seed = seed * 31 + c;
    }

    // Generate pseudo-random values based on seed
    srand(static_cast<unsigned>(seed));
    for (int32_t i = 0; i < dimension_; ++i) {
        // Box-Muller transform for normal distribution
        float u1 = static_cast<float>(rand()) / RAND_MAX;
        float u2 = static_cast<float>(rand()) / RAND_MAX;
        float z = std::sqrt(-2.0f * std::log(u1 + 1e-10f)) * std::cos(2.0f * M_PI * u2);
        emb.data[i] = z;
    }

    // L2 normalize
    float norm = 0.0f;
    for (float v : emb.data) {
        norm += v * v;
    }
    norm = std::sqrt(norm) + 1e-10f;
    for (float& v : emb.data) {
        v /= norm;
    }

    return emb;
}

std::future<Embedding> CPUEmbeddingService::encode_async(const std::string& text) {
    return encode_batch_async({text}).then([](std::vector<Embedding> embeddings) {
        return std::move(embeddings[0]);
    });
}

std::future<std::vector<Embedding>> CPUEmbeddingService::encode_batch_async(
    const std::vector<std::string>& texts
) {
    std::promise<std::vector<Embedding>> promise;

    if (texts.empty()) {
        promise.set_value({});
        return promise.get_future();
    }

    try {
        std::vector<Embedding> results;
        results.reserve(texts.size());

        for (const auto& text : texts) {
            results.push_back(encode(text));
        }

        promise.set_value(std::move(results));
    } catch (const std::exception& e) {
        SPDLOG_ERROR("CPU embedding failed: {}", e.what());
        promise.set_exception(std::current_exception());
    }

    return promise.get_future();
}

}  // namespace rag_engine
