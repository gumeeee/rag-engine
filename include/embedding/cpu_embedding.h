#pragma once

#include "embedding_service.h"
#include <memory>
#include <vector>
#include <string>

namespace rag_engine {

/**
 * CPU-only embedding service for development and CI environments.
 * Does not require CUDA.
 */
class CPUEmbeddingService : public IEmbeddingService {
public:
    explicit CPUEmbeddingService(int32_t dimension = 384);
    ~CPUEmbeddingService() override = default;

    std::future<Embedding> encode_async(const std::string& text) override;
    std::future<std::vector<Embedding>> encode_batch_async(
        const std::vector<std::string>& texts
    ) override;

    int32_t dimension() const override { return dimension_; }
    bool uses_gpu() const override { return false; }
    std::string name() const override { return "CPU"; }

private:
    Embedding encode(const std::string& text);

    int32_t dimension_{384};
};

}  // namespace rag_engine
