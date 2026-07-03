#pragma once

#include "common/types.h"
#include <memory>
#include <future>
#include <vector>
#include <string>

namespace rag_engine {

/**
 * Interface for embedding services.
 * Implementations can use GPU (ONNX + CUDA) or CPU fallback.
 */
class IEmbeddingService {
public:
    virtual ~IEmbeddingService() = default;

    /**
     * Encode a single text into an embedding vector.
     * Returns a future that resolves to the embedding.
     */
    virtual std::future<Embedding> encode_async(const std::string& text) = 0;

    /**
     * Encode a batch of texts into embedding vectors.
     * This is the performance-critical path for batch processing.
     */
    virtual std::future<std::vector<Embedding>> encode_batch_async(
        const std::vector<std::string>& texts
    ) = 0;

    /**
     * Get the embedding dimension.
     */
    virtual int32_t dimension() const = 0;

    /**
     * Check if CUDA/GPU is being used.
     */
    virtual bool uses_gpu() const = 0;

    /**
     * Get service name for logging.
     */
    virtual std::string name() const = 0;
};

}  // namespace rag_engine
