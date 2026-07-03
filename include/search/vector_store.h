#pragma once

#include "common/types.h"
#include <memory>
#include <string>
#include <vector>

namespace rag_engine {

/**
 * Interface for vector store implementations.
 * Allows swapping between different index types.
 */
class IVectorStore {
public:
    virtual ~IVectorStore() = default;

    /**
     * Load index from file.
     */
    virtual void load(const std::string& path) = 0;

    /**
     * Save index to file.
     */
    virtual void save(const std::string& path) = 0;

    /**
     * Search for nearest neighbors.
     */
    virtual SearchResultVector search(
        const Embedding& query,
        int32_t k,
        int32_t ef_search = 64
    ) = 0;

    /**
     * Get number of indexed vectors.
     */
    virtual size_t size() const = 0;

    /**
     * Get embedding dimension.
     */
    virtual int32_t dimension() const = 0;

    /**
     * Check if index is ready.
     */
    virtual bool is_loaded() const = 0;
};

}  // namespace rag_engine
