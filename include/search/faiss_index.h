#pragma once

#include "search/vector_store.h"
#include "common/config.h"
#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace rag_engine {

/**
 * FAISS HNSW index implementation for approximate nearest neighbor search.
 * Uses M=32, efConstruction=40, efSearch=64 as per spec.
 */
class FAISSIndex : public IVectorStore {
public:
    explicit FAISSIndex(const ConfigManager::RAGConfig::SearchConfig& config);
    ~FAISSIndex() override;

    void load(const std::string& path) override;
    void save(const std::string& path) override;

    SearchResultVector search(
        const Embedding& query,
        int32_t k,
        int32_t ef_search = 64
    ) override;

    size_t size() const override;
    int32_t dimension() const override { return dimension_; }
    bool is_loaded() const override { return index_ != nullptr && !metadata_.empty(); }

    /**
     * Add documents to the index.
     */
    void add(const std::vector<DocumentChunk>& chunks, const std::vector<Embedding>& embeddings);

    /**
     * Build index from existing vectors.
     */
    void build_index();

private:
    void load_corpus(const std::string& path);

    ConfigManager::RAGConfig::SearchConfig config_;
    int32_t dimension_{384};
    faiss::idx_t num_vectors_{0};

    // FAISS HNSW index
    std::unique_ptr<faiss::Index> index_;

    // Metadata for result enrichment
    std::vector<DocumentChunk> metadata_;
    std::unordered_map<faiss::idx_t, size_t> id_to_index_;
};

}  // namespace rag_engine
