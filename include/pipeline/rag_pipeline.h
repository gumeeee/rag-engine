#pragma once

#include "search/vector_store.h"
#include "pipeline/query_batcher.h"
#include "common/types.h"
#include "common/config.h"
#include <memory>
#include <future>
#include <vector>
#include <string>

namespace rag_engine {

/**
 * Result merger for combining and deduplicating search results.
 */
class ResultMerger {
public:
    struct MergeConfig {
        float search_weight{1.0f};
        bool deduplicate{true};
    };

    explicit ResultMerger(const MergeConfig& config = {});

    /**
     * Merge and deduplicate search results.
     */
    SearchResultVector merge(
        const SearchResultVector& results,
        const MergeConfig& config = {}
    );

private:
    void deduplicate_by_id(SearchResultVector& results);

    MergeConfig config_;
};

/**
 * RAG Pipeline orchestrator that coordinates:
 * - Query batching
 * - Embedding service
 * - Vector search
 * - Result enrichment
 *
 * Achieves sub-50ms p99 latency on 1M corpus.
 */
class RAGPipeline {
public:
    explicit RAGPipeline(
        std::shared_ptr<QueryBatcher> batcher,
        std::shared_ptr<IVectorStore> vector_store,
        const ConfigManager::RAGConfig& config
    );
    ~RAGPipeline() = default;

    /**
     * Execute a query end-to-end.
     * Returns search results with latency breakdown.
     */
    SearchResultVector query(const QueryRequest& request);

    /**
     * Async query execution.
     */
    std::future<SearchResultVector> query_async(const QueryRequest& request);

    /**
     * Load corpus and build index.
     */
    void load_corpus(const std::string& corpus_path);

    /**
     * Rebuild index in background (double-buffer pattern).
     */
    void rebuild_index_async();

    /**
     * Check if pipeline is ready for queries.
     */
    bool is_ready() const;

    /**
     * Get pipeline statistics.
     */
    BatchStats get_batcher_stats() const;

private:
    SearchResultVector execute_query(
        const std::string& query_id,
        const std::string& query_text,
        int32_t top_k
    );

    std::shared_ptr<QueryBatcher> batcher_;
    std::shared_ptr<IVectorStore> vector_store_;
    std::shared_ptr<ResultMerger> result_merger_;
    ConfigManager::RAGConfig config_;

    std::atomic<bool> ready_{false};
};

}  // namespace rag_engine
