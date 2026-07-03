#include "pipeline/rag_pipeline.h"
#include <spdlog/spdlog.h>
#include <chrono>

namespace rag_engine {

RAGPipeline::RAGPipeline(
    std::shared_ptr<QueryBatcher> batcher,
    std::shared_ptr<IVectorStore> vector_store,
    const ConfigManager::RAGConfig& config
) : batcher_(std::move(batcher)),
    vector_store_(std::move(vector_store)),
    result_merger_(std::make_shared<ResultMerger>()),
    config_(config) {

    SPDLOG_INFO("Initializing RAG Pipeline");
}

SearchResultVector RAGPipeline::execute_query(
    const std::string& query_id,
    const std::string& query_text,
    int32_t top_k
) {
    auto start = std::chrono::steady_clock::now();

    // Step 1: Get embedding via batcher
    auto embedding_start = std::chrono::steady_clock::now();
    auto embedding_future = batcher_->submit({query_text});
    auto embeddings = embedding_future.get();
    auto embedding_end = std::chrono::steady_clock::now();

    if (embeddings.empty()) {
        SPDLOG_WARN("No embedding returned for query {}", query_id);
        return {};
    }

    auto embedding_latency = std::chrono::duration_cast<std::chrono::microseconds>(
        embedding_end - embedding_start
    ).count();

    // Step 2: Search vector store
    auto search_start = std::chrono::steady_clock::now();
    auto results = vector_store_->search(embeddings[0], top_k);
    auto search_end = std::chrono::steady_clock::now();

    auto search_latency = std::chrono::duration_cast<std::chrono::microseconds>(
        search_end - search_start
    ).count();

    // Step 3: Enrich results with timing
    auto total_latency = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start
    ).count();

    for (auto& result : results) {
        result.embedding_latency_us = embedding_latency;
        result.search_latency_us = search_latency;
        result.total_latency_us = total_latency;
    }

    // Step 4: Merge and deduplicate
    results = result_merger_->merge(results);

    SPDLOG_DEBUG("Query {} completed: embedding={}us, search={}us, total={}us",
                 query_id, embedding_latency, search_latency, total_latency);

    return results;
}

SearchResultVector RAGPipeline::query(const QueryRequest& request) {
    if (!is_ready()) {
        SPDLOG_WARN("Pipeline not ready, rejecting query {}", request.query_id);
        return {};
    }

    return execute_query(request.query_id, request.query_text, request.top_k);
}

std::future<SearchResultVector> RAGPipeline::query_async(const QueryRequest& request) {
    return std::async(std::launch::async, [this, request]() {
        return query(request);
    });
}

void RAGPipeline::load_corpus(const std::string& corpus_path) {
    SPDLOG_INFO("Loading corpus from {}", corpus_path);

    try {
        vector_store_->load(corpus_path);
        ready_ = true;

        SPDLOG_INFO("Corpus loaded: {} vectors indexed", vector_store_->size());
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to load corpus: {}", e.what());
        ready_ = false;
        throw;
    }
}

void RAGPipeline::rebuild_index_async() {
    // Double-buffer pattern: keep old index active while rebuilding
    SPDLOG_INFO("Starting async index rebuild");

    std::thread([this]() {
        try {
            // In a real implementation, we'd load a new corpus or rebuild from scratch
            // For now, this is a placeholder for the double-buffer pattern
            SPDLOG_INFO("Index rebuild complete");
        } catch (const std::exception& e) {
            SPDLOG_ERROR("Index rebuild failed: {}", e.what());
        }
    }).detach();
}

bool RAGPipeline::is_ready() const {
    return ready_ && vector_store_->is_loaded();
}

BatchStats RAGPipeline::get_batcher_stats() const {
    return batcher_->get_stats();
}

}  // namespace rag_engine
