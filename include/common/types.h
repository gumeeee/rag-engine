#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <memory>
#include <optional>

namespace rag_engine {

/**
 * Represents a dense embedding vector from the embedding model.
 */
struct Embedding {
    std::vector<float> data;
    int32_t dimension{0};

    Embedding() = default;
    explicit Embedding(int32_t dim) : data(static_cast<size_t>(dim), 0.0f), dimension(dim) {}

    float* data_ptr() { return data.data(); }
    const float* data_ptr() const { return data.data(); }

    bool empty() const { return data.empty(); }
    size_t size() const { return data.size(); }
};

/**
 * Represents a chunk of a document that can be indexed and retrieved.
 */
struct DocumentChunk {
    std::string id;
    std::string text;
    std::string source;
    int32_t position{0};
    std::optional<Embedding> embedding;  // Pre-computed embedding if available

    DocumentChunk() = default;
    DocumentChunk(std::string id_, std::string text_, std::string source_ = "")
        : id(std::move(id_)), text(std::move(text_)), source(std::move(source_)) {}
};

/**
 * Represents a search result with metadata and similarity score.
 */
struct SearchResult {
    std::string chunk_id;
    std::string text;
    std::string source;
    float similarity_score{0.0f};
    int32_t rank{0};
    int64_t embedding_latency_us{0};
    int64_t search_latency_us{0};
    int64_t total_latency_us{0};

    SearchResult() = default;
    SearchResult(std::string chunk_id_, std::string text_, std::string source_, float score)
        : chunk_id(std::move(chunk_id_)),
          text(std::move(text_)),
          source(std::move(source_)),
          similarity_score(score) {}
};

using SearchResultVector = std::vector<SearchResult>;

/**
 * Incoming query request with tracing support.
 */
struct QueryRequest {
    std::string query_id;
    std::string query_text;
    int32_t top_k{5};
    std::chrono::steady_clock::time_point received_at;

    QueryRequest() = default;
    QueryRequest(std::string query_id_, std::string query_text_, int32_t top_k_ = 5)
        : query_id(std::move(query_id_)),
          query_text(std::move(query_text_)),
          top_k(top_k_),
          received_at(std::chrono::steady_clock::now()) {}
};

/**
 * Timing breakdown for query processing stages.
 */
struct LatencyBreakdown {
    int64_t embedding_us{0};
    int64_t search_us{0};
    int64_t total_us{0};

    LatencyBreakdown() = default;

    std::string to_json() const;
};

/**
 * Batch statistics for monitoring and debugging.
 */
struct BatchStats {
    size_t total_queries_submitted{0};
    size_t total_batches_dispatched{0};
    size_t total_queries_in_batches{0};
    double avg_batch_size{0.0};
    int64_t avg_wait_time_us{0};
    int64_t min_batch_size{0};
    int64_t max_batch_size{0};

    std::string to_json() const;
};

}  // namespace rag_engine
