#pragma once

#include "embedding/embedding_service.h"
#include "common/types.h"
#include "common/config.h"
#include <memory>
#include <future>
#include <vector>
#include <string>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <queue>

namespace rag_engine {

/**
 * Query batcher that collects queries and dispatches them in batches
 * for efficient GPU utilization and 5-8x throughput improvement.
 */
class QueryBatcher {
public:
    explicit QueryBatcher(
        std::shared_ptr<IEmbeddingService> embedding_service,
        const ConfigManager::RAGConfig::BatcherConfig& config
    );
    ~QueryBatcher();

    /**
     * Submit a batch of queries for encoding.
     * Returns a future that resolves to the embeddings.
     * Non-blocking - returns within 1ms.
     */
    std::future<std::vector<Embedding>> submit(const std::vector<std::string>& queries);

    /**
     * Force flush of current pending batch.
     */
    void flush();

    /**
     * Get batching statistics.
     */
    BatchStats get_stats() const;

    /**
     * Shutdown the batcher gracefully.
     */
    void shutdown();

private:
    void background_loop();
    void dispatch_batch(std::vector<std::string> queries,
                       std::vector<std::promise<std::vector<Embedding>>> promises);

    ConfigManager::RAGConfig::BatcherConfig config_;
    std::shared_ptr<IEmbeddingService> embedding_service_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_{true};

    struct PendingBatch {
        std::vector<std::string> queries;
        std::vector<std::promise<std::vector<Embedding>>> promises;
        std::chrono::steady_clock::time_point enqueued_at;
    };
    std::queue<PendingBatch> pending_batches_;

    std::thread dispatcher_thread_;

    // Statistics
    std::atomic<size_t> total_queries_submitted_{0};
    std::atomic<size_t> total_batches_dispatched_{0};
    std::atomic<size_t> total_queries_in_batches_{0};
    std::atomic<int64_t> total_wait_time_us_{0};
    std::atomic<int64_t> batch_size_sum_{0};
    std::atomic<int64_t> batch_count_{0};
    std::atomic<int64_t> min_batch_size_{INT64_MAX};
    std::atomic<int64_t> max_batch_size_{0};
};

}  // namespace rag_engine
