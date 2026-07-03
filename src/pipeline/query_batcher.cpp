#include "pipeline/query_batcher.h"
#include <spdlog/spdlog.h>

namespace rag_engine {

QueryBatcher::QueryBatcher(
    std::shared_ptr<IEmbeddingService> embedding_service,
    const ConfigManager::RAGConfig::BatcherConfig& config
) : config_(config), embedding_service_(std::move(embedding_service)) {

    SPDLOG_INFO("Initializing QueryBatcher: max_batch_size={}, max_wait_ms={}",
                config_.max_batch_size, config_.max_wait_ms);

    // Start background dispatcher thread
    dispatcher_thread_ = std::thread([this] { background_loop(); });
}

QueryBatcher::~QueryBatcher() {
    shutdown();
}

void QueryBatcher::shutdown() {
    running_ = false;
    cv_.notify_all();

    if (dispatcher_thread_.joinable()) {
        dispatcher_thread_.join();
    }
}

void QueryBatcher::background_loop() {
    while (running_) {
        std::vector<PendingBatch> batches_to_dispatch;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            // Wait for either:
            // 1. Time to check batches
            // 2. Shutdown signal
            cv_.wait_for(lock, std::chrono::milliseconds(config_.max_wait_ms), [this] {
                return !running_;
            });

            if (!running_) break;

            // Check all pending batches for dispatch conditions
            auto now = std::chrono::steady_clock::now();

            std::queue<PendingBatch> remaining;
            while (!pending_batches_.empty()) {
                auto batch = std::move(const_cast<PendingBatch&>(pending_batches_.front()));
                pending_batches_.pop();

                auto wait_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    now - batch.enqueued_at
                ).count();

                bool should_dispatch =
                    wait_time >= (config_.max_wait_ms * 1000) ||  // Time trigger
                    batch.queries.size() >= static_cast<size_t>(config_.max_batch_size);  // Size trigger

                if (should_dispatch) {
                    batches_to_dispatch.push_back(std::move(batch));
                } else {
                    remaining.push(std::move(batch));
                }
            }
            pending_batches_ = std::move(remaining);
        }

        // Dispatch batches outside lock
        for (auto& batch : batches_to_dispatch) {
            auto query_count = batch.queries.size();
            auto enqueued_at = batch.enqueued_at;

            dispatch_batch(std::move(batch.queries), std::move(batch.promises));

            // Update statistics
            total_batches_dispatched_++;
            total_queries_in_batches_ += query_count;

            auto dispatch_time = std::chrono::steady_clock::now();
            auto wait_time = std::chrono::duration_cast<std::chrono::microseconds>(
                dispatch_time - enqueued_at
            ).count();
            total_wait_time_us_ += wait_time;
            batch_size_sum_ += query_count;
            batch_count_++;

            if (query_count < static_cast<size_t>(min_batch_size_)) {
                min_batch_size_ = query_count;
            }
            if (query_count > static_cast<size_t>(max_batch_size_)) {
                max_batch_size_ = query_count;
            }
        }
    }

    // Flush remaining batches on shutdown
    flush();
}

void QueryBatcher::dispatch_batch(
    std::vector<std::string> queries,
    std::vector<std::promise<std::vector<Embedding>>> promises
) {
    auto query_count = queries.size();

    embedding_service_->encode_batch_async(queries)
        .then([promises = std::move(promises)](auto result_future) mutable {
            try {
                auto embeddings = result_future.get();

                // Resolve each individual promise with its embedding
                for (size_t i = 0; i < std::min(promises.size(), embeddings.size()); ++i) {
                    std::vector<Embedding> single_emb;
                    single_emb.push_back(std::move(embeddings[i]));
                    promises[i].set_value(std::move(single_emb));
                }

                // Handle case where batch failed but some promises need resolution
                for (size_t i = embeddings.size(); i < promises.size(); ++i) {
                    try {
                        throw std::runtime_error("Embedding batch incomplete");
                    } catch (...) {
                        promises[i].set_exception(std::current_exception());
                    }
                }
            } catch (const std::exception& e) {
                SPDLOG_ERROR("Batch dispatch failed: {}", e.what());
                for (auto& promise : promises) {
                    try {
                        throw;
                    } catch (...) {
                        promise.set_exception(std::current_exception());
                    }
                }
            }
        });
}

std::future<std::vector<Embedding>> QueryBatcher::submit(const std::vector<std::string>& queries) {
    std::promise<std::vector<Embedding>> promise;

    if (queries.empty()) {
        promise.set_value({});
        return promise.get_future();
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);

        PendingBatch batch;
        batch.queries = queries;
        batch.promises.push_back(std::move(promise));
        batch.enqueued_at = std::chrono::steady_clock::now();

        pending_batches_.push(std::move(batch));
        total_queries_submitted_ += queries.size();

        // Signal dispatcher to check batch sizes
        cv_.notify_one();
    }

    // Return future immediately (non-blocking)
    return pending_batches_.back().promises[0].get_future();
}

void QueryBatcher::flush() {
    std::vector<PendingBatch> batches_to_dispatch;

    {
        std::lock_guard<std::mutex> lock(mutex_);

        while (!pending_batches_.empty()) {
            batches_to_dispatch.push_back(std::move(const_cast<PendingBatch&>(pending_batches_.front())));
            pending_batches_.pop();
        }
    }

    // Dispatch all pending batches
    for (auto& batch : batches_to_dispatch) {
        auto query_count = batch.queries.size();
        auto enqueued_at = batch.enqueued_at;

        dispatch_batch(std::move(batch.queries), std::move(batch.promises));

        // Update statistics
        total_batches_dispatched_++;
        total_queries_in_batches_ += query_count;
        batch_count_++;
    }
}

BatchStats QueryBatcher::get_stats() const {
    BatchStats stats;
    stats.total_queries_submitted = total_queries_submitted_;
    stats.total_batches_dispatched = total_batches_dispatched_;
    stats.total_queries_in_batches = total_queries_in_batches_;

    auto bc = batch_count_.load();
    if (bc > 0) {
        stats.avg_batch_size = static_cast<double>(batch_size_sum_.load()) / bc;
        stats.avg_wait_time_us = total_wait_time_us_.load() / bc;
    }

    stats.min_batch_size = min_batch_size_.load();
    stats.max_batch_size = max_batch_size_.load();

    return stats;
}

}  // namespace rag_engine
