#pragma once

#include "common/types.h"
#include <string>
#include <unordered_map>
#include <atomic>
#include <vector>

namespace rag_engine {

/**
 * Prometheus-compatible metrics collector.
 */
class MetricsCollector {
public:
    static MetricsCollector& instance();

    // Counter operations
    void increment_counter(const std::string& name, double value = 1.0);
    double get_counter(const std::string& name) const;

    // Histogram operations (for latency)
    void observe_histogram(const std::string& name, double value);
    std::vector<double> get_histogram(const std::string& name) const;

    // Gauge operations
    void set_gauge(const std::string& name, double value);
    double get_gauge(const std::string& name) const;

    // Export for Prometheus scraping
    std::string export_prometheus_format() const;

    // Reset all metrics
    void reset();

private:
    MetricsCollector() = default;

    struct Counter {
        std::atomic<double> value{0.0};
        std::chrono::steady_clock::time_point last_update;
    };

    struct Histogram {
        std::vector<double> values;
        mutable std::mutex mutex;
        size_t max_size{1000};
    };

    struct Gauge {
        std::atomic<double> value{0.0};
    };

    mutable std::mutex mutex_;
    std::unordered_map<std::string, Counter> counters_;
    std::unordered_map<std::string, Histogram> histograms_;
    std::unordered_map<std::string, Gauge> gauges_;
};

// Predefined metric names
namespace metrics {
    constexpr const char* QUERY_DURATION_SECONDS = "rag_query_duration_seconds";
    constexpr const char* QUERIES_TOTAL = "rag_queries_total";
    constexpr const char* QUERIES_IN_FLIGHT = "rag_queries_in_flight";
    constexpr const char* EMBEDDING_BATCH_SIZE = "rag_embedding_batch_size";
    constexpr const char* INDEX_SIZE = "rag_index_size";
    constexpr const char* GPU_MEMORY_USED_GB = "rag_gpu_memory_used_bytes";
    constexpr const char* LAST_RESULT_COUNT = "rag_last_result_count";
}

}  // namespace rag_engine
