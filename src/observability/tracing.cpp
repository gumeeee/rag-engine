#include "observability/tracing.h"
#include <spdlog/spdlog.h>

namespace rag_engine {

ScopedTracer::ScopedTracer(const std::string& operation, const std::string& query_id)
    : operation_(operation), query_id_(query_id), start_(std::chrono::steady_clock::now()) {
    SPDLOG_DEBUG("[{}] Starting: {}", query_id_, operation_);
}

ScopedTracer::~ScopedTracer() {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start_
    ).count();
    SPDLOG_DEBUG("[{}] Completed {} in {}us", query_id_, operation_, elapsed);
}

void ScopedTracer::add_tag(const std::string& key, const std::string& value) {
    // In a real implementation, this would add tags to a tracing system
    SPDLOG_DEBUG("[{}] {}={}", query_id_, key, value);
}

}  // namespace rag_engine
