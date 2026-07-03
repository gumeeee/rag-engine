#pragma once

#include <string>
#include <chrono>

namespace rag_engine {

/**
 * Trace context for distributed tracing.
 * Propagates query_id through all pipeline stages.
 */
struct TraceContext {
    std::string query_id;
    std::chrono::steady_clock::time_point start_time;

    TraceContext() : start_time(std::chrono::steady_clock::now()) {}
    explicit TraceContext(const std::string& id) : query_id(id), start_time(std::chrono::steady_clock::now()) {}
};

/**
 * Scoped tracer that automatically logs entry/exit.
 */
class ScopedTracer {
public:
    ScopedTracer(const std::string& operation, const std::string& query_id);
    ~ScopedTracer();

    void add_tag(const std::string& key, const std::string& value);

private:
    std::string operation_;
    std::string query_id_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace rag_engine
