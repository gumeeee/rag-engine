#include "common/types.h"
#include <sstream>

namespace rag_engine {

std::string LatencyBreakdown::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"embedding_us\":" << embedding_us << ",";
    oss << "\"search_us\":" << search_us << ",";
    oss << "\"total_us\":" << total_us;
    oss << "}";
    return oss.str();
}

std::string BatchStats::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"total_queries_submitted\":" << total_queries_submitted << ",";
    oss << "\"total_batches_dispatched\":" << total_batches_dispatched << ",";
    oss << "\"avg_batch_size\":" << avg_batch_size << ",";
    oss << "\"avg_wait_time_us\":" << avg_wait_time_us << ",";
    oss << "\"min_batch_size\":" << min_batch_size << ",";
    oss << "\"max_batch_size\":" << max_batch_size;
    oss << "}";
    return oss.str();
}

}  // namespace rag_engine
