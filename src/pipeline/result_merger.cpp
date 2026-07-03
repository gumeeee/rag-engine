#include "rag_pipeline.h"
#include <unordered_set>

namespace rag_engine {

/**
 * Result merger implementation.
 */
ResultMerger::ResultMerger(const MergeConfig& config) : config_(config) {}

SearchResultVector ResultMerger::merge(
    const SearchResultVector& results,
    const MergeConfig& config
) {
    SearchResultVector merged = results;

    if (config.deduplicate) {
        deduplicate_by_id(merged);
    }

    return merged;
}

void ResultMerger::deduplicate_by_id(SearchResultVector& results) {
    std::unordered_set<std::string> seen_ids;
    SearchResultVector deduplicated;

    for (auto& result : results) {
        if (seen_ids.insert(result.chunk_id).second) {
            deduplicated.push_back(std::move(result));
        }
    }

    results = std::move(deduplicated);
}

}  // namespace rag_engine
