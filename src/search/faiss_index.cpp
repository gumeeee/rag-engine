#include "search/faiss_index.h"
#include "common/cuda_utils.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <cstring>

namespace rag_engine {

FAISSIndex::FAISSIndex(const ConfigManager::RAGConfig::SearchConfig& config)
    : config_(config) {
    SPDLOG_INFO("Initializing FAISS HNSW index with M=32, efConstruction=40");
}

FAISSIndex::~FAISSIndex() = default;

void FAISSIndex::load(const std::string& path) {
    SPDLOG_INFO("Loading corpus from {}", path);

    try {
        load_corpus(path);
        build_index();

        SPDLOG_INFO("Index loaded: {} vectors, dimension {}", num_vectors_, dimension_);
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to load corpus: {}", e.what());
        throw;
    }
}

void FAISSIndex::load_corpus(const std::string& path) {
    // Load protobuf corpus
    // In production, this would use protobuf parsing
    // For now, load raw float vectors

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        SPDLOG_WARN("Corpus file not found: {}", path);
        return;
    }

    auto file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size < sizeof(int32_t) * 2) {
        SPDLOG_ERROR("Invalid corpus file format");
        return;
    }

    // Read header
    int32_t count, dim;
    file.read(reinterpret_cast<char*>(&count), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));

    dimension_ = dim;
    num_vectors_ = count;

    // Read embeddings
    size_t embedding_size = static_cast<size_t>(dim) * sizeof(float);
    std::vector<float> embeddings(static_cast<size_t>(count) * dim);

    for (int32_t i = 0; i < count; ++i) {
        file.read(reinterpret_cast<char*>(embeddings.data() + i * dim), embedding_size);

        // Create placeholder metadata
        DocumentChunk chunk;
        chunk.id = "chunk_" + std::to_string(i);
        chunk.text = "Document chunk " + std::to_string(i);
        chunk.source = path;
        chunk.position = i;
        metadata_.push_back(std::move(chunk));
    }

    // Build flat index first, then wrap with HNSW
    auto* flat_index = new faiss::IndexFlatIP(dimension_);
    index_.reset(flat_index);

    // Add vectors
    flat_index->add(num_vectors_, embeddings.data());
}

void FAISSIndex::build_index() {
    if (metadata_.empty()) {
        SPDLOG_WARN("No vectors to index");
        return;
    }

    // Create HNSW index
    // M=32: number of connections per node
    // efConstruction=40: build-time search width
    auto* hnsw_index = new faiss::IndexHNSW(dimension_, config_.m_parameter, config_.ef_construction);

    // Set search parameter
    hnsw_index->hnsw.efSearch = config_.ef_search;

    // If we have existing vectors, transfer them
    if (index_) {
        // Get vectors from flat index
        auto* flat_index = dynamic_cast<faiss::IndexFlat*>(index_.get());
        if (flat_index) {
            // Rebuild HNSW with existing vectors
            hnsw_index->add(num_vectors_, flat_index->xb.data());
        }
    }

    index_.reset(hnsw_index);

    SPDLOG_INFO("HNSW index built: M={}, efConstruction={}, efSearch={}",
                config_.m_parameter, config_.ef_construction, config_.ef_search);
}

void FAISSIndex::save(const std::string& path) {
    if (!index_) {
        SPDLOG_ERROR("Cannot save null index");
        return;
    }

    try {
        std::ofstream file(path, std::ios::binary);

        // Write header
        int32_t count = static_cast<int32_t>(num_vectors_);
        int32_t dim = dimension_;
        file.write(reinterpret_cast<const char*>(&count), sizeof(int32_t));
        file.write(reinterpret_cast<const char*>(&dim), sizeof(int32_t));

        // Write vectors
        auto* flat_index = dynamic_cast<faiss::IndexFlat*>(index_.get());
        if (flat_index) {
            file.write(reinterpret_cast<const char*>(flat_index->xb.data()),
                       static_cast<size_t>(count) * dim * sizeof(float));
        }

        SPDLOG_INFO("Index saved to {}", path);
    } catch (const std::exception& e) {
        SPDLOG_ERROR("Failed to save index: {}", e.what());
        throw;
    }
}

SearchResultVector FAISSIndex::search(
    const Embedding& query,
    int32_t k,
    int32_t ef_search
) {
    if (!index_ || query.empty()) {
        return {};
    }

    // Set search parameter
    auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(index_.get());
    if (hnsw && ef_search != config_.ef_search) {
        hnsw->hnsw.efSearch = ef_search;
    }

    // Search
    std::vector<faiss::idx_t> indices(k);
    std::vector<float> distances(k);

    index_->search(1, query.data_ptr(), k, distances.data(), indices.data());

    // Build results
    SearchResultVector results;
    for (int32_t i = 0; i < k; ++i) {
        auto idx = indices[i];
        if (idx < 0 || idx >= static_cast<faiss::idx_t>(metadata_.size())) {
            continue;
        }

        SearchResult result;
        result.chunk_id = metadata_[idx].id;
        result.text = metadata_[idx].text;
        result.source = metadata_[idx].source;
        result.similarity_score = 1.0f / (1.0f + distances[i]);  // Convert L2 to similarity
        result.rank = i + 1;

        results.push_back(std::move(result));
    }

    return results;
}

size_t FAISSIndex::size() const {
    return num_vectors_;
}

}  // namespace rag_engine
