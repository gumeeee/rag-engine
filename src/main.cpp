#include "common/config.h"
#include "common/cuda_utils.h"
#include "embedding/onnx_runtime.h"
#include "embedding/cpu_embedding.h"
#include "search/faiss_index.h"
#include "pipeline/query_batcher.h"
#include "pipeline/rag_pipeline.h"
#include "api/http_server.h"
#include "observability/metrics.h"
#include <spdlog/spdlog.h>
#include <iostream>

using namespace rag_engine;

int main(int argc, char* argv[]) {
    // Initialize spdlog
    spdlog::set_pattern("%Y-%m-%d %H:%M:%S.%e | %l | %t | %v");
    spdlog::set_level(spdlog::level::info);

    SPDLOG_INFO("RAG Inference Engine v1.0.0");

    // Load configuration
    auto& config_manager = ConfigManager::instance();

    std::string config_path = "/app/config/default_config.pb.txt";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
    }

    SPDLOG_INFO("Loading config from {}", config_path);
    config_manager.load_from_file(config_path);
    config_manager.load_from_env();

    const auto& config = config_manager.config();

    // Log configuration
    SPDLOG_INFO("Server config: port={}, threads={}", config.server.port, config.server.num_threads);
    SPDLOG_INFO("Embedding config: model={}, cuda={}, batch_size={}",
               config.embedding.model_path, config.embedding.use_cuda, config.embedding.batch_size);
    SPDLOG_INFO("Search config: ef_search={}, num_results={}", config.search.ef_search, config.search.num_results);

    // Initialize CUDA context
    if (CUDAContext::instance().is_available()) {
        SPDLOG_INFO("CUDA available, GPU memory: {:.2f} GB", CUDAContext::instance().get_total_memory_gb());
    } else {
        SPDLOG_WARN("CUDA not available, using CPU fallback");
    }

    // Create embedding service
    std::shared_ptr<IEmbeddingService> embedding_service;

    if (config.embedding.use_cuda && CUDAContext::instance().is_available()) {
        try {
            embedding_service = std::make_shared<ONNXEmbeddingService>(config.embedding);
            SPDLOG_INFO("Using ONNX+CUDA embedding service");
        } catch (const std::exception& e) {
            SPDLOG_WARN("Failed to initialize CUDA embedding: {}, falling back to CPU", e.what());
            embedding_service = std::make_shared<CPUEmbeddingService>(config.embedding.dimension);
        }
    } else {
        embedding_service = std::make_shared<CPUEmbeddingService>(config.embedding.dimension);
        SPDLOG_INFO("Using CPU embedding service");
    }

    // Create batcher
    auto batcher = std::make_shared<QueryBatcher>(embedding_service, config.batcher);

    // Create vector store
    auto vector_store = std::make_shared<FAISSIndex>(config.search);

    // Create pipeline
    auto pipeline = std::make_shared<RAGPipeline>(batcher, vector_store, config);

    // Load corpus if path specified
    if (!config.search.corpus_path.empty()) {
        try {
            pipeline->load_corpus(config.search.corpus_path);
        } catch (const std::exception& e) {
            SPDLOG_WARN("Failed to load corpus: {}", e.what());
        }
    }

    // Create HTTP handler and server
    auto handler = std::make_shared<RequestHandler>(pipeline);
    HTTPServer server(config.server.port, handler);

    // Start server (blocking)
    SPDLOG_INFO("Starting HTTP server on port {}", config.server.port);

    if (!server.start()) {
        SPDLOG_ERROR("Failed to start server");
        return 1;
    }

    return 0;
}
