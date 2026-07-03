#pragma once

#include <string>
#include <unordered_map>

namespace rag_engine {

/**
 * Configuration manager singleton that loads settings from protobuf text format
 * and environment variables.
 */
class ConfigManager {
public:
    static ConfigManager& instance();

    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;

    void load_from_file(const std::string& path);
    void load_from_env();
    void load_defaults();

    struct RAGConfig {
        struct ServerConfig {
            int32_t port{8080};
            int32_t num_threads{4};
            int32_t backlog{128};
        };

        struct EmbeddingConfig {
            std::string model_path{"./models/all-MiniLM-L6-v2.onnx"};
            int32_t batch_size{32};
            int32_t max_seq_length{256};
            bool use_cuda{true};
            int32_t cuda_device_id{0};
            int32_t dimension{384};
        };

        struct SearchConfig {
            std::string index_path{"./data/indices/default.index"};
            std::string corpus_path{"./data/corpus/corpus.pb"};
            int32_t ef_search{64};
            int32_t num_results{10};
            int32_t m_parameter{32};
            int32_t ef_construction{40};
        };

        struct BatcherConfig {
            int32_t max_batch_size{32};
            int32_t max_wait_ms{10};
            int32_t num_dispatcher_threads{1};
        };

        struct ObservabilityConfig {
            bool enable_tracing{true};
            bool enable_prometheus{true};
            std::string log_level{"info"};
            std::string log_file{""};
        };

        ServerConfig server;
        EmbeddingConfig embedding;
        SearchConfig search;
        BatcherConfig batcher;
        ObservabilityConfig observability;
    };

    const RAGConfig& config() const { return config_; }
    RAGConfig& config() { return config_; }

private:
    ConfigManager() { load_defaults(); }

    RAGConfig config_;
    std::unordered_map<std::string, std::string> env_overrides_;
};

}  // namespace rag_engine
