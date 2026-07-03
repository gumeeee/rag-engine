#include "common/config.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <algorithm>

namespace rag_engine {

ConfigManager& ConfigManager::instance() {
    static ConfigManager instance;
    return instance;
}

void ConfigManager::load_defaults() {
    // Default configuration is already set via member initialization
}

void ConfigManager::load_from_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return;
    }

    std::ostringstream content;
    content << file.rdbuf();
    std::istringstream lines(content.str());
    std::string line;

    while (std::getline(lines, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') continue;

        // Simple key=value parsing
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;

        std::string key = line.substr(0, eq_pos);
        std::string value = line.substr(eq_pos + 1);

        // Trim whitespace
        auto trim = [](std::string& s) {
            s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) { return !std::isspace(ch); }));
            s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) { return !std::isspace(ch); }).base(), s.end());
        };
        trim(key);
        trim(value);

        env_overrides_[key] = value;
    }
}

void ConfigManager::load_from_env() {
    // Server config
    if (const char* val = std::getenv("RAG_PORT")) {
        config_.server.port = std::stoi(val);
    }
    if (const char* val = std::getenv("RAG_NUM_THREADS")) {
        config_.server.num_threads = std::stoi(val);
    }

    // Embedding config
    if (const char* val = std::getenv("RAG_MODEL_PATH")) {
        config_.embedding.model_path = val;
    }
    if (const char* val = std::getenv("RAG_USE_CUDA")) {
        config_.embedding.use_cuda = (std::string(val) == "1" || std::string(val) == "true");
    }
    if (const char* val = std::getenv("RAG_CUDA_DEVICE")) {
        config_.embedding.cuda_device_id = std::stoi(val);
    }

    // Search config
    if (const char* val = std::getenv("RAG_INDEX_PATH")) {
        config_.search.index_path = val;
    }
    if (const char* val = std::getenv("RAG_CORPUS_PATH")) {
        config_.search.corpus_path = val;
    }

    // Batcher config
    if (const char* val = std::getenv("RAG_MAX_BATCH_SIZE")) {
        config_.batcher.max_batch_size = std::stoi(val);
    }
    if (const char* val = std::getenv("RAG_MAX_WAIT_MS")) {
        config_.batcher.max_wait_ms = std::stoi(val);
    }

    // Observability config
    if (const char* val = std::getenv("RAG_LOG_LEVEL")) {
        config_.observability.log_level = val;
    }
    if (const char* val = std::getenv("RAG_LOG_FILE")) {
        config_.observability.log_file = val;
    }
}

}  // namespace rag_engine
