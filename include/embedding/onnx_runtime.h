#pragma once

#include "embedding_service.h"
#include "common/config.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>

namespace rag_engine {

/**
 * ONNX Runtime embedding service with CUDA execution provider support.
 * Falls back to CPU if CUDA is not available.
 */
class ONNXEmbeddingService : public IEmbeddingService {
public:
    explicit ONNXEmbeddingService(const ConfigManager::RAGConfig::EmbeddingConfig& config);
    ~ONNXEmbeddingService() override;

    std::future<Embedding> encode_async(const std::string& text) override;
    std::future<std::vector<Embedding>> encode_batch_async(
        const std::vector<std::string>& texts
    ) override;

    int32_t dimension() const override { return config_.dimension; }
    bool uses_gpu() const override { return use_cuda_; }
    std::string name() const override { return use_cuda_ ? "ONNX+CUDA" : "ONNX+CPU"; }

private:
    void load_model();
    std::vector<float> preprocess(const std::vector<std::string>& texts);
    std::vector<Embedding> postprocess(const float* output_buffer, size_t batch_size);

    ConfigManager::RAGConfig::EmbeddingConfig config_;
    Ort::Env env_;
    Ort::Session session_;
    Ort::SessionOptions session_options_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;
    bool use_cuda_{false};
    int device_id_{0};
};

}  // namespace rag_engine
