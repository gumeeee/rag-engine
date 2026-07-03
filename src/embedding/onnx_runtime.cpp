#include "embedding/onnx_runtime.h"
#include "common/cuda_utils.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>

namespace rag_engine {

ONNXEmbeddingService::ONNXEmbeddingService(const ConfigManager::RAGConfig::EmbeddingConfig& config)
    : config_(config), env_(ORT_LOGGING_LEVEL_WARNING) {

    // Check CUDA availability
    use_cuda_ = config.use_cuda && CUDAContext::instance().is_available();
    device_id_ = config.cuda_device_id;

    if (use_cuda_) {
        SPDLOG_INFO("Initializing ONNX embedding service with CUDA (device {})", device_id_);
        CUDAContext::instance().set_device(device_id_);
    } else {
        SPDLOG_WARN("CUDA not available, falling back to CPU for embeddings");
    }

    // Configure session options
    session_options_.SetIntraOpNumThreads(config.num_threads);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    if (use_cuda_) {
        // Register CUDA EP
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = device_id_;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
        cuda_options.gpu_mem_limit = 0;  // No limit
        cuda_options.arena_extend_strategy = 0;
        cuda_options.do_copy_in_default_stream = true;

        session_options_.AppendExecutionProvider_CUDA(cuda_options);
    }

    load_model();
}

ONNXEmbeddingService::~ONNXEmbeddingService() = default;

void ONNXEmbeddingService::load_model() {
    try {
        session_ = Ort::Session(env_, config_.model_path.c_str(), session_options_);

        // Get input/output info
        size_t num_input_nodes = session_.GetInputCount();
        size_t num_output_nodes = session_.GetOutputCount();

        Ort::AllocatorWithDefaultOptions allocator;

        // Input names
        for (size_t i = 0; i < num_input_nodes; ++i) {
            auto input_name = session_.GetInputNameAllocated(i, allocator);
            input_names_.push_back(input_name.get());
        }

        // Output names
        for (size_t i = 0; i < num_output_nodes; ++i) {
            auto output_name = session_.GetOutputNameAllocated(i, allocator);
            output_names_.push_back(output_name.get());
        }

        // Get input shape
        auto input_shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        input_shape_ = input_shape;

        // Verify it's [batch_size, sequence_length]
        if (input_shape_.size() >= 2) {
            SPDLOG_INFO("ONNX model loaded: input shape [?, {}], output count: {}",
                       input_shape_[1], num_output_nodes);
        }
    } catch (const Ort::Exception& e) {
        SPDLOG_ERROR("Failed to load ONNX model: {}", e.what());
        throw std::runtime_error("ONNX model loading failed");
    }
}

std::vector<float> ONNXEmbeddingService::preprocess(const std::vector<std::string>& texts) {
    // Simple tokenization - in production, use proper tokenizer
    // This creates a [batch_size, seq_len] tensor of token IDs
    // For simplicity, we use a fixed vocabulary simulation

    size_t batch_size = texts.size();
    size_t seq_len = config_.max_seq_length;

    std::vector<float> input_data(batch_size * seq_len, 0.0f);

    for (size_t i = 0; i < texts.size(); ++i) {
        const auto& text = texts[i];
        // Simple hash-based tokenization for demonstration
        // In production, use proper tokenizers library
        size_t text_len = std::min(text.length(), static_cast<size_t>(seq_len));

        for (size_t j = 0; j < text_len; ++j) {
            // Map characters to pseudo-token IDs (0-255 range)
            input_data[i * seq_len + j] = static_cast<float>(
                static_cast<unsigned char>(text[j]) + 1
            );
        }
        // Add CLS token at position 0
        input_data[i * seq_len] = 101.0f;  // CLS token ID
    }

    return input_data;
}

std::vector<Embedding> ONNXEmbeddingService::postprocess(
    const float* output_buffer,
    size_t batch_size
) {
    std::vector<Embedding> results;
    results.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        Embedding emb(config_.dimension);
        std::memcpy(emb.data_ptr(), output_buffer + i * config_.dimension,
                    static_cast<size_t>(config_.dimension) * sizeof(float));
        results.push_back(std::move(emb));
    }

    return results;
}

std::future<Embedding> ONNXEmbeddingService::encode_async(const std::string& text) {
    return encode_batch_async({text}).then([](std::vector<Embedding> embeddings) {
        return std::move(embeddings[0]);
    });
}

std::future<std::vector<Embedding>> ONNXEmbeddingService::encode_batch_async(
    const std::vector<std::string>& texts
) {
    std::promise<std::vector<Embedding>> promise;

    if (texts.empty()) {
        promise.set_value({});
        return promise.get_future();
    }

    try {
        size_t batch_size = texts.size();

        // Preprocess
        auto input_data = preprocess(texts);

        // Create input tensor
        std::vector<int64_t> input_shape = {
            static_cast<int64_t>(batch_size),
            static_cast<int64_t>(config_.max_seq_length)
        };

        Ort::MemoryInfo mem_info("Cpu", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()
        );

        // Run inference
        auto output_tensors = session_.Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &input_tensor,
            1,
            output_names_.data(),
            output_names_.size()
        );

        // Get output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        // Extract embeddings (use mean pooling if multiple outputs)
        std::vector<Embedding> results;

        if (output_shape.size() >= 2) {
            // [batch, seq_len, hidden] or [batch, hidden]
            size_t hidden_dim = output_shape.back();

            for (size_t i = 0; i < batch_size; ++i) {
                Embedding emb(config_.dimension);
                const float* start = output_data + i * hidden_dim;
                size_t copy_size = std::min(static_cast<size_t>(config_.dimension), hidden_dim);
                std::memcpy(emb.data_ptr(), start, copy_size * sizeof(float));
                results.push_back(std::move(emb));
            }
        }

        promise.set_value(std::move(results));

    } catch (const std::exception& e) {
        SPDLOG_ERROR("ONNX inference failed: {}", e.what());
        promise.set_exception(std::current_exception());
    }

    return promise.get_future();
}

}  // namespace rag_engine
