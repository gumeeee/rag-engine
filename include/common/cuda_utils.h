#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <cstddef>

namespace rag_engine {

/**
 * CUDA context manager for device selection and stream management.
 * Implements singleton pattern for global CUDA state.
 */
class CUDAContext {
public:
    static CUDAContext& instance();

    CUDAContext(const CUDAContext&) = delete;
    CUDAContext& operator=(const CUDAContext&) = delete;

    void set_device(int device_id);
    int current_device() const { return device_id_; }

    cudaStream_t create_stream();
    void destroy_stream(cudaStream_t stream);
    const std::vector<cudaStream_t>& streams() const { return managed_streams_; }

    float get_memory_usage_gb() const;
    float get_total_memory_gb() const;
    bool is_available() const { return available_; }

private:
    CUDAContext();

    int device_id_{0};
    bool available_{false};
    std::vector<cudaStream_t> managed_streams_;
};

/**
 * CUDA memory pool for efficient GPU memory allocation and reuse.
 * Reduces fragmentation and allocation overhead.
 */
class CUDAMemoryPool {
public:
    explicit CUDAMemoryPool(size_t max_bytes);
    ~CUDAMemoryPool();

    void* allocate(size_t bytes);
    void deallocate(void* ptr);

    // Async copy with pinned memory
    void copy_async(void* dst, const void* src, size_t bytes, cudaStream_t stream);

    size_t used_bytes() const { return used_bytes_; }
    size_t peak_bytes() const { return peak_bytes_; }
    float used_gb() const { return static_cast<float>(used_bytes_) / (1024.0f * 1024.0f * 1024.0f); }

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
    size_t used_bytes_{0};
    size_t peak_bytes_{0};
    size_t max_bytes_{0};
};

/**
 * RAII wrapper for CUDA device memory.
 */
class CUDAMemory {
public:
    CUDAMemory() = default;
    explicit CUDAMemory(size_t bytes);
    ~CUDAMemory();

    CUDAMemory(CUDAMemory&& other) noexcept;
    CUDAMemory& operator=(CUDAMemory&& other) noexcept;

    CUDAMemory(const CUDAMemory&) = delete;
    CUDAMemory& operator=(const CUDAMemory&) = delete;

    void* ptr() { return ptr_; }
    const void* ptr() const { return ptr_; }
    size_t bytes() const { return bytes_; }
    bool valid() const { return ptr_ != nullptr; }

    explicit operator bool() const { return valid(); }

private:
    void* ptr_{nullptr};
    size_t bytes_{0};
};

/**
 * RAII wrapper for CUDA streams.
 */
class CUDAStream {
public:
    CUDAStream();
    explicit CUDAStream(unsigned int flags);
    ~CUDAStream();

    CUDAStream(CUDAStream&& other) noexcept;
    CUDAStream& operator=(CUDAStream&& other) noexcept;

    cudaStream_t get() { return stream_; }
    cudaStream_t get() const { return stream_; }

    void synchronize() const;
    bool is_valid() const { return stream_ != nullptr; }

private:
    cudaStream_t stream_{nullptr};
};

}  // namespace rag_engine
