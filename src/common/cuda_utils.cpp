#include "common/cuda_utils.h"
#include <stdexcept>
#include <cstring>

namespace rag_engine {

// ============================================================================
// CUDAContext Implementation
// ============================================================================

CUDAContext::CUDAContext() : device_id_(0) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    available_ = (err == cudaSuccess && device_count > 0);

    if (available_) {
        cudaGetDevice(&device_id_);
    }
}

void CUDAContext::set_device(int device_id) {
    if (!available_) return;

    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_id < 0 || device_id >= device_count) {
        device_id = 0;
    }

    cudaSetDevice(device_id);
    device_id_ = device_id;
}

cudaStream_t CUDAContext::create_stream() {
    if (!available_) return nullptr;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    managed_streams_.push_back(stream);
    return stream;
}

void CUDAContext::destroy_stream(cudaStream_t stream) {
    if (!available_ || stream == nullptr) return;

    cudaStreamDestroy(stream);
    managed_streams_.erase(
        std::remove(managed_streams_.begin(), managed_streams_.end(), stream),
        managed_streams_.end()
    );
}

float CUDAContext::get_memory_usage_gb() const {
    if (!available_) return 0.0f;

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    return static_cast<float>(total_bytes - free_bytes) / (1024.0f * 1024.0f * 1024.0f);
}

float CUDAContext::get_total_memory_gb() const {
    if (!available_) return 0.0f;

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    return static_cast<float>(total_bytes) / (1024.0f * 1024.0f * 1024.0f);
}

CUDAContext& CUDAContext::instance() {
    static CUDAContext instance;
    return instance;
}

// ============================================================================
// CUDAStream Implementation
// ============================================================================

CUDAStream::CUDAStream() {
    if (CUDAContext::instance().is_available()) {
        cudaStreamCreate(&stream_);
    }
}

CUDAStream::CUDAStream(unsigned int flags) {
    if (CUDAContext::instance().is_available()) {
        cudaStreamCreateWithFlags(&stream_, flags);
    }
}

CUDAStream::~CUDAStream() {
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

CUDAStream::CUDAStream(CUDAStream&& other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
}

CUDAStream& CUDAStream::operator=(CUDAStream&& other) noexcept {
    if (this != &other) {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
        stream_ = other.stream_;
        other.stream_ = nullptr;
    }
    return *this;
}

void CUDAStream::synchronize() const {
    if (stream_ != nullptr) {
        cudaStreamSynchronize(stream_);
    }
}

// ============================================================================
// CUDAMemoryPool Implementation
// ============================================================================

struct CUDAMemoryPool::Impl {
    std::vector<std::pair<void*, size_t>> allocations;
    size_t max_bytes;
    size_t used_bytes;
};

CUDAMemoryPool::CUDAMemoryPool(size_t max_bytes)
    : pImpl_(std::make_unique<Impl>()), max_bytes_(max_bytes) {
    pImpl_->max_bytes = max_bytes;
}

CUDAMemoryPool::~CUDAMemoryPool() {
    for (auto& alloc : pImpl_->allocations) {
        cudaFree(alloc.first);
    }
}

void* CUDAMemoryPool::allocate(size_t bytes) {
    if (!CUDAContext::instance().is_available()) {
        return nullptr;
    }

    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);

    if (err != cudaSuccess) {
        return nullptr;
    }

    pImpl_->allocations.emplace_back(ptr, bytes);
    used_bytes_ += bytes;
    peak_bytes_ = std::max(peak_bytes_, used_bytes_);

    return ptr;
}

void CUDAMemoryPool::deallocate(void* ptr) {
    if (!ptr || !CUDAContext::instance().is_available()) return;

    for (auto it = pImpl_->allocations.begin(); it != pImpl_->allocations.end(); ++it) {
        if (it->first == ptr) {
            cudaFree(ptr);
            used_bytes_ -= it->second;
            pImpl_->allocations.erase(it);
            return;
        }
    }
}

void CUDAMemoryPool::copy_async(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    if (!CUDAContext::instance().is_available()) return;
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
}

// ============================================================================
// CUDAMemory Implementation
// ============================================================================

CUDAMemory::CUDAMemory(size_t bytes) : bytes_(bytes) {
    if (CUDAContext::instance().is_available() && bytes > 0) {
        cudaMalloc(&ptr_, bytes);
    }
}

CUDAMemory::~CUDAMemory() {
    if (ptr_ != nullptr && CUDAContext::instance().is_available()) {
        cudaFree(ptr_);
    }
}

CUDAMemory::CUDAMemory(CUDAMemory&& other) noexcept
    : ptr_(other.ptr_), bytes_(other.bytes_) {
    other.ptr_ = nullptr;
    other.bytes_ = 0;
}

CUDAMemory& CUDAMemory::operator=(CUDAMemory&& other) noexcept {
    if (this != &other) {
        if (ptr_ != nullptr && CUDAContext::instance().is_available()) {
            cudaFree(ptr_);
        }
        ptr_ = other.ptr_;
        bytes_ = other.bytes_;
        other.ptr_ = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

}  // namespace rag_engine
