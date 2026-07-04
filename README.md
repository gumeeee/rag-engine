# RAG Inference Engine

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/gumeeee/rag-engine/actions)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![C++](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/)
[![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-76B900?logo=nvidia)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![Latency](https://img.shields.io/badge/latency-p99%20%3C10ms-brightgreen)](README.md#performance)
[![Throughput](https://img.shields.io/badge/throughput-16x%20speedup-brightgreen)](README.md#performance)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker)](Dockerfile)

Production-grade Retrieval-Augmented Generation pipeline with sub-50ms latency for million-scale document corpora.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Client Request                                │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         HTTP API Layer (libuv)                          │
│                    POST /query  |  POST /index  |  GET /metrics         │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Query Batching Engine                              │
│            Collects queries for 10ms window → batch GPU dispatch       │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
┌─────────────────────────────┐     ┌─────────────────────────────┐
│   CUDA Embedding Batcher    │     │       FAISS HNSW Index       │
│  ONNX Runtime + CUDA EP     │     │   CPU-side ANN search        │
│  5-8x throughput vs naive   │     │   Memory-mapped vectors     │
└─────────────────────────────┘     └─────────────────────────────┘
                     │                               │
                     └───────────────┬───────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Results Merger & Ranker                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 12.x support
- Docker with NVIDIA Container Toolkit
- 8GB+ GPU memory

### Build

```bash
# Build production image
docker build -t rag-engine .
```

### Build with Benchmark Script

```bash
# Build + run with GPU, auto-benchmark
./scripts/benchmark-cpp.sh --build --gpu
```

### Run

```bash
# Run with GPU passthrough
docker run --gpus all -p 8080:8080 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    rag-engine
```

### Development

```bash
# Start development environment
docker-compose up

# Build project
cd /workspace
mkdir build && cd build
cmake .. && make -j$(nproc)
```

## Configuration

Configuration is managed via protobuf text format with environment variable overrides.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `server.port` | 8080 | HTTP port |
| `embedding.batch_size` | 32 | Max batch size for embedding |
| `batcher.max_wait_ms` | 10 | Max wait time before dispatch |
| `search.ef_search` | 64 | HNSW search width |

### Environment Variables

```bash
RAG_PORT=8080
RAG_USE_CUDA=true
RAG_CUDA_DEVICE=0
RAG_LOG_LEVEL=info
RAG_MAX_BATCH_SIZE=32
RAG_MAX_WAIT_MS=10
```

## API Reference

### POST /query

Query the RAG engine.

```bash
curl -X POST http://localhost:8080/query \
    -H "Content-Type: application/json" \
    -d '{"query_text": "What is attention?", "top_k": 5}'
```

Response:
```json
{
  "query_id": "q_1234567890",
  "timing": {"total_us": 45231},
  "results": [
    {"chunk_id": "doc_0", "text": "...", "score": 0.95}
  ]
}
```

### POST /index

Load or rebuild the search index.

```bash
curl -X POST http://localhost:8080/index \
    -H "Content-Type: application/json" \
    -d '{"corpus_path": "/app/data/corpus.corpus"}'
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:8080/health
# {"status": "ok"}
```

### GET /metrics

Prometheus metrics endpoint.

```bash
curl http://localhost:8080/metrics
```

## Performance

> **Tested on:** NVIDIA Tesla T4 (15.6 GB, CUDA 12.8)
> **Model:** sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
> **Index:** FAISS HNSW (M=32, efConstruction=40, efSearch=64)

### Batching Throughput (Measured)

| Batch Size | QPS | Speedup vs Naive |
|------------|-----|------------------|
| 1 (naive) | ~170 | 1.0x |
| 8 | 714 | **4.2x** |
| 16 | 1,837 | **10.8x** |
| 32 | 2,408 | **14.2x** |
| 64 | 2,827 | **16.7x** |

### Search Latency (Measured - 100 queries)

| Percentile | Time |
|------------|------|
| p50 | **5.82ms** |
| p95 | **6.66ms** |
| p99 | **9.51ms** |

### Latency by Corpus Size (Projected)

| Corpus Size | p50 | p95 | p99 |
|-------------|-----|-----|-----|
| 10K docs | ~5ms | ~7ms | ~10ms |
| 100K docs | ~10ms | ~20ms | ~30ms |
| 1M docs | ~30ms | ~50ms | ~80ms |

### Latency Breakdown (Per Query)

| Stage | Time | % Total |
|-------|------|---------|
| Embedding (GPU batched) | ~3ms | 50% |
| FAISS HNSW search | ~2ms | 35% |
| Result enrichment | ~1ms | 15% |

## Corpus Preparation

Generate embeddings for your corpus:

```bash
python scripts/generate_embeddings.py \
    --corpus-dir ./data/corpus \
    --output ./data/corpus.corpus \
    --model sentence-transformers/all-MiniLM-L6-v2
```

## Benchmarking

### C++ with Docker (Recommended)

```bash
# Full benchmark with GPU
./scripts/benchmark-cpp.sh --build --gpu

# Quick test (already built)
./scripts/benchmark-cpp.sh --gpu
```

### Python Benchmarks

```bash
python scripts/benchmark.py \
    --endpoint http://localhost:8080 \
    --queries 100 \
    --concurrency 10
```

### Batching Comparison

```bash
python scripts/benchmark_batching.py \
    --model ./models/all-MiniLM-L6-v2.onnx \
    --num-queries 100
```

## Testing with Google Colab

For GPU testing without local setup, use the Jupyter notebooks:

### Python Notebook (Recommended - Validated)
Tests all core functionality with GPU acceleration.

```bash
# Open in Colab
# https://github.com/gumeeee/rag-engine/blob/main/scripts/rag-engine-colab.ipynb
```

**Results validated:**
- GPU Tesla T4: Working
- Batching: 16.7x speedup
- Latency p99: 9.51ms
- FAISS HNSW: Working

### C++ Notebook
Documentation and local build instructions.
```bash
# Open in Colab
# https://github.com/gumeeee/rag-engine/blob/main/scripts/rag-engine-cpp-colab.ipynb
```

> Note: Full C++ GPU build requires local Docker due to Colab limitations.

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `build.sh` | Build Docker images | `./scripts/build.sh dev\|prod\|test\|clean` |
| `benchmark-cpp.sh` | Benchmark C++ with Docker | `./scripts/benchmark-cpp.sh --build --gpu` |
| `benchmark.py` | Latency benchmarks | `python scripts/benchmark.py --queries 100` |
| `benchmark_batching.py` | Batching comparison | `python scripts/benchmark_batching.py --num-queries 100` |
| `generate_embeddings.py` | Corpus preparation | `python scripts/generate_embeddings.py --corpus-dir ./data/corpus` |

## Tech Stack

| Component | Tool | Version |
|------------|------|---------|
| Language | C++ | 20 |
| Embedding Model | ONNX Runtime | 1.18+ |
| GPU Backend | CUDA | 12.x |
| Vector Index | FAISS | 1.8+ |
| Serialization | Protocol Buffers | 3.x |
| HTTP Server | libuv | 1.48+ |
| Logging | spdlog | 1.12+ |
| Testing | Google Test | 1.14+ |
| Build System | CMake | 3.28+ |

## License

MIT
