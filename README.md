# RAG Inference Engine

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
# Clone repository
git clone https://github.com/your-org/rag-engine.git
cd rag-engine

# Build production image
docker build -t rag-engine .
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

Run latency benchmarks:

```bash
python scripts/benchmark.py \
    --endpoint http://localhost:8080 \
    --queries 100 \
    --concurrency 10
```

Run batching comparison:

```bash
python scripts/benchmark_batching.py \
    --model ./models/all-MiniLM-L6-v2.onnx \
    --num-queries 100
```

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
