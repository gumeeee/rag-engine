#!/bin/bash
# RAG Engine C++ Benchmark Script
# Usage: ./scripts/benchmark-cpp.sh [--build] [--gpu]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="rag-engine:benchmark"
CONTAINER_NAME="rag-engine-bench"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build     Build Docker image before running"
    echo "  --gpu       Enable GPU passthrough (requires NVIDIA GPU)"
    echo "  --cpu       Run in CPU-only mode (default)"
    echo "  --clean     Stop and remove container after benchmark"
    echo "  --help      Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --build --gpu      # Build and run with GPU"
    echo "  $0 --gpu              # Run existing image with GPU"
    echo "  $0 --cpu              # Run existing image with CPU"
}

# Parse arguments
BUILD=false
GPU=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --build) BUILD=true; shift ;;
        --gpu) GPU=true; shift ;;
        --cpu) GPU=false; shift ;;
        --clean) CLEAN=true; shift ;;
        --help) usage; exit 0 ;;
        *) error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Build Docker image
if [ "$BUILD" = true ]; then
    log "Building Docker image..."
    cd "$PROJECT_DIR"
    docker build -t "$IMAGE_NAME" .
    log "Build complete!"
fi

# Stop existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "Stopping existing container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# Docker run options
DOCKER_OPTS="-d --name $CONTAINER_NAME -p 8080:8080"
if [ "$GPU" = true ]; then
    if docker info 2>/dev/null | grep -q "nvidia"; then
        DOCKER_OPTS="$DOCKER_OPTS --gpus all"
        log "GPU passthrough enabled"
    else
        warn "No NVIDIA GPU detected, running in CPU mode"
        GPU=false
    fi
fi

# Run container
log "Starting container..."
cd "$PROJECT_DIR"

# Create data directories
mkdir -p data/corpus data/models

docker run $DOCKER_OPTS \
    -v "$PROJECT_DIR/data:/app/data" \
    -e RAG_LOG_LEVEL=info \
    "$IMAGE_NAME" &

CONTAINER_ID=$!
log "Container started (ID: $CONTAINER_ID)"

# Wait for server to be ready
log "Waiting for server to start..."
for i in {1..30}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        log "Server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        error "Server failed to start after 30 seconds"
        docker logs "$CONTAINER_NAME"
        exit 1
    fi
    sleep 1
done

# Run benchmarks
log ""
log "========================================"
log "Running C++ RAG Engine Benchmarks"
log "========================================"
log ""

# Health check
log "1. Health Check"
curl -s http://localhost:8080/health | python3 -m json.tool
log ""

# Test queries
log "2. Test Queries"
QUERIES=(
    "What is the Transformer architecture?"
    "How does BERT work?"
    "What is RAG?"
    "Explain vector similarity search"
    "What is attention mechanism?"
)

for query in "${QUERIES[@]}"; do
    START=$(date +%s%N)
    RESPONSE=$(curl -s -X POST http://localhost:8080/query \
        -H "Content-Type: application/json" \
        -d "{\"query_text\": \"$query\", \"top_k\": 3}")
    END=$(date +%s%N)
    ELAPSED=$(( (END - START) / 1000000 ))

    echo "Query: $query"
    echo "Latency: ${ELAPSED}ms"
    echo "Response: $(echo $RESPONSE | python3 -c 'import sys,json; print(f"{len(json.load(sys.stdin).get(\"results\",[]))} results")')"
    echo ""
done

# Load test (concurrent queries)
log "3. Concurrent Load Test (10 queries)"
START=$(date +%s%N)

for i in {1..10}; do
    curl -s -X POST http://localhost:8080/query \
        -H "Content-Type: application/json" \
        -d '{"query_text": "What is machine learning?", "top_k": 5}' > /dev/null &
done

wait
END=$(date +%s%N)
TOTAL=$(( (END - START) / 1000000 ))
AVG=$(( TOTAL / 10 ))

echo "Total time: ${TOTAL}ms"
echo "Average per query: ${AVG}ms"
echo "Throughput: $(( 10000 / TOTAL )) QPS"
log ""

# Metrics endpoint
log "4. Prometheus Metrics"
curl -s http://localhost:8080/metrics | head -20
log ""

# Summary
log "========================================"
log "Benchmark Complete!"
log "========================================"
log ""
log "To test manually:"
log "  curl -X POST http://localhost:8080/query \\"
log "    -H 'Content-Type: application/json' \\"
log "    -d '{\"query_text\": \"Your question here\", \"top_k\": 5}'"
log ""
log "To stop: docker stop $CONTAINER_NAME"
log "To view logs: docker logs $CONTAINER_NAME"

# Clean up if requested
if [ "$CLEAN" = true ]; then
    log "Cleaning up..."
    docker stop "$CONTAINER_NAME" > /dev/null
    docker rm "$CONTAINER_NAME" > /dev/null
    log "Done!"
fi
