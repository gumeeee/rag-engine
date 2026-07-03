#!/bin/bash
# RAG Engine - Build & Test Script
# Usage: ./scripts/build.sh [dev|prod|test|clean]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

usage() {
    echo "Usage: $0 [dev|prod|test|clean|help]"
    echo ""
    echo "  dev     - Build development image with Docker Compose"
    echo "  prod    - Build production Docker image"
    echo "  test    - Run tests in Docker container"
    echo "  clean   - Remove build artifacts"
    echo "  help    - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 dev        # Start development environment"
    echo "  $0 prod       # Build production image"
    echo "  $0 test       # Run unit tests"
}

# Build production Docker image
build_prod() {
    log_info "Building production Docker image..."
    cd "$PROJECT_DIR"

    docker build \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        -t rag-engine:prod \
        -f Dockerfile .

    log_info "Production image built: rag-engine:prod"
    docker images | grep rag-engine
}

# Build with Docker Compose (development)
build_dev() {
    log_info "Starting development environment..."
    cd "$PROJECT_DIR"

    # Check if nvidia-docker is available
    if docker info | grep -q "nvidia"; then
        log_info "NVIDIA GPU support detected"
    else
        log_warn "NVIDIA GPU not detected - GPU features disabled"
    fi

    docker-compose up --build
}

# Run tests
run_tests() {
    log_info "Running tests in Docker..."
    cd "$PROJECT_DIR"

    # Build test image
    docker build \
        --target builder \
        -t rag-engine:test \
        -f Dockerfile .

    # Run tests
    docker run --rm rag-engine:test \
        ./rag-engine-tests || {
        log_error "Tests failed!"
        exit 1
    }

    log_info "All tests passed!"
}

# Clean build artifacts
clean() {
    log_info "Cleaning build artifacts..."
    cd "$PROJECT_DIR"

    rm -rf build/
    docker rmi rag-engine:prod rag-engine:test 2>/dev/null || true

    log_info "Clean complete"
}

# Generate corpus (placeholder)
generate_corpus() {
    log_info "Generating sample corpus..."

    # Create sample documents
    mkdir -p "$PROJECT_DIR/data/corpus"

    cat > "$PROJECT_DIR/data/corpus/sample1.txt" << 'EOF'
Attention is all you need. The Transformer architecture was introduced in 2017
and has become the foundation for many state-of-the-art models. It uses
self-attention mechanisms to process input sequences in parallel.
EOF

    cat > "$PROJECT_DIR/data/corpus/sample2.txt" << 'EOF'
BERT uses bidirectional transformers for language understanding. It was
pre-trained on a large corpus using masked language modeling and next
sentence prediction objectives.
EOF

    cat > "$PROJECT_DIR/data/corpus/sample3.txt" << 'EOF'
GPT models are autoregressive transformers trained on large amounts of
text data. They use next-token prediction during training and can generate
coherent text given a prompt.
EOF

    log_info "Sample corpus created in data/corpus/"
    log_info "To generate embeddings, run:"
    echo "  python3 scripts/generate_embeddings.py \\"
    echo "    --corpus-dir ./data/corpus \\"
    echo "    --output ./data/corpus.corpus"
}

# Download ONNX model (placeholder)
download_model() {
    log_info "Downloading ONNX model..."

    mkdir -p "$PROJECT_DIR/models"

    # Download all-MiniLM-L6-v2 ONNX model
    # Note: This is a placeholder - you may need to convert from PyTorch
    log_warn "Model download not implemented - using CPU fallback"
    log_info "To use GPU acceleration:"
    echo "  1. Download model from HuggingFace"
    echo "  2. Convert to ONNX format"
    echo "  3. Place in ./models/all-MiniLM-L6-v2.onnx"
}

# Main
case "${1:-help}" in
    dev)
        build_dev
        ;;
    prod)
        build_prod
        ;;
    test)
        run_tests
        ;;
    clean)
        clean
        ;;
    corpus)
        generate_corpus
        ;;
    model)
        download_model
        ;;
    help|*)
        usage
        ;;
esac
