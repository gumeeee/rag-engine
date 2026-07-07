#!/bin/bash
# Minimal CPU-only build script
# Usage: ./scripts/build-cpu.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Building RAG Engine (CPU-only)..."

# Install dependencies
if command -v apt-get &> /dev/null; then
    echo "Installing dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
        cmake build-essential git \
        libuv1-dev libprotobuf-dev protobuf-compiler \
        libgoogle-glog-dev libspdlog-dev libnlohmann-json-dev \
        libgtest-dev libgmock-dev pkg-config
fi

cd "$PROJECT_DIR"

# Create build directory
rm -rf build-cpu
mkdir -p build-cpu && cd build-cpu

# Generate protobuf
echo "Generating protobuf..."
protoc --cpp_out=include/ proto/config.proto proto/rag_service.proto 2>/dev/null || true

# Configure using the CPU CMakeLists
echo "Configuring CMake..."
cmake "$PROJECT_DIR" -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"

# Build
echo "Building..."
cmake --build . -j$(nproc)

echo ""
echo "=================================="
echo "Build complete!"
echo "Binary: $PROJECT_DIR/build-cpu/rag-engine"
echo "=================================="
