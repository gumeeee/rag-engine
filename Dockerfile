# syntax=docker/dockerfile:1.4
# RAG Inference Engine - Production Dockerfile

# Stage 1: Build
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libuv1-dev \
    libprotobuf-dev \
    protobuf-compiler \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Build dependencies
FROM base AS builder

WORKDIR /build

# Install vcpkg
RUN git clone https://github.com/Microsoft/vcpkg.git /vcpkg && \
    /vcpkg/bootstrap-vcpkg.sh

# Copy build files
COPY vcpkg.json CMakeLists.txt ./
COPY proto/ ./proto/
COPY include/ ./include/
COPY src/ ./src/
COPY config/ ./config/

# Build dependencies with vcpkg
ENV VCPKG_ROOT=/vcpkg
RUN /vcpkg/vcpkg install

# Configure and build
RUN cmake . -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --config Release -j$(nproc)

# Stage 3: Runtime
FROM base AS runtime

WORKDIR /app

# Copy binary and configs
COPY --from=builder /build/rag-engine ./
COPY config/ ./config/
COPY models/ ./models/
COPY data/ ./data/

# Environment variables
ENV RAG_CONFIG_PATH=/app/config/default_config.pb.txt
ENV CUDA_VISIBLE_DEVICES=0
ENV RAG_LOG_LEVEL=info

# Expose HTTP port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

ENTRYPOINT ["./rag-engine"]
CMD ["--config", "/app/config/default_config.pb.txt"]
