# syntax=docker/dockerfile:1.4
# RAG Inference Engine - Production Dockerfile

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# Install all dependencies in one layer
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    curl \
    zip \
    unzip \
    tar \
    wget \
    pkg-config \
    libuv1-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    g++ \
    ninja-build \
    libomp-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install vcpkg (lightweight build)
RUN git clone --depth 1 https://github.com/Microsoft/vcpkg.git /vcpkg && \
    /vcpkg/bootstrap-vcpkg.sh && rm -rf /vcpkg/.git

# Set vcpkg in PATH
ENV VCPKG_ROOT=/vcpkg
ENV PATH=/vcpkg:$PATH

# Copy source files
COPY vcpkg.json CMakeLists.txt ./
COPY proto/ ./proto/
COPY include/ ./include/
COPY src/ ./src/
COPY tests/ ./tests/
COPY config/ ./config/

# Build dependencies and project (vcpkg.json defines packages)
RUN /vcpkg/vcpkg install --triplet x64-linux && \
    cmake . -DCMAKE_BUILD_TYPE=Release \
           -DCMAKE_TOOLCHAIN_FILE=/vcpkg/scripts/buildsystems/vcpkg.cmake \
           -DVCPKG_TARGET_TRIPLET=x64-linux && \
    cmake --build . --config Release -j$(nproc)

# Runtime setup
COPY config/ ./config/

# Environment
ENV RAG_CONFIG_PATH=/app/config/default_config.pb.txt
ENV RAG_LOG_LEVEL=info

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

EXPOSE 8080

ENTRYPOINT ["./rag-engine"]
CMD ["--config", "/app/config/default_config.pb.txt"]
