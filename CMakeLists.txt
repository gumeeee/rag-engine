cmake_minimum_required(VERSION 3.28)
project(rag-engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find packages
find_package(CUDA)
find_package(ONNXRuntime CONFIG)
find_package(faiss REQUIRED)
find_package(uv REQUIRED)
find_package(protobuf CONFIG REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# CUDA is optional - enable GPU support if available
if(CUDA_FOUND)
    message(STATUS "CUDA found - GPU support enabled")
    enable_language(CUDA)
else()
    message(STATUS "CUDA not found - CPU-only mode")
endif()

# Source groups
file(GLOB_RECURSE COMMON_SOURCES "src/common/*.cpp")
file(GLOB_RECURSE EMBEDDING_SOURCES "src/embedding/*.cpp")
file(GLOB_RECURSE SEARCH_SOURCES "src/search/*.cpp")
file(GLOB_RECURSE PIPELINE_SOURCES "src/pipeline/*.cpp")
file(GLOB_RECURSE API_SOURCES "src/api/*.cpp")
file(GLOB_RECURSE OBSERVABILITY_SOURCES "src/observability/*.cpp")

# Main library
add_library(rag_engine_lib STATIC
    ${COMMON_SOURCES}
    ${EMBEDDING_SOURCES}
    ${SEARCH_SOURCES}
    ${PIPELINE_SOURCES}
    ${API_SOURCES}
    ${OBSERVABILITY_SOURCES}
)

target_include_directories(rag_engine_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(rag_engine_lib PUBLIC
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

target_link_libraries(rag_engine_lib PRIVATE
    onnxruntime::onnxruntime
    faiss
    uv::uv
    protobuf::libprotobuf
)

if(CUDA_FOUND)
    target_link_libraries(rag_engine_lib PRIVATE CUDA::cudart)
endif()

# Generate protobuf headers
protobuf_generate(TARGET rag_engine_lib PROTOS proto/rag_service.proto)
protobuf_generate(TARGET rag_engine_lib PROTOS proto/config.proto LANGUAGE cpp)

# Main executable
add_executable(rag-engine src/main.cpp)
target_link_libraries(rag-engine PRIVATE rag_engine_lib)

# Tests
enable_testing()
add_subdirectory(tests)

# Installation
install(TARGETS rag-engine rag_engine_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config/ DESTINATION etc/rag-engine)
