cmake_minimum_required(VERSION 3.22)
project(rag-engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)

# Find packages
find_package(CUDA)
find_package(faiss REQUIRED)
find_package(libuv REQUIRED)  # Changed from 'uv' to 'libuv'
find_package(protobuf CONFIG REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# ONNX Runtime (installed via pip, not vcpkg)
set(ONNXRUNTIME_ROOT "/usr/local" CACHE PATH "ONNX Runtime installation path")
find_library(ONNXRUNTIME_LIB NAMES onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib NO_DEFAULT_PATH)
if(ONNXRUNTIME_LIB)
    message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIB}")
    set(ONNXRUNTIME_FOUND TRUE)
else()
    message(STATUS "ONNX Runtime not found in ${ONNXRUNTIME_ROOT}/lib - will link at runtime")
    set(ONNXRUNTIME_FOUND FALSE)
endif()

# CUDA is optional
if(CUDA_FOUND)
    message(STATUS "CUDA found - GPU support enabled")
    enable_language(CUDA)
else()
    message(STATUS "CUDA not found - CPU-only mode")
endif()

# Source files
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
    ${libuv_LIBRARIES}
    protobuf::libprotobuf
    faiss
)

if(ONNXRUNTIME_FOUND)
    target_include_directories(rag_engine_lib PRIVATE ${ONNXRUNTIME_ROOT}/include)
    target_link_libraries(rag_engine_lib PRIVATE ${ONNXRUNTIME_LIB})
endif()

if(CUDA_FOUND)
    target_link_libraries(rag_engine_lib PRIVATE CUDA::cudart)
endif()

# Generate protobuf headers
find_program(PROTOC protoc)
if(PROTOC AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/proto")
    execute_process(
        COMMAND ${PROTOC} --cpp_out=${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/proto/config.proto
                ${CMAKE_CURRENT_SOURCE_DIR}/proto/rag_service.proto
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

# Main executable
add_executable(rag-engine src/main.cpp)
target_link_libraries(rag-engine PRIVATE rag_engine_lib)

# Installation
install(TARGETS rag-engine rag_engine_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
