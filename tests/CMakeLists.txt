cmake_minimum_required(VERSION 3.28)

# GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Test sources
file(GLOB_RECURSE TEST_SOURCES "*.cpp")

add_executable(rag-engine-tests ${TEST_SOURCES})
target_link_libraries(rag-engine-tests PRIVATE
    rag_engine_lib
    gtest_main
    gtest
    pthread
)

include(GoogleTest)
gtest_discover_tests(rag-engine-tests)
