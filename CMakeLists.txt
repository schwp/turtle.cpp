cmake_minimum_required(VERSION 3.18)
project(turtle.cpp)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

include(FetchContent)

FetchContent_Declare(
    libtorch
    URL https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.12.1%2Bcpu.zip
    # PyTorch CPU version (lighter than the full version with CUDA)
)
FetchContent_MakeAvailable(libtorch)

set(CMAKE_PREFIX_PATH "${libtorch_SOURCE_DIR}" ${CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

set(TURTLE_CPP_SRC
    src/gguf.cpp
    src/main.cpp
    src/quantize.cpp
    src/model.cpp
    src/ops.cpp
    src/cli.cpp
)

enable_testing()

add_executable(turtle_cpp ${TURTLE_CPP_SRC})

add_executable(test_ops tests/test_ops.cpp src/ops.cpp)
target_link_libraries(test_ops PRIVATE GTest::gtest_main "${TORCH_LIBRARIES}")

include(GoogleTest)
gtest_discover_tests(test_ops)
