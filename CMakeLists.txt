cmake_minimum_required(VERSION 3.18)
project(turtle.cpp)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

set(TURTLE_CPP_SRC
    src/gguf.cpp
    src/main.cpp
    src/quantize.cpp
)

add_executable(turtle_cpp ${TURTLE_CPP_SRC})
