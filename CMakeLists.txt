cmake_minimum_required(VERSION 3.20)
project(sgemm)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

enable_language(CUDA) # Enable CUDA support, set this to your CUDA standard if needed.

set(INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
set(LIB_DIRS ${PROJECT_SOURCE_DIR}/lib)

add_executable(sgemm ${LIB_DIRS}/sgemm.cu)
target_include_directories(sgemm PRIVATE ${INCLUDE_DIRS})