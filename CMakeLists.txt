cmake_minimum_required(VERSION 3.20)
project(sgemm)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

enable_language(CUDA) # Enable CUDA support, set this to your CUDA standard if needed.

set(INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
set(LIB_DIRS ${PROJECT_SOURCE_DIR}/lib)


# add_library(sgemm_core STATIC ${INCLUDE_DIRS} ${LIB_DIRS}/sgemm.cu)

include_directories(${INCLUDE_DIRS})

option(matmul "matmul" ON)
option(softmax "softmax" OFF)

if (matmul)
    add_executable(sgemm ${LIB_DIRS}/sgemm.cu)
    target_link_libraries(sgemm cublas)
endif ()

if (softmax)
    add_executable(softmax ${LIB_DIRS}/softmax.cu)
    target_link_libraries(softmax)
endif ()