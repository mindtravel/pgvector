cmake_minimum_required(VERSION 3.18)
project(test_select_k LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../cuda)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

# 源文件
set(SOURCES
    test_select_k.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/test_utils.cu
)

# 创建可执行文件
add_executable(test_select_k ${SOURCES})

# 链接库
target_link_libraries(test_select_k
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

# CUDA 编译选项
set_target_properties(test_select_k PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

