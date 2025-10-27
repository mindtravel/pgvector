cmake_minimum_required(VERSION 3.10)
project(test_stream_pass_data)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 查找CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../cuda)

# 源文件
set(SOURCES
    test_stream_pass_data.cu
    stream_pass_data.cu
)

# 创建可执行文件
add_executable(test_stream_pass_data ${SOURCES})

# 链接CUDA库
target_link_libraries(test_stream_pass_data 
    ${CUDA_LIBRARIES}
    cudart
    cublas
)

# 设置编译选项
target_compile_options(test_stream_pass_data PRIVATE
    -O3
    -std=c++17
)

# CUDA编译选项
set_target_properties(test_stream_pass_data PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
