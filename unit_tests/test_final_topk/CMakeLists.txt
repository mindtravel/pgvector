cmake_minimum_required(VERSION 3.10)
project(test_final_topk)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找CUDA
find_package(CUDA REQUIRED)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../cuda)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

# 源文件
set(SOURCES
    test_final_topk.cu
    ../common/test_utils.cu
    ../../cuda/fine_screen_top_n/fine_screen_top_n.cu
    ../../cuda/l2norm.cu
)

# 创建可执行文件
add_executable(test_final_topk ${SOURCES})

# 链接CUDA库
target_link_libraries(test_final_topk 
    ${CUDA_LIBRARIES}
    cublas
    curand
)

# 编译选项
set_target_properties(test_final_topk PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
