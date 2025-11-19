cmake_minimum_required(VERSION 3.18)
project(test_final_topk LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA架构（使用与根目录一致的架构，避免多架构编译）
# set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../cuda)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

# 源文件
set(SOURCES
    test_final_topk.cu
    ../common/test_utils.cu
    ../../cuda/fine_screen_top_n/fine_screen_top_n.cu
    ../../cuda/fine_screen_top_n/fine_screen_top_n_v1.cu
    ../../cuda/fusion_cos_topk/fusion_cos_topk_warpsort_fine.cu
    ../../cuda/fusion_cos_topk/fusion_cos_topk_warpsort_fine_v1.cu
    ../../cuda/indexed_gemm/indexed_gemm.cu
    ../../cuda/indexed_gemm/indexed_gemm_v1.cu
    ../../cuda/warpsortfilter/warpsort_topk.cu
    ../../cuda/l2norm/l2norm.cu
)

# 创建可执行文件
add_executable(test_final_topk ${SOURCES})

# 链接CUDA库
find_package(CUDAToolkit REQUIRED)
target_link_libraries(test_final_topk 
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)

# 编译选项
set_target_properties(test_final_topk PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
