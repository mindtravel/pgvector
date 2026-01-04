cmake_minimum_required(VERSION 3.18)
project(test_ivfflat_search_lookup LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 统一架构为70，与其他测试保持一致，加快编译速度
set(CMAKE_CUDA_ARCHITECTURES 70)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../cuda)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

# Source files
set(SOURCES
    test_ivfflat_search_lookup.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/kmeans/kmeans.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/kmeans/kmeans_minibatch.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/kmeans/kmeans_utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/kmeans/kmeans_reorder.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/kmeans/kmeans_reorder_utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/test_utils.cu
)

# Create executable
add_executable(test_ivfflat_search_lookup ${SOURCES})

# Ensure dependencies are built first
add_dependencies(test_ivfflat_search_lookup utils_lib l2norm_lib indexed_gemm_lib fusion_dist_topk_warpsort_lib ivf_search_lib)

# Link libraries
target_link_libraries(test_ivfflat_search_lookup
    PRIVATE
    utils_lib
    l2norm_lib
    ${CUDA_LIBRARIES}
    cublas
    curand
    indexed_gemm_lib
    ivf_search_lib
    fusion_dist_topk_warpsort_lib
)

# CUDA properties
set_target_properties(test_ivfflat_search_lookup PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Add test
enable_testing()
add_test(NAME test_ivfflat_search_lookup COMMAND test_ivfflat_search_lookup)

