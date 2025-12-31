cmake_minimum_required(VERSION 3.18)
project(test_ivfflat_search_pipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 统一架构为70，与其他测试保持一致，加快编译速度
set(CMAKE_CUDA_ARCHITECTURES 70)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../cuda)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

set(SOURCES
    test_ivfflat_search_pipeline.cu
    ../common/test_utils.cu
)

add_executable(test_ivfflat_search_pipeline ${SOURCES})

target_link_libraries(test_ivfflat_search_pipeline
    ${CUDA_LIBRARIES}
    cublas
    curand
    l2norm_lib
    indexed_gemm_lib
    ivf_search_lib  # 包含 batch_search_pipeline (device指针版本)
)

set_target_properties(test_ivfflat_search_pipeline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

enable_testing()
add_test(NAME test_ivfflat_search_pipeline COMMAND test_ivfflat_search_pipeline)
