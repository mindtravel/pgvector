cmake_minimum_required(VERSION 3.10)
project(test_integrated_coarse_fine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

# 统一架构为70，与其他测试保持一致，加快编译速度
set(CMAKE_CUDA_ARCHITECTURES 70)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../cuda)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

set(SOURCES
    test_integrated_coarse_fine.cu
    ../common/test_utils.cu
    # 使用共享库，不再重复编译
    # ../../cuda/l2norm/l2norm.cu
    # ../../cuda/fusion_cos_topk/fusion_cos_topk_warpsort.cu
    # 需要编译fusion_cos_topk_warpsort_fine.cu（fine_screen_top_n.cu依赖它）
    ../../cuda/fusion_cos_topk/fusion_cos_topk_warpsort_fine.cu
    # 需要编译indexed_gemm.cu（fusion_cos_topk_warpsort_fine.cu依赖indexed_inner_product_kernel）
    ../../cuda/indexed_gemm/indexed_gemm.cu
    ../../cuda/fine_screen_top_n/fine_screen_top_n.cu
    ../../cuda/integrate_screen/integrate_screen.cu
)

add_executable(test_integrated_coarse_fine ${SOURCES})

# 链接共享库
target_link_libraries(test_integrated_coarse_fine 
    fusion_cos_topk_warpsort_lib
    l2norm_lib
)

target_link_libraries(test_integrated_coarse_fine
    ${CUDA_LIBRARIES}
    cublas
    curand
)

set_target_properties(test_integrated_coarse_fine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

enable_testing()
add_test(NAME test_integrated_coarse_fine COMMAND test_integrated_coarse_fine)
