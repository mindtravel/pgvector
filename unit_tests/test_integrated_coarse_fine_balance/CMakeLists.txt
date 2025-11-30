cmake_minimum_required(VERSION 3.18)
project(test_integrated_coarse_fine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 统一架构为70，与其他测试保持一致，加快编译速度
set(CMAKE_CUDA_ARCHITECTURES 70)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../cuda)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

set(SOURCES
    test_integrated_coarse_fine.cu
    ../common/test_utils.cu
    # 使用共享库，不再重复编译
    # ../../cuda/l2norm/l2norm.cu - 在 l2norm_lib 中
    # ../../cuda/fusion_cos_topk/fusion_cos_topk_warpsort.cu - 在 fusion_cos_topk_warpsort_lib 中
    # 需要编译fusion_cos_topk_warpsort_fine.cu（fine_screen_top_n.cu依赖它）
    ../../cuda/fusion_cos_topk/fusion_cos_topk_warpsort_fine.cu
    # 需要编译indexed_gemm.cu（fusion_cos_topk_warpsort_fine.cu依赖indexed_inner_product_kernel）
    ../../cuda/indexed_gemm/indexed_gemm.cu
    # 固定probe版本的精筛实现 - 已在 fusion_cos_topk_warpsort_lib 中，不再重复编译
    # ../../cuda/fusion_cos_topk/fusion_cos_topk_warpsort_fine_v3_fixed_probe.cu
    # ../../cuda/indexed_gemm/indexed_gemm_v3_fixed_probe.cu - 已在 fusion_cos_topk_warpsort_lib 中
    # warpsort_topk 用于规约 - 已在 fusion_cos_topk_warpsort_lib 中
    # ../../cuda/warpsortfilter/warpsort_topk.cu
    ../../cuda/fine_screen_top_n/fine_screen_top_n.cu
    ../../cuda/integrate_screen/integrate_screen.cu
)

add_executable(test_integrated_coarse_fine ${SOURCES})

target_link_libraries(test_integrated_coarse_fine
    ${CUDA_LIBRARIES}
    cublas
    curand
    l2norm_lib
    fusion_cos_topk_warpsort_lib
)

set_target_properties(test_integrated_coarse_fine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

enable_testing()
add_test(NAME test_integrated_coarse_fine COMMAND test_integrated_coarse_fine)
