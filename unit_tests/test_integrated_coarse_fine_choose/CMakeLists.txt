cmake_minimum_required(VERSION 3.18)
project(test_integrated_coarse_fine_choose LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 统一架构为70，与其他测试保持一致，加快编译速度
set(CMAKE_CUDA_ARCHITECTURES 70)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../cuda)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

set(SOURCES
    test_integrated_coarse_fine_choose.cu
    ../common/test_utils.cu
    # 使用共享库，不再重复编译
    # ../../cuda/l2norm/l2norm.cu - 在 l2norm_lib 中
    # ../../cuda/integrate_screen/integrate_screen.cu - 在 integrate_screen_lib 中
    # ../../cuda/fusion_cos_topk/fusion_cos_topk_warpsort.cu - 在 integrate_screen_lib 中
    # indexed_gemm_v3_fixed_probe.cu, indexed_gemm_v5.cu - 在 indexed_gemm_lib 中
    # warpsort_topk - 在 integrate_screen_lib 中
)

add_executable(test_integrated_coarse_fine_choose ${SOURCES})

target_link_libraries(test_integrated_coarse_fine_choose
    ${CUDA_LIBRARIES}
    cublas
    curand
    l2norm_lib
    indexed_gemm_lib
    integrate_screen_choose_lib  # 包含 fusion_cos_topk_warpsort 和 warpsort_topk
)

set_target_properties(test_integrated_coarse_fine_choose PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

enable_testing()
add_test(NAME test_integrated_coarse_fine_choose COMMAND test_integrated_coarse_fine_choose)
