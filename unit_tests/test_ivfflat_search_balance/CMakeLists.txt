cmake_minimum_required(VERSION 3.18)
project(test_ivfflat_search_balance LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 统一架构为70，与其他测试保持一致，加快编译速度
set(CMAKE_CUDA_ARCHITECTURES 70)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../cuda)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

set(SOURCES
    test_ivfflat_search_balance.cu
    ../common/test_utils.cu
    # 使用共享库，不再重复编译
    # ../../cuda/l2norm/l2norm.cu - 在 l2norm_lib 中
    # 需要编译fusion_cos_topk_warpsort_fine.cu（fine_screen_top_n.cu依赖它）
    ../../cuda/fusion_cos_topk/fusion_cos_topk_warpsort_fine.cu
    # indexed_gemm.cu 为弃用版本，需要单独编译（不在 indexed_gemm_lib 中）
    ../../cuda/indexed_gemm/indexed_gemm.cu
    ../../cuda/fine_screen_top_n/fine_screen_top_n.cu
)

add_executable(test_ivfflat_search_balance ${SOURCES})

target_link_libraries(test_ivfflat_search_balance
    ${CUDA_LIBRARIES}
    cublas
    curand
    l2norm_lib
    indexed_gemm_lib
    ivf_search_lib  # 包含 ivf_search.cu, fusion_cos_topk_warpsort.cu, warpsort_topk.cu
    fusion_cos_topk_warpsort_lib
)

set_target_properties(test_ivfflat_search_balance PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

enable_testing()
add_test(NAME test_ivfflat_search_balance COMMAND test_ivfflat_search_balance)
