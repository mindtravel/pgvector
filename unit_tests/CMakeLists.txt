cmake_minimum_required(VERSION 3.18)
project(pgvector_unit_tests LANGUAGES CXX CUDA)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 启用CUDA语言
enable_language(CUDA)

# 优化编译选项：加快编译速度
# -O0: 无优化，最快编译
# -use_fast_math: 快速数学库
# -lineinfo: 保留行号信息（用于调试，但比-g快）
# -w: 禁用所有警告，减少编译输出
# -Xcompiler -w: 禁用主机编译器警告
# --disable-warnings: 禁用CUDA警告
# --threads 0: 使用所有可用CPU核心并行编译
# --expt-relaxed-constexpr: 放宽constexpr限制
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -use_fast_math -lineinfo -w -Xcompiler -w --disable-warnings -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE --expt-relaxed-constexpr")

# CUDA编译选项
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_STANDARD 17)
# 优化编译选项：使用快速数学、减少调试信息、启用并行编译
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math -lineinfo -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE --expt-relaxed-constexpr")
# 设置并行编译线程数（nvcc的--threads选项）
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 0")
# 禁用 ptxas 详细输出信息（移除 -v 选项即可禁用输出）
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

# CUDA包含路径
set(CUDA_INCLUDE_DIRS
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

# CUDA库
set(CUDA_LIBRARIES
    ${CUDA_LIBRARIES}
    cudart
    curand
    cublas
    cusparse
)

# 通用头文件包含
set(COMMON_HEADERS "-include ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/pch.h")

# 创建共享的l2norm库，避免重复编译
# 注意：使用可分离编译以便其他目标可以链接device符号
add_library(l2norm_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/l2norm/l2norm.cu
)
target_include_directories(l2norm_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(l2norm_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    # 优化：减少device link时间
    CUDA_ARCHITECTURES 70
)

# 创建共享的fusion_cos_topk_warpsort库，避免重复编译
# 被test_fusion_cos_topk和test_integrated_coarse_fine使用
add_library(fusion_cos_topk_warpsort_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/fusion_cos_topk/fusion_cos_topk_warpsort.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/fusion_cos_topk/fusion_cos_topk_warpsort_fine_v3_fixed_probe.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/indexed_gemm/indexed_gemm_v3_fixed_probe.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/warpsortfilter/warpsort_topk.cu
)
target_include_directories(fusion_cos_topk_warpsort_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(fusion_cos_topk_warpsort_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
)
# 链接l2norm库和cublas（fusion_cos_topk_warpsort需要l2_norm_kernel和cublas）
target_link_libraries(fusion_cos_topk_warpsort_lib l2norm_lib cublas)

# 添加测试子目录
add_subdirectory(test_vector_normalizer)
add_subdirectory(test_print_cuda)
add_subdirectory(test_matrix_multiply)
add_subdirectory(test_cos_distance)
add_subdirectory(test_fusion_cos_topk)
add_subdirectory(test_fusion_cos_topk_fine)
add_subdirectory(test_warpsort)
add_subdirectory(test_bitonic)
add_subdirectory(test_stream_pass_data)
add_subdirectory(test_final_topk)
add_subdirectory(test_fusion_cos_topk_fine_v3_fixed_probe)
add_subdirectory(test_integrated_coarse_fine)
add_subdirectory(test_select_k)

# 统一测试目标
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS 
        test_vector_normalizer 
        test_print_cuda
        test_matrix_multiply 
        test_cos_distance 
        test_fusion_cos_topk
        test_fusion_cos_topk_fine
        test_fusion_cos_topk_fine_v3_fixed_probe
        test_warpsort
        test_bitonic
        test_stream_pass_data
        test_final_topk
        test_integrated_coarse_fine
    COMMENT "运行所有单元测试"
)
