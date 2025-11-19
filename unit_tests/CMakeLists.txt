cmake_minimum_required(VERSION 3.18)
project(pgvector_unit_tests LANGUAGES CXX CUDA)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 启用CUDA语言
enable_language(CUDA)

# CUDA编译选项
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_STANDARD 17)
# 优化编译选项：使用快速数学、减少调试信息、启用并行编译
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math -lineinfo -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE --expt-relaxed-constexpr")
# 设置并行编译线程数（nvcc的--threads选项）
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 0")

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
# 链接l2norm库（fusion_cos_topk_warpsort需要l2_norm_kernel）
target_link_libraries(fusion_cos_topk_warpsort_lib l2norm_lib)

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
add_subdirectory(test_integrated_coarse_fine)

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
        test_warpsort
        test_bitonic
        test_stream_pass_data
        test_final_topk
        test_integrated_coarse_fine
    COMMENT "运行所有单元测试"
)
