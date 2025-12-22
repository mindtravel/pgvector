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

# 创建共享的utils库，避免重复编译
# 被以下库使用：
#   - fusion_cos_topk_warpsort_coarse_lib (generate_sequence_indices_kernel)
#   - fusion_cos_topk_warpsort_lib (generate_sequence_indices_kernel)
#   - integrate_screen_lib (多个utils函数)
# 
# 包含的文件：
#   - utils.cu: 通用工具kernel和host函数
add_library(utils_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/utils.cu
)
target_include_directories(utils_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(utils_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
)

# 创建共享的indexed_gemm库，避免重复编译
# 被以下库使用：
#   - fusion_cos_topk_warpsort_lib
#   - integrate_screen_lib
# 
# 包含的文件（仅保留当前使用的版本）：
#   - indexed_gemm_cos.cu: entry-based版本（当前使用）
#   - indexed_gemm_l2.cu: entry-based版本（当前使用）
# 
add_library(indexed_gemm_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/indexed_gemm/indexed_gemm_cos.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/indexed_gemm/indexed_gemm_l2.cu
)
target_include_directories(indexed_gemm_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(indexed_gemm_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
)
# 链接依赖库：
target_link_libraries(indexed_gemm_lib 
    PUBLIC
    utils_lib
)

# 创建共享的fusion_cos_topk_warpsort粗筛库（只包含粗筛实现）
# 被以下测试使用：
#   - test_fusion_cos_topk (只需要粗筛)
# 
# 包含的文件：
#   - fusion_cos_topk_warpsort.cu: 粗筛实现
add_library(fusion_cos_topk_warpsort_coarse_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/fusion_dist_topk/fusion_cos_topk_warpsort.cu
)
target_include_directories(fusion_cos_topk_warpsort_coarse_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(fusion_cos_topk_warpsort_coarse_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
)
target_link_libraries(fusion_cos_topk_warpsort_coarse_lib 
    PUBLIC
    l2norm_lib 
    utils_lib
    cublas
)

# 创建共享的fusion_dist_topk_warpsort完整库（包含粗筛和精筛）
# 被以下测试使用：
#   - test_fusion_cos_topk_fine
# 
# 包含的文件：
#   - fusion_cos_topk_warpsort.cu: 粗筛实现（已在coarse_lib中，但为了完整性也包含）
#   - fusion_cos_topk_warpsort_fine.cu: entry-based版本的精筛实现
#   - warpsort_topk.cu: warp-level top-k选择
# 注意：
#   - inner_product_utils.cuh 是头文件，不需要添加到库中
add_library(fusion_dist_topk_warpsort_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/fusion_dist_topk/fusion_cos_topk_warpsort.cu
    # ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/fusion_dist_topk/fusion_cos_topk_warpsort_fine.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/warpsortfilter/warpsort_topk.cu
)
target_include_directories(fusion_dist_topk_warpsort_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
# CUDA编译选项说明：
#   CUDA_SEPARABLE_COMPILATION ON: 启用分离编译，允许增量编译和库化
#   CUDA_RESOLVE_DEVICE_SYMBOLS ON: 在库中解析device符号，减少最终链接时间
#   CUDA_ARCHITECTURES 70: 只编译SM 7.0，加快编译速度
set_target_properties(fusion_dist_topk_warpsort_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
)
# 链接依赖库：
#   - l2norm_lib: L2范数计算（fusion_cos_topk_warpsort需要l2_norm_kernel）
#   - indexed_gemm_lib: indexed_gemm实现
#   - cublas: CUDA基础线性代数库
target_link_libraries(fusion_dist_topk_warpsort_lib 
    PUBLIC
    l2norm_lib 
    utils_lib
    indexed_gemm_lib
    cublas
)

# 创建共享的integrate_screen库（供pgvector主项目和测试使用）
# 被以下使用：
#   - pgvector主项目（ivfscanbatch.c 调用 batch_search_pipeline）
#   - test_integrated_coarse_fine (测试integrate_screen.cu)
# 
# 包含的文件：
#   - integrate_screen.cu: 主实现文件（包含batch_search_pipeline函数）
#   - fusion_cos_topk_warpsort.cu: 粗筛实现（cosine距离版本）
#   - fusion_l2_topk_warpsort.cu: 粗筛实现（L2距离版本，integrate_screen.cu需要）
#   - warpsort_topk.cu: warp-level top-k选择（integrate_screen.cu需要select_k）
# 
# 注意：
#   - indexed_gemm_v3_fixed_probe.cu 和 indexed_gemm_v5.cu 已移到 indexed_gemm_lib 中
#   - utils.cu 已移到独立的 utils_lib 中，通过链接依赖引入
#   - inner_product_utils.cuh 是头文件，不需要添加到库中
add_library(integrate_screen_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/integrate_screen/integrate_screen.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/fusion_dist_topk/fusion_cos_topk_warpsort.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/fusion_dist_topk/fusion_l2_topk_warpsort.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/warpsortfilter/warpsort_topk.cu
)
target_include_directories(integrate_screen_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(integrate_screen_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
)
target_link_libraries(integrate_screen_lib 
    PUBLIC
    l2norm_lib 
    utils_lib
    indexed_gemm_lib
    cublas
)

# 创建共享的integrate_screen_pipeline库（供测试使用）
# 被以下使用：
#   - test_integrated_coarse_fine_pipeline (测试流水线版本)
# 
# 包含的文件：
#   - integrate_screen.cu: 主实现文件（包含batch_search_pipeline函数，接受float*参数）
#   - integrate_screen_pipeline.cu: 流水线版本实现（包含batch_search_pipeline函数，接受float**参数）
#   - fusion_cos_topk_warpsort.cu: 粗筛实现（cosine距离版本）
#   - fusion_l2_topk_warpsort.cu: 粗筛实现（L2距离版本，integrate_screen.cu和integrate_screen_pipeline.cu需要）
#   - warpsort_topk.cu: warp-level top-k选择（integrate_screen_pipeline.cu需要select_k）

add_library(integrate_screen_pipeline_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/integrate_screen/integrate_screen.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/integrate_screen/integrate_screen_pipeline.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/fusion_dist_topk/fusion_cos_topk_warpsort.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/fusion_dist_topk/fusion_l2_topk_warpsort.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/warpsortfilter/warpsort_topk.cu
)
target_include_directories(integrate_screen_pipeline_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(integrate_screen_pipeline_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
)
target_link_libraries(integrate_screen_pipeline_lib 
    PUBLIC
    l2norm_lib 
    utils_lib
    indexed_gemm_lib
    cublas
)

# 添加测试子目录
add_subdirectory(test_vector_normalizer)
add_subdirectory(test_print_cuda)
add_subdirectory(test_matrix_multiply)
add_subdirectory(test_cos_distance)
add_subdirectory(test_fusion_cos_topk)
# add_subdirectory(test_fusion_cos_topk_fine)
add_subdirectory(test_warpsort)
add_subdirectory(test_bitonic)
add_subdirectory(test_stream_pass_data)
add_subdirectory(test_final_topk)
add_subdirectory(test_integrated_coarse_fine)
add_subdirectory(test_integrated_coarse_fine_pipeline)
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
        # test_fusion_cos_topk_fine
        test_warpsort
        test_bitonic
        test_stream_pass_data
        test_final_topk
        test_integrated_coarse_fine
        test_integrated_coarse_fine_pipeline
    COMMENT "运行所有单元测试"
)
