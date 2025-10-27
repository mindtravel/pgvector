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
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE --expt-relaxed-constexpr")

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

# 添加测试子目录
add_subdirectory(test_vector_normalizer)
add_subdirectory(test_print_cuda)
add_subdirectory(test_matrix_multiply)
add_subdirectory(test_cos_distance)
add_subdirectory(test_fusion_cos_topk)
add_subdirectory(test_warpsort)
add_subdirectory(test_bitonic)
add_subdirectory(test_stream_pass_data)

# 统一测试目标
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS 
        test_vector_normalizer 
        test_print_cuda
        test_matrix_multiply 
        test_cos_distance 
        test_fusion_cos_topk
        test_warpsort
        test_bitonic
        test_stream_pass_data
    COMMENT "运行所有单元测试"
)