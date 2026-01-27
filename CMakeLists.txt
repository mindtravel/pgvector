cmake_minimum_required(VERSION 3.18)

set(CMAKE_C_COMPILER "/usr/bin/gcc-10")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-10")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-10")

project(ivftensor_unit_tests LANGUAGES CXX CUDA)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)

# 设置CUDA标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin /usr/bin/g++-10")


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
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -use_fast_math -lineinfo -w -Xcompiler -w --disable-warnings --expt-relaxed-constexpr")

# CUDA编译选项
set(CMAKE_CUDA_ARCHITECTURES 70)
# 优化编译选项：使用快速数学、减少调试信息、启用并行编译
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math -lineinfo -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE --expt-relaxed-constexpr")
# 设置并行编译线程数（nvcc的--threads选项）
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 0")
# 禁用 ptxas 详细输出信息（移除 -v 选项即可禁用输出）
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

# CUDA包含路径
set(CUDA_INCLUDE_DIRS
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/common
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
set(COMMON_HEADERS "-include ${CMAKE_CURRENT_SOURCE_DIR}/cuda/pch.h")

# 创建共享的l2norm库，避免重复编译
# 注意：使用可分离编译以便其他目标可以链接device符号
add_library(l2norm_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/l2norm/l2norm.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/l2norm/l2norm_squared.cu
)
target_include_directories(l2norm_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/common
)
set_target_properties(l2norm_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
    POSITION_INDEPENDENT_CODE ON  # 用于链接到共享库
)

# 创建共享的utils库，避免重复编译
# 包含的文件：
#   - utils.cu: 通用工具kernel和host函数
add_library(utils_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/utils.cu
)
target_include_directories(utils_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/common
)
set_target_properties(utils_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
    POSITION_INDEPENDENT_CODE ON  # 用于链接到共享库
)

# 创建共享的kmeans库（包含 kmeans 相关函数，供 dataset_wrapper 使用）
add_library(kmeans_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/kmeans/kmeans.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/kmeans/kmeans_minibatch.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/kmeans/kmeans_utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/kmeans/kmeans_reorder.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/kmeans/kmeans_reorder_utils.cu
)
target_include_directories(kmeans_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/common
)
set_target_properties(kmeans_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
    POSITION_INDEPENDENT_CODE ON  # 用于链接到共享库
)
target_link_libraries(kmeans_lib PUBLIC
    utils_lib
    l2norm_lib
    cublas
    curand
)

# 创建共享的indexed_gemm库，避免重复编译
# 包含的文件（仅保留当前使用的版本）：
#   - indexed_gemm_cos.cu: v5 entry-based版本（当前使用）
#   - indexed_gemm_l2.cu: v5 entry-based版本（当前使用）
# 
# 注意：
#   - test_fusion_cos_topk_fine 如需测试弃用版本，需单独编译对应文件
#   - inner_product_utils.cuh 是头文件，不需要添加到库中
add_library(indexed_gemm_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/indexed_gemm/indexed_gemm_cos.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/indexed_gemm/indexed_gemm_l2.cu
)
target_include_directories(indexed_gemm_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/common
)
set_target_properties(indexed_gemm_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
    POSITION_INDEPENDENT_CODE ON  # 用于链接到共享库
)
# 链接依赖库：
target_link_libraries(indexed_gemm_lib 
    PUBLIC
    utils_lib
)

# 创建fusion_dist_topk_warpsort库（包含cosine和L2距离的warpsort实现）
# 被以下使用：
#   - test_fusion_cos_topk_fine (通过fusion_dist_topk_warpsort_lib)
#   - ivf_search_lib (需要fusion_cos_topk_warpsort)
# 
# 包含的文件：
#   - fusion_cos_topk_warpsort.cu: cosine距离的warpsort实现
#   - fusion_l2_topk_warpsort.cu: L2距离的warpsort实现
#   - warpsort_topk.cu: warp-level top-k选择
add_library(fusion_dist_topk_warpsort_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/fusion_dist_topk/fusion_cos_topk_warpsort.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/fusion_dist_topk/fusion_l2_topk_warpsort.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/warpsortfilter/warpsort_topk.cu
)
target_include_directories(fusion_dist_topk_warpsort_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/common
)
set_target_properties(fusion_dist_topk_warpsort_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
    POSITION_INDEPENDENT_CODE ON  # 用于链接到共享库
)
target_link_libraries(fusion_dist_topk_warpsort_lib 
    PUBLIC
    l2norm_lib 
    utils_lib
    cublas
)

# 创建共享的ivf_search库（供ivftensor主项目和测试使用）
# 被以下使用：
#   - ivftensor主项目（ivfscanbatch.c 调用 batch_search_pipeline）
#   - test_ivfflat_search (测试ivf_search.cu，基础测试，数据集常驻显存)
# 
# 包含的文件：
#   - ivf_search.cu: 主实现文件（包含batch_search_pipeline函数）
# 
add_library(ivf_search_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ivf_search/ivf_search.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ivf_search/ivf_search_lookup.cu
)
target_include_directories(ivf_search_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/common
)
set_target_properties(ivf_search_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
    POSITION_INDEPENDENT_CODE ON  # 用于链接到共享库
)
target_link_libraries(ivf_search_lib 
    PUBLIC
    l2norm_lib 
    utils_lib
    indexed_gemm_lib
    fusion_dist_topk_warpsort_lib
    cublas
)

# 创建共享的ivf_search_pipeline库（供测试使用）
# 被以下使用：
#   - test_ivfflat_search_pipeline (测试流水线版本，数据调度，暂时不编译)
#   - test_ivfflat_search_balance (测试平衡版本，聚类平衡数据调度，暂时不编译)
# 
# 包含的文件：
#   - ivf_search_pipeline.cu: 流水线版本实现（包含batch_search_pipeline函数）

add_library(ivf_search_pipeline_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ivf_search/ivf_search_pipeline.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ivf_search/ivf_search_separated.cu  # 包含流式上传函数
)
target_include_directories(ivf_search_pipeline_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/common
)
set_target_properties(ivf_search_pipeline_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
    POSITION_INDEPENDENT_CODE ON  # 用于链接到共享库
)
target_link_libraries(ivf_search_pipeline_lib 
    PUBLIC
    l2norm_lib 
    utils_lib
    indexed_gemm_lib
    fusion_dist_topk_warpsort_lib
    cublas
)

# 添加测试子目录
add_subdirectory(unit_tests/test_print_cuda)
add_subdirectory(unit_tests/test_matrix_multiply)
add_subdirectory(unit_tests/test_fusion_cos_topk)
add_subdirectory(unit_tests/test_warpsort)
add_subdirectory(unit_tests/test_bitonic)
add_subdirectory(unit_tests/test_select_k)
add_subdirectory(unit_tests/test_kmeans)
add_subdirectory(unit_tests/test_ivfflat_search)
add_subdirectory(unit_tests/test_ivfflat_search_lookup)
# 暂时不编译以下测试
# add_subdirectory(unit_tests/test_ivfflat_search_pipeline)
# add_subdirectory(unit_tests/test_ivfflat_search_balance)

# 创建 ivf_search_wrapper 库（包含 C 包装函数）
# 注意：使用 STATIC 但需要 PIC，因为会被链接到共享库
add_library(ivf_search_wrapper_lib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ivf_search/ivf_search_wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/dataset/dataset_wrapper.cpp
)
target_include_directories(ivf_search_wrapper_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/common
)
set_target_properties(ivf_search_wrapper_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
    POSITION_INDEPENDENT_CODE ON  # 添加 PIC 支持，用于链接到共享库
)
target_link_libraries(ivf_search_wrapper_lib 
    PUBLIC
    kmeans_lib  # 添加 kmeans 库以提供 init_centroids_by_sampling 等函数
    ivf_search_lib
    ivf_search_pipeline_lib
    l2norm_lib
    utils_lib
    indexed_gemm_lib
    fusion_dist_topk_warpsort_lib
    cublas
    curand
    cusparse
    cudart
)

# Python 绑定：IVFTensor
# 查找 Python 3.10（优先使用系统 Python）
find_package(Python3 COMPONENTS Interpreter Development QUIET)
if(NOT Python3_FOUND)
    # 如果找不到，尝试查找 Python
    find_package(Python COMPONENTS Interpreter Development QUIET)
    if(Python_FOUND)
        set(Python3_EXECUTABLE ${Python_EXECUTABLE})
        set(Python3_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
        set(Python3_LIBRARIES ${Python_LIBRARIES})
    endif()
endif()

# 查找 pybind11
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    message(STATUS "Found pybind11, building Python bindings")
    if(Python3_EXECUTABLE)
        message(STATUS "Using Python: ${Python3_EXECUTABLE}")
        message(STATUS "Python version: ${Python3_VERSION}")
    endif()
    
    # 创建 Python 模块，指定使用找到的 Python
    pybind11_add_module(PyIVFTensor
        ${CMAKE_CURRENT_SOURCE_DIR}/python/ivf_tensor_bindings.cpp
    )
    
    # 如果找到了特定的 Python，确保使用它
    if(Python3_EXECUTABLE)
        set_target_properties(PyIVFTensor PROPERTIES
            Python_EXECUTABLE ${Python3_EXECUTABLE}
        )
    endif()
    
    target_include_directories(PyIVFTensor PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/common
        ${pybind11_INCLUDE_DIRS}
    )
    
    target_link_libraries(PyIVFTensor PRIVATE
        ivf_search_wrapper_lib
        kmeans_lib  # 添加 kmeans 库
        ivf_search_lib
        ivf_search_pipeline_lib
        l2norm_lib
        utils_lib
        indexed_gemm_lib
        fusion_dist_topk_warpsort_lib
        cudart
        cublas
        curand
        cusparse
    )
    
    # 设置输出目录
    set_target_properties(PyIVFTensor PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/build
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES 70
    )
    
    # 创建共享库（供 ctypes 使用）
    add_library(ivf_search_shared SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ivf_search/ivf_search_wrapper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda/dataset/dataset_wrapper.cpp
    )
    target_include_directories(ivf_search_shared PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/common
    )
    target_link_libraries(ivf_search_shared PUBLIC
        kmeans_lib  # 添加 kmeans 库
        ivf_search_lib
        ivf_search_pipeline_lib
        l2norm_lib
        utils_lib
        indexed_gemm_lib
        fusion_dist_topk_warpsort_lib
        cudart
        cublas
        curand
        cusparse
    )
    set_target_properties(ivf_search_shared PROPERTIES
        OUTPUT_NAME "ivf_search"
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES 70
    )
else()
    message(WARNING "pybind11 not found, skipping Python bindings. Install with: pip install pybind11")
endif()

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
        test_select_k
        test_kmeans
        test_ivfflat_search
        test_ivfflat_search_lookup
        # 暂时不编译以下测试
        # test_ivfflat_search_pipeline
        # test_ivfflat_search_balance
    COMMENT "运行所有单元测试"
)
