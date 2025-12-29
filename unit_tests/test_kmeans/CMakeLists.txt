# test_kmeans的CMake配置

# 源文件列表
set(SOURCES
    test_kmeans.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/kmeans/kmeans.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/kmeans/kmeans_utils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/test_utils.cu
)

# 创建可执行文件
add_executable(test_kmeans ${SOURCES})

# 确保 utils_lib 和 l2norm_lib 先于 test_kmeans 构建
add_dependencies(test_kmeans utils_lib l2norm_lib)

# 设置包含目录
target_include_directories(test_kmeans PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# 设置C++标准
set_target_properties(test_kmeans PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# 设置CUDA编译选项
set_target_properties(test_kmeans PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 添加编译选项以修复 std::function 兼容性问题
target_compile_options(test_kmeans PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler -std=c++17
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -std=c++17
    >
)

# 链接CUDA库和工具库
target_link_libraries(test_kmeans
    utils_lib
    l2norm_lib
    ${CUDA_LIBRARIES}
)

# 添加测试
enable_testing()
add_test(NAME kmeans_test COMMAND test_kmeans)

