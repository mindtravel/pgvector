# test_vector_normalizer的CMake配置

# 源文件列表
set(SOURCES
    test_vector_normalizer.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/normalize.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/test_utils.cu
)

# 创建可执行文件
add_executable(test_vector_normalizer ${SOURCES})

# 设置包含目录
target_include_directories(test_vector_normalizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# 设置CUDA编译选项
set_target_properties(test_vector_normalizer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 添加测试
enable_testing()
add_test(NAME vector_normalizer_test COMMAND test_vector_normalizer)