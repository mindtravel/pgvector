# test_matrix_multiply的CMake配置

# 源文件列表
set(SOURCES
    test_matrix_multiply.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/experiments/matrix-multiply.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/test_utils.cu
)

# 创建可执行文件
add_executable(test_matrix_multiply ${SOURCES})

# 设置包含目录
target_include_directories(test_matrix_multiply PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# 设置CUDA编译选项
set_target_properties(test_matrix_multiply PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接CUDA库
target_link_libraries(test_matrix_multiply
    ${CUDA_LIBRARIES}
)

# 添加测试
enable_testing()
add_test(NAME matrix_multiply_test COMMAND test_matrix_multiply)
