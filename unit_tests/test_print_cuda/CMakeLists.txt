# test_print_cuda的CMake配置

# 源文件列表
set(SOURCES
    test_print_cuda.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/test_utils.cu
)

# 创建可执行文件
add_executable(test_print_cuda ${SOURCES})

# 设置包含目录
target_include_directories(test_print_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# 设置CUDA编译选项
set_target_properties(test_print_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 添加测试
enable_testing()
add_test(NAME print_cuda_test COMMAND test_print_cuda)
