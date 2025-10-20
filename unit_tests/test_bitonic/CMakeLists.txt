# test_bitonic的CMake配置

# 源文件列表
set(SOURCES
    test_bitonic.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/test_utils.cu
)

# 创建可执行文件
add_executable(test_bitonic ${SOURCES})

# 设置包含目录
target_include_directories(test_bitonic PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# 设置CUDA编译选项
set_target_properties(test_bitonic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接CUDA库
target_link_libraries(test_bitonic
    ${CUDA_LIBRARIES}
)

# 添加测试
enable_testing()
add_test(NAME bitonic_test COMMAND test_bitonic)

