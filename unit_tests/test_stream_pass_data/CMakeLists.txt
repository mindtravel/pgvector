# test_stream_pass_data的CMake配置

# 源文件列表
set(SOURCES
    test_stream_pass_data.cu
    stream_pass_data.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/test_utils.cu
)

# 创建可执行文件
add_executable(test_stream_pass_data ${SOURCES})

# 设置包含目录
target_include_directories(test_stream_pass_data PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# 设置CUDA编译选项
set_target_properties(test_stream_pass_data PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接CUDA库
target_link_libraries(test_stream_pass_data
    ${CUDA_LIBRARIES}
)

# 添加测试
enable_testing()
add_test(NAME stream_pass_data_test COMMAND test_stream_pass_data)
