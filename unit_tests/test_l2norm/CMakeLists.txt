# test_l2norm的CMake配置

# 源文件列表
set(SOURCES
    test_l2norm.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/l2norm/l2norm.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/test_utils.cu
)

# 创建可执行文件
add_executable(test_l2norm ${SOURCES})

# 设置包含目录
target_include_directories(test_l2norm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# 设置CUDA编译选项
set_target_properties(test_l2norm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接CUDA库
target_link_libraries(test_l2norm
    ${CUDA_LIBRARIES}
)

# 添加测试
enable_testing()
add_test(NAME l2norm_test COMMAND test_l2norm)

