# test_cos_distance的CMake配置

# 源文件列表
set(SOURCES
    test_cos_distance.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/experiments/distances.cu
    # 使用共享库，不再重复编译l2norm.cu
    # ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/l2norm/l2norm.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/test_utils.cu
)

# 创建可执行文件
add_executable(test_cos_distance ${SOURCES})

# 链接共享的l2norm库
target_link_libraries(test_cos_distance l2norm_lib)

# 设置包含目录
target_include_directories(test_cos_distance PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# 设置CUDA编译选项
set_target_properties(test_cos_distance PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接CUDA库
target_link_libraries(test_cos_distance
    ${CUDA_LIBRARIES}
)

# 添加测试
enable_testing()
add_test(NAME cos_distance_test COMMAND test_cos_distance)
