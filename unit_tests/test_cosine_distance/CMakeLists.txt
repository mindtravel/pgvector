# test_cosine_distance的CMake配置

# 源文件列表
set(SOURCES
    test_cosine_distance.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/distances.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/fusion_cos_topk.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/test_utils.cu
)

# 创建可执行文件
add_executable(test_cosine_distance ${SOURCES})

# 设置包含目录
target_include_directories(test_cosine_distance PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# 设置CUDA编译选项
set_target_properties(test_cosine_distance PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接CUDA库
target_link_libraries(test_cosine_distance
    ${CUDA_LIBRARIES}
)

# 添加测试
enable_testing()
add_test(NAME cosine_distance_test COMMAND test_cosine_distance)
