# test_warp_sort的CMake配置

# 源文件列表
set(SOURCES
    test_warpsort.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/warpsortfilter/warpsort_topk.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/test_utils.cu
)

# 创建可执行文件
add_executable(test_warpsort ${SOURCES})

# 确保依赖项先构建
add_dependencies(test_warpsort utils_lib)

# 设置包含目录
target_include_directories(test_warpsort PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# 设置CUDA编译选项
set_target_properties(test_warpsort PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 链接CUDA库和utils库
target_link_libraries(test_warpsort
    utils_lib
    ${CUDA_LIBRARIES}
)

# 添加测试
enable_testing()
add_test(NAME warpsort_test COMMAND test_warpsort)


