# test_fusion_cos_topk的CMake配置

# 源文件列表 - 包含所有算法版本
set(SOURCES
    test_fusion_cos_topk.cu
    # 使用共享库，不再重复编译
    # ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/l2norm/l2norm.cu
    # ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/fusion_cos_topk/fusion_cos_topk_warpsort.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/fusion_cos_topk/fusion_cos_topk_heap.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/fusion_cos_topk/fusion_cos_topk_heap_sharedmem.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/test_utils.cu
)

# 创建可执行文件
add_executable(test_fusion_cos_topk ${SOURCES})

# 链接共享库
target_link_libraries(test_fusion_cos_topk 
    fusion_cos_topk_warpsort_lib
    l2norm_lib
)

# 设置包含目录
target_include_directories(test_fusion_cos_topk PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# 设置CUDA编译选项
set_target_properties(test_fusion_cos_topk PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接CUDA库
target_link_libraries(test_fusion_cos_topk
    ${CUDA_LIBRARIES}
)

# 添加测试
enable_testing()
add_test(NAME fusion_cos_topk_test COMMAND test_fusion_cos_topk)
