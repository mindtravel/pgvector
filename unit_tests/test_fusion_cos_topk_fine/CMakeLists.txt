# test_fusion_cos_topk_v2的CMake配置

# 源文件列表
set(SOURCES
    test_fusion_cos_topk_fine.cu
    # 使用共享库，不再重复编译
    # ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/l2norm/l2norm.cu - 在 l2norm_lib 中
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/fusion_cos_topk/fusion_cos_topk_warpsort_fine_v2.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/fusion_cos_topk/fusion_cos_topk_warpsort_fine_v3.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/fusion_cos_topk/fusion_cos_topk_warpsort_fine_v3_32.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/fusion_cos_topk/fusion_cos_topk_warpsort_fine_v4.cu
    # 弃用版本的 indexed_gemm 需要单独编译（不在 indexed_gemm_lib 中）
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/indexed_gemm/indexed_gemm_v2.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/indexed_gemm/indexed_gemm_v3.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/indexed_gemm/indexed_gemm_v4.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/warpsortfilter/warpsort_topk.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/test_utils.cu
)

# 创建可执行文件
add_executable(test_fusion_cos_topk_fine ${SOURCES})

# 链接共享库
target_link_libraries(test_fusion_cos_topk_fine 
    l2norm_lib
    utils_lib  # indexed_gemm_v2, v3 需要 inner_product_utils
)

# 设置包含目录
target_include_directories(test_fusion_cos_topk_fine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# 设置CUDA编译选项
set_target_properties(test_fusion_cos_topk_fine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接CUDA库
target_link_libraries(test_fusion_cos_topk_fine
    ${CUDA_LIBRARIES}
)

# 添加测试
enable_testing()
add_test(NAME fusion_cos_topk_v2_test COMMAND test_fusion_cos_topk_fine)

