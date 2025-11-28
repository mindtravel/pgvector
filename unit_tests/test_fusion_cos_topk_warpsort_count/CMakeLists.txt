# test_fusion_cos_topk_warpsort_count的CMake配置

# 源文件列表
set(SOURCES
    test_fusion_cos_topk_warpsort_count.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/test_utils.cu
)

# 创建可执行文件
add_executable(test_fusion_cos_topk_warpsort_count ${SOURCES})

# 链接共享库
# 注意：需要链接包含fusion_cos_topk_warpsort_count.cu的库
# 由于count版本包含额外的功能，需要直接编译或创建单独的库
target_link_libraries(test_fusion_cos_topk_warpsort_count 
    fusion_cos_topk_warpsort_coarse_lib
    l2norm_lib
    cublas
)

# 添加源文件（直接编译fusion_cos_topk_warpsort_count.cu）
target_sources(test_fusion_cos_topk_warpsort_count PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda/fusion_cos_topk/fusion_cos_topk_warpsort_count.cu
)

# 设置包含目录
target_include_directories(test_fusion_cos_topk_warpsort_count PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# 设置CUDA编译选项
# CUDA_SEPARABLE_COMPILATION ON: 启用分离编译，允许增量编译
# CUDA_RESOLVE_DEVICE_SYMBOLS ON: 在库中解析device符号，减少最终链接时间
# CUDA_ARCHITECTURES 70: 只编译SM 7.0，加快编译速度
set_target_properties(test_fusion_cos_topk_warpsort_count PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
)

# 链接CUDA库
target_link_libraries(test_fusion_cos_topk_warpsort_count
    ${CUDA_LIBRARIES}
)

# 添加测试
enable_testing()
add_test(NAME fusion_cos_topk_warpsort_count_test COMMAND test_fusion_cos_topk_warpsort_count)

