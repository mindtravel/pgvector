# test_fusion_cos_topk_fine_v5的CMake配置
# v5版本：Entry-based线程模型

# 源文件列表
set(SOURCES
    test_fusion_cos_topk_fine_v5.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/test_utils.cu
)

# 创建可执行文件
add_executable(test_fusion_cos_topk_fine_v5 ${SOURCES})

# 链接共享库
target_link_libraries(test_fusion_cos_topk_fine_v5 
    fusion_cos_topk_warpsort_lib
    l2norm_lib
)

# 设置包含目录
target_include_directories(test_fusion_cos_topk_fine_v5 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# 设置CUDA编译选项
# CUDA_SEPARABLE_COMPILATION ON: 启用分离编译，允许增量编译
# CUDA_RESOLVE_DEVICE_SYMBOLS ON: 在库中解析device符号，减少最终链接时间
# CUDA_ARCHITECTURES 70: 只编译SM 7.0，加快编译速度
# CUDA_OPTIMIZE_DEVICE_LINK ON: 优化设备链接（如果支持）
set_target_properties(test_fusion_cos_topk_fine_v5 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 70
)

# 链接CUDA库
target_link_libraries(test_fusion_cos_topk_fine_v5
    ${CUDA_LIBRARIES}
)

# 添加测试
enable_testing()
add_test(NAME fusion_cos_topk_fine_v5_test COMMAND test_fusion_cos_topk_fine_v5)

