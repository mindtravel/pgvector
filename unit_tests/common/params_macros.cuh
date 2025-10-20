// ============================================================================
// 参数扫描宏 (Parameter Sweep Macros) - Hydra-like style
// ============================================================================

#ifndef PARAMS_MACROS_CUH
#define PARAMS_MACROS_CUH

/**
 * 一维参数扫描宏（支持自动类型推导）
 * 
 * 使用示例：
 * PARAM_1D(k, 8, 16, 32, 64, 128, 256) {
 *     bool result = test_warpsort(10000, 1024, k);
 *     check_pass("Test with k=" + std::to_string(k), result);
 * }
 * 
 * @param VAR 循环变量名
 * @param ... 参数值列表（直接写值，逗号分隔）
 */
#define PARAM_1D(VAR, ...) \
    for (auto VAR : {__VA_ARGS__})

/**
 * 一维参数扫描宏（带索引版本） 感觉没什么用
 * 
 * 使用示例：
 * int idx = 0;
 * PARAM_1D(k, 8, 16, 32, 64, 128, 256) {
 *     COUT_ENDL("[", idx++, "] Testing with k=", k);
 *     bool result = test_warpsort(10000, 1024, k);
 * }
 * 
 * 注意：索引需要在外部手动定义和递增
 */

/**
 * 二维参数扫描宏
 * 
 * 使用示例：
 * PARAM_2D(k, (16, 32, 64), batch, (10, 100, 1000)) {
 *     COUT_ENDL("Testing k=", k, "batch=", batch);
 *     test_warpsort(batch, 1024, k);
 * }
 * 
 * 注意：每组参数值需要用小括号 () 包裹，以防止宏展开时的逗号解析问题
 * 
 * @param VAR1 第一个循环变量名
 * @param VALUES1 第一个参数值列表（用括号包裹）
 * @param VAR2 第二个循环变量名  
 * @param VALUES2 第二个参数值列表（用括号包裹）
 */
#define PARAM_2D(VAR1, VALUES1, VAR2, VALUES2) \
    for (auto VAR1 : {UNWRAP VALUES1}) \
        for (auto VAR2 : {UNWRAP VALUES2})

/* 辅助宏：用于展开括号内的内容 */
#define UNWRAP(...) __VA_ARGS__

/**
 * 三维参数扫描宏
 * 
 * 使用示例：
 * PARAM_3D(k, (16, 32), batch, (10, 100), len, (512, 1024)) {
 *     test_warpsort(batch, len, k);
 * }
 * 
 * @param VAR1 第一个循环变量名
 * @param VALUES1 第一个参数值列表（用括号包裹）
 * @param VAR2 第二个循环变量名
 * @param VALUES2 第二个参数值列表（用括号包裹）
 * @param VAR3 第三个循环变量名
 * @param VALUES3 第三个参数值列表（用括号包裹）
 */
#define PARAM_3D(VAR1, VALUES1, VAR2, VALUES2, VAR3, VALUES3) \
    for (auto VAR1 : {UNWRAP VALUES1}) \
        for (auto VAR2 : {UNWRAP VALUES2}) \
            for (auto VAR3 : {UNWRAP VALUES3})

#endif