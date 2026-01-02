#ifndef CPU_UTILS_H
#define CPU_UTILS_H

/**
 * CPU工具函数库 - 统一入口
 * 
 * 该文件提供向后兼容性，包含所有拆分的模块
 * 新代码建议直接包含具体的模块头文件：
 *   - cpu_distance.h: 距离计算函数
 *   - cpu_array_utils.h: 数组工具函数
 *   - cpu_kmeans.h: K-means参考实现
 *   - cpu_search.h: 搜索参考实现
 */

#include "cpu_distance.h"
#include "cpu_kmeans.h"
#include "cpu_search.h"
#include "cpu_sort.h"

#endif // CPU_UTILS_H

