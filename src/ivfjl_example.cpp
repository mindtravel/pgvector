/*
 * IVFJL索引使用示例
 * 
 * 这个文件展示了如何在C++代码中直接调用IVFJL接口来创建索引
 * 
 * 编译方法：
 * g++ -o ivfjl_example ivfjl_example.cpp -lpq -I/usr/include/postgresql
 * 
 * 使用前确保：
 * 1. PostgreSQL服务器正在运行
 * 2. 已经创建了包含vector列的表
 * 3. pgvector扩展已安装并启用
 */

#include <iostream>
#include <string>
#include <vector>

extern "C" {
    #include "postgres.h"
    #include "ivfjl.h"
}

int main() {
    std::cout << "IVFJL索引创建示例" << std::endl;
    
    // 示例参数
    const char* table_name = "sift_data";      // 表名
    const char* index_name = "sift_ivfjl_idx"; // 索引名
    const char* column_name = "vector";        // 向量列名
    int lists = 100;                          // 聚类数量
    
    std::cout << "正在创建IVFJL索引..." << std::endl;
    std::cout << "表名: " << table_name << std::endl;
    std::cout << "索引名: " << index_name << std::endl;
    std::cout << "列名: " << column_name << std::endl;
    std::cout << "聚类数量: " << lists << std::endl;
    
    // 调用IVFJL接口创建索引
    bool success = CreateIvfjlIndex(table_name, index_name, column_name, lists);
    
    if (success) {
        std::cout << "✓ IVFJL索引创建成功！" << std::endl;
        std::cout << "索引特点：" << std::endl;
        std::cout << "  - 使用Johnson-Lindenstrauss投影降维至64维" << std::endl;
        std::cout << "  - 基于IVFFlat聚类算法" << std::endl;
        std::cout << "  - 支持高维向量的高效近似最近邻搜索" << std::endl;
        std::cout << "  - 显著减少存储空间和计算开销" << std::endl;
    } else {
        std::cout << "✗ IVFJL索引创建失败！" << std::endl;
        std::cout << "请检查：" << std::endl;
        std::cout << "  - 表和列是否存在" << std::endl;
        std::cout << "  - PostgreSQL连接是否正常" << std::endl;
        std::cout << "  - 是否有足够的权限" << std::endl;
        std::cout << "  - pgvector扩展是否已安装" << std::endl;
    }
    
    return success ? 0 : 1;
}

/*
 * 使用说明：
 * 
 * 1. IVFJL索引建立流程：
 *    - 初始化构建状态
 *    - 采样原始高维向量数据
 *    - 生成Johnson-Lindenstrauss投影矩阵
 *    - 将样本投影到64维空间
 *    - 在降维后的空间中进行k-means聚类
 *    - 创建元数据页并保存投影矩阵
 *    - 创建列表页和条目页
 *    - 完成索引构建
 * 
 * 2. IVFJL优势：
 *    - 降维：将高维向量降至64维，大幅减少存储和计算开销
 *    - 保距：JL变换保持向量间的相对距离
 *    - 兼容：复用IVFFlat的成熟聚类和搜索算法
 *    - 高效：在保证搜索质量的同时提升性能
 * 
 * 3. 应用场景：
 *    - 高维向量数据库
 *    - 图像检索系统
 *    - 文本相似度搜索
 *    - 推荐系统
 *    - 机器学习特征匹配
 */ 