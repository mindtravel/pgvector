#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <libpq-fe.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <getopt.h>
#include <thread>
#include <mutex>
#include <random>
#include <unordered_map>
#include <omp.h>
#include <iomanip>

double pre_process_time;

// 数据库配置
const std::string DB_CONN_STR = 
    "host=localhost port=5432 dbname=mydatabase user=postgres password=1";

// 数据结构定义
struct QueryResult {
    std::vector<int> ids;
    double latency;
};

struct BenchmarkResult {
    int nprob;
    double recall;
    double throughput;
    BenchmarkResult(int n, double r, double t) 
        : nprob(n), recall(r), throughput(t) {}
};

// 计算向量范数
float vector_norm(const std::vector<float>& vec) {
    float sum = 0.0f;
    for (float val : vec) {
        sum += val * val;
    }
    return std::sqrt(sum);
}

// 向量标准化
std::vector<float> normalize_vector(const std::vector<float>& vec) {
    float norm = vector_norm(vec);
    if (norm < 1e-8) return vec; // 避免除零
    
    std::vector<float> normalized(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        normalized[i] = vec[i] / norm;
    }
    return normalized;
}

// 对整个数据集进行标准化
std::vector<std::vector<float>> normalize_dataset(const std::vector<std::vector<float>>& dataset) {
    std::vector<std::vector<float>> normalized(dataset.size());
    for (size_t i = 0; i < dataset.size(); ++i) {
        normalized[i] = normalize_vector(dataset[i]);
    }
    return normalized;
}

// 计算两个向量之间的欧氏距离平方
float euclidean_distance_squared(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// 并行矩阵乘法
std::vector<std::vector<float>> parallel_matrix_multiply(
    const std::vector<std::vector<float>>& A,
    const std::vector<std::vector<float>>& B,
    int num_threads = std::thread::hardware_concurrency()) {
    
    int m = A.size();
    int n = B[0].size();
    int k = B.size();
    
    std::vector<std::vector<float>> C(m, std::vector<float>(n, 0.0f));
    
    // 每个线程处理的行数
    int rows_per_thread = (m + num_threads - 1) / num_threads;
    
    auto worker = [&](int thread_id) {
        int start_row = thread_id * rows_per_thread;
        int end_row = std::min(start_row + rows_per_thread, m);
        
        for (int i = start_row; i < end_row; ++i) {
            for (int p = 0; p < k; ++p) {
                float a_ip = A[i][p];
                for (int j = 0; j < n; ++j) {
                    C[i][j] += a_ip * B[p][j];
                }
            }
        }
    };
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    return C;
}

std::vector<std::vector<float>> batched_parallel_matrix_multiply(
    const std::vector<std::vector<float>>& A,
    const std::vector<std::vector<float>>& B,
    int batch_size = 8,
    int num_threads = std::thread::hardware_concurrency()) {
    
    int m = A.size();
    int n = B[0].size();
    int k = B.size();
    
    std::vector<std::vector<float>> C(m, std::vector<float>(n, 0.0f));
    
    // 每个线程处理的批次数
    int batches_per_thread = (m + batch_size * num_threads - 1) / (batch_size * num_threads);
    
    auto worker = [&](int thread_id) {
        int start_batch = thread_id * batches_per_thread;
        int end_batch = std::min(start_batch + batches_per_thread, (m + batch_size - 1) / batch_size);
        
        for (int b = start_batch; b < end_batch; ++b) {
            int start_i = b * batch_size;
            int end_i = std::min(start_i + batch_size, m);
            
            for (int i = start_i; i < end_i; ++i) {
                for (int j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (int p = 0; p < k; ++p) {
                        sum += A[i][p] * B[p][j];
                    }
                    C[i][j] = sum;
                }
            }
        }
    };
    
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker, t);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    return C;
}

std::vector<int> rerank_results(
    const std::vector<std::vector<float>>& original_vectors,
    const std::vector<float>& query,
    const std::vector<int>& candidate_ids,
    int k) {
    
    if (candidate_ids.empty() || k <= 0) return {};
    
    // 计算原始空间中的真实距离
    std::vector<std::pair<float, int>> distances;
    for (int id : candidate_ids) {
        if (id >= 0 && id < original_vectors.size()) {
            float dist = euclidean_distance_squared(original_vectors[id], query);
            distances.emplace_back(dist, id);
        }
    }
    
    // 按真实距离排序
    std::sort(distances.begin(), distances.end(), 
              [](const auto& a, const auto& b) { return a.first < b.first; });
    
    // 提取前k个结果
    std::vector<int> reranked_ids;
    for (int i = 0; i < std::min(k, static_cast<int>(distances.size())); ++i) {
        reranked_ids.push_back(distances[i].second);
    }
    
    return reranked_ids;
}

// 生成JL随机投影矩阵
std::vector<std::vector<float>> generate_jl_projection_matrix(int original_dim, int target_dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution bernoulli(1.0f / 3.0f); // 1/3概率非零
    
    std::vector<std::vector<float>> projection(target_dim, std::vector<float>(original_dim, 0.0f));
    
    for (int i = 0; i < target_dim; ++i) {
        for (int j = 0; j < original_dim; ++j) {
            if (bernoulli(gen)) { // 非零概率1/3
                projection[i][j] = (gen() % 2 == 0) ? 1.0f : -1.0f; // 等概率±1
            }
        }
    }
    
    return projection;
}

// 执行JL降维
std::vector<std::vector<float>> perform_jl_reduction(
    const std::vector<std::vector<float>>& data,
    int target_dim) {
    
    int original_dim = data[0].size();
    int num_vectors = data.size();
    
    // 生成随机投影矩阵
    auto projection = generate_jl_projection_matrix(original_dim, target_dim);
    
    // 转置数据以便于矩阵乘法
    std::vector<std::vector<float>> data_t(original_dim, std::vector<float>(num_vectors));
    for (int i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < original_dim; ++j) {
            data_t[j][i] = data[i][j];
        }
    }
    
    // 使用优化的矩阵乘法执行降维
    auto reduced_t = parallel_matrix_multiply(projection, data_t);
    
    // 转置结果回原始格式
    std::vector<std::vector<float>> reduced(num_vectors, std::vector<float>(target_dim));
    for (int i = 0; i < target_dim; ++i) {
        for (int j = 0; j < num_vectors; ++j) {
            reduced[j][i] = reduced_t[i][j];
        }
    }
    
    return reduced;
}

// 生成SJLT随机投影矩阵（稀疏版本）
using SparseMatrix = std::pair<std::vector<std::vector<int>>, std::vector<std::vector<float>>>;

SparseMatrix generate_sjlt_projection_matrix(int original_dim, int target_dim, int sparsity = 3) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // 非零元素的概率分布
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    // 非零值的分布（三值分布：{-√(s/k), 0, +√(s/k)}）
    std::uniform_int_distribution<int> val_dist(0, 2);
    
    // 初始化稀疏矩阵（使用CSR格式表示）
    std::vector<std::vector<int>> col_indices(target_dim);    // 每一行的列索引
    std::vector<std::vector<float>> values(target_dim);       // 每一行的非零值
    
    float scale = std::sqrt(sparsity / static_cast<float>(target_dim));
    
    for (int i = 0; i < target_dim; ++i) {
        for (int j = 0; j < original_dim; ++j) {
            // 以s/k的概率选择非零元素
            if (prob_dist(gen) < static_cast<float>(sparsity) / original_dim) {
                // 三值分布：1/sqrt(s)概率为+1，1/sqrt(s)概率为-1，(s-2)/s概率为0
                float val = 0.0f;
                int choice = val_dist(gen);
                if (choice == 0) val = -scale;
                else if (choice == 1) val = scale;
                
                if (val != 0.0f) {
                    col_indices[i].push_back(j);
                    values[i].push_back(val);
                }
            }
        }
    }
    
    return {col_indices, values};
}

// 使用CSR格式的稀疏矩阵执行SJLT降维
std::vector<std::vector<float>> perform_sjlt_reduction(
    const std::vector<std::vector<float>>& data,
    int target_dim,
    int sparsity = 3) {
    
    int original_dim = data[0].size();
    int num_vectors = data.size();
    
    // 生成稀疏投影矩阵（CSR格式）
    auto [col_indices, values] = generate_sjlt_projection_matrix(original_dim, target_dim, sparsity);
    
    // 执行降维
    std::vector<std::vector<float>> reduced(num_vectors, std::vector<float>(target_dim, 0.0f));
    
    // 对每个向量进行投影
    for (int v = 0; v < num_vectors; ++v) {
        const auto& vec = data[v];
        
        // 对每个目标维度
        for (int i = 0; i < target_dim; ++i) {
            // 稀疏矩阵乘法
            float dot_product = 0.0f;
            const auto& row_cols = col_indices[i];
            const auto& row_vals = values[i];
            
            for (size_t k = 0; k < row_cols.size(); ++k) {
                dot_product += row_vals[k] * vec[row_cols[k]];
            }
            
            reduced[v][i] = dot_product;
        }
    }
    
    return reduced;
}

// 数据加载函数 - 增加对ivecs格式的支持
std::vector<std::vector<float>> load_data(const std::string& path, int dim, bool is_ivecs = false) {
    std::ifstream file(path, std::ios::binary);
    std::vector<std::vector<float>> data;
    
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
        return data;
    }

    // 计算向量数量
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t element_size = is_ivecs ? sizeof(int32_t) : sizeof(float);
    size_t vector_size = (dim * element_size) + sizeof(int32_t);
    size_t num_vectors = file_size / vector_size;
    
    std::cout << "Loading " << (is_ivecs ? "ivecs" : "fvecs") 
              << " file: " << path << std::endl;
    std::cout << "File size: " << file_size << " bytes" << std::endl;
    std::cout << "Vector size: " << vector_size << " bytes" << std::endl;
    std::cout << "Number of vectors expected: " << num_vectors << std::endl;
    
    for (size_t i = 0; i < num_vectors; ++i) {
        int32_t actual_dim;
        file.read(reinterpret_cast<char*>(&actual_dim), sizeof(int32_t));
        
        if (actual_dim != dim) {
            std::cerr << "Dimension mismatch at vector " << i 
                      << ": expected " << dim << ", got " << actual_dim << std::endl;
            break;
        }
        
        std::vector<float> vec(dim);
        if (is_ivecs) {
            // 读取整型ID并转换为float
            std::vector<int32_t> int_vec(dim);
            file.read(reinterpret_cast<char*>(int_vec.data()), dim * sizeof(int32_t));
            for (int j = 0; j < dim; ++j) {
                vec[j] = static_cast<float>(int_vec[j]);
            }
        } else {
            // 正常读取float
            file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        }
        
        if (file.gcount() != static_cast<std::streamsize>(dim * element_size)) {
            std::cerr << "Incomplete read at vector " << i << std::endl;
            break;
        }
        
        data.push_back(vec);
    }
    
    std::cout << "Successfully loaded " << data.size() << " vectors" << std::endl;
    return data;
}

// 专门加载groundtruth（返回整数向量）
std::vector<std::vector<int>> load_groundtruth(const std::string& path, int dim) {
    std::ifstream file(path, std::ios::binary);
    std::vector<std::vector<int>> data;
    
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
        return data;
    }

    // 计算向量数量
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t element_size = sizeof(int32_t);
    size_t vector_size = (dim * element_size) + sizeof(int32_t);
    size_t num_vectors = file_size / vector_size;
    
    std::cout << "Loading groundtruth file: " << path << std::endl;
    std::cout << "File size: " << file_size << " bytes" << std::endl;
    std::cout << "Vector size: " << vector_size << " bytes" << std::endl;
    std::cout << "Number of queries: " << num_vectors << std::endl;
    
    for (size_t i = 0; i < num_vectors; ++i) {
        int32_t actual_dim;
        file.read(reinterpret_cast<char*>(&actual_dim), sizeof(int32_t));
        
        if (actual_dim != dim) {
            std::cerr << "Dimension mismatch at query " << i 
                      << ": expected " << dim << ", got " << actual_dim << std::endl;
            break;
        }
        
        std::vector<int> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int32_t));
        
        if (file.gcount() != static_cast<std::streamsize>(dim * element_size)) {
            std::cerr << "Incomplete read at query " << i << std::endl;
            break;
        }
        
        data.push_back(vec);
    }
    
    std::cout << "Successfully loaded groundtruth for " << data.size() << " queries" << std::endl;
    return data;
}

// 数据库连接
PGconn* connect_to_db() {
    PGconn* conn = PQconnectdb(DB_CONN_STR.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(conn) << std::endl;
        PQfinish(conn);
        return nullptr;
    }
    return conn;
}

// 执行SQL命令
void execute_sql(PGconn* conn, const std::string& sql) {
    PGresult* res = PQexec(conn, sql.c_str());
    if (PQresultStatus(res) != PGRES_COMMAND_OK) {
        std::cerr << "SQL Error: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        exit(1);
    }
    PQclear(res);
}

// 创建表并加载数据
void create_table_and_load_data(PGconn* conn, 
                              const std::vector<std::vector<float>>& data,
                              int reduced_dim) {
    execute_sql(conn, "CREATE EXTENSION IF NOT EXISTS vector;");
    execute_sql(conn, "DROP TABLE IF EXISTS sift_data;");
    execute_sql(conn, 
        "CREATE TABLE sift_data (id serial, vector vector(" + 
        std::to_string(reduced_dim) + "));");

    if (data.empty()) {
        std::cout << "Warning: No data to insert!" << std::endl;
        return;
    }

    // 使用COPY命令高效插入
    PGresult* res = PQexec(conn, "COPY sift_data (vector) FROM STDIN WITH (FORMAT CSV)");
    if (PQresultStatus(res) != PGRES_COPY_IN) {
        std::cerr << "COPY failed: " << PQerrorMessage(conn) << std::endl;
        PQclear(res);
        return;
    }
    PQclear(res);

    const size_t batch_size = 5000;
    size_t inserted = 0;
    
    for (size_t i = 0; i < data.size(); ++i) {
        std::stringstream row;
        row << "\"["; 
        for (size_t j = 0; j < data[i].size(); ++j) {
            row << (j > 0 ? "," : "") << std::fixed << std::setprecision(6) << data[i][j];
        }
        row << "]\"\n";
        
        if (PQputCopyData(conn, row.str().c_str(), row.str().size()) != 1) {
            std::cerr << "PQputCopyData failed: " << PQerrorMessage(conn) << std::endl;
            break;
        }
        
        if ((i + 1) % batch_size == 0) {
            std::cout << "Inserted " << (i + 1) << " records..." << std::endl;
        }
    }

    if (PQputCopyEnd(conn, NULL) != 1) {
        std::cerr << "PQputCopyEnd failed: " << PQerrorMessage(conn) << std::endl;
    } else {
        std::cout << "Successfully inserted " << data.size() << " records." << std::endl;
    }
}

// 创建索引
void create_ivfflat_index(PGconn* conn, int n_lists) {
    execute_sql(conn, "SET maintenance_work_mem = '1GB';");
    execute_sql(conn,
        "CREATE INDEX ON sift_data USING ivfflat (vector vector_l2_ops) "
        "WITH (lists = " + std::to_string(n_lists) + ");");
}

// 验证向量与groundtruth的距离关系
void validate_groundtruth(
    const std::vector<std::vector<float>>& database,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<int>>& groundtruth,  // 修改为int类型
    int sample_size = 10) {
    
    std::cout << "\nValidating groundtruth..." << std::endl;
    
    for (int i = 0; i < sample_size; ++i) {
        const auto& query = queries[i];
        std::cout << "Query " << i << ":\n";
        
        // 计算与groundtruth中前5个ID的距离
        for (int j = 0; j < std::min(5, static_cast<int>(groundtruth[i].size())); ++j) {
            int gt_id = groundtruth[i][j];
            if (gt_id >= 0 && gt_id < database.size()) {
                float dist = std::sqrt(euclidean_distance_squared(query, database[gt_id]));
                std::cout << "  Groundtruth #" << j << " (ID=" << gt_id << "): distance=" << dist << "\n";
            }
        }
        
        // 计算与数据库中随机选择的5个向量的距离
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, database.size() - 1);
        
        std::cout << "  Random vectors:\n";
        for (int j = 0; j < 5; ++j) {
            int random_id = dis(gen);
            float dist = std::sqrt(euclidean_distance_squared(query, database[random_id]));
            std::cout << "    Random ID=" << random_id << ": distance=" << dist << "\n";
        }
    }
}

// 获取向量所属的聚类中心ID
std::vector<int> get_vector_cluster_assignments(
    const std::vector<std::vector<float>>& vectors,
    const std::vector<std::vector<float>>& centroids) {
    
    std::vector<int> assignments(vectors.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < vectors.size(); ++i) {
        float min_dist = std::numeric_limits<float>::max();
        int closest_centroid = -1;
        
        for (size_t c = 0; c < centroids.size(); ++c) {
            float dist = euclidean_distance_squared(vectors[i], centroids[c]);
            if (dist < min_dist) {
                min_dist = dist;
                closest_centroid = c;
            }
        }
        
        assignments[i] = closest_centroid;
    }
    
    return assignments;
}

// 从数据库中获取聚类中心
std::vector<std::vector<float>> get_ivfflat_centroids(PGconn* conn, int n_lists, int dim) {
    std::vector<std::vector<float>> centroids(n_lists, std::vector<float>(dim));
    
    // 使用系统表查询聚类中心
    for (int i = 0; i < n_lists; ++i) {
        std::string sql = "SELECT * FROM pg_catalog.ivfflat_getlists('sift_data') WHERE listno = " + std::to_string(i);
        PGresult* res = PQexec(conn, sql.c_str());
        
        if (PQresultStatus(res) != PGRES_TUPLES_OK || PQntuples(res) == 0) {
            std::cerr << "Failed to get centroid for list " << i << std::endl;
            PQclear(res);
            continue;
        }
        
        // 解析向量数据
        char* vec_str = PQgetvalue(res, 0, 1); // 第二列是向量数据
        std::stringstream ss(vec_str);
        std::string token;
        int j = 0;
        
        while (std::getline(ss, token, ',') && j < dim) {
            // 移除括号和空格
            token.erase(std::remove(token.begin(), token.end(), '['), token.end());
            token.erase(std::remove(token.begin(), token.end(), ']'), token.end());
            token.erase(std::remove(token.begin(), token.end(), ' '), token.end());
            
            if (!token.empty()) {
                centroids[i][j] = std::stof(token);
                j++;
            }
        }
        
        PQclear(res);
    }
    
    return centroids;
}

// 分析查询的聚类中心匹配情况
std::vector<std::pair<int, int>> analyze_cluster_matches(
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<int>>& groundtruth,  // 修改为int类型
    const std::vector<std::vector<int>>& topk_results,  // 修改为int类型
    const std::vector<std::vector<float>>& centroids,
    const std::vector<std::vector<float>>& database_vectors) {
    
    // 计算每个查询的groundtruth所属聚类中心
    std::vector<int> gt_clusters(queries.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < queries.size(); ++i) {
        // 只考虑groundtruth中的第一个（最近邻）
        if (!groundtruth[i].empty()) {
            int gt_id = groundtruth[i][0];
            if (gt_id >= 0 && gt_id < database_vectors.size()) {
                float min_dist = std::numeric_limits<float>::max();
                int closest_centroid = -1;
                
                for (size_t c = 0; c < centroids.size(); ++c) {
                    float dist = euclidean_distance_squared(database_vectors[gt_id], centroids[c]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest_centroid = c;
                    }
                }
                
                gt_clusters[i] = closest_centroid;
            } else {
                gt_clusters[i] = -1; // 无效ID
            }
        } else {
            gt_clusters[i] = -1; // 空的groundtruth
        }
    }
    
    // 计算每个查询的TopK结果所属聚类中心
    std::vector<int> topk_clusters(queries.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < queries.size(); ++i) {
        // 统计TopK结果中最频繁出现的聚类中心
        std::vector<int> result_ids = topk_results[i];
        std::unordered_map<int, int> cluster_counts;
        
        for (int id : result_ids) {
            if (id >= 0 && id < database_vectors.size()) {
                float min_dist = std::numeric_limits<float>::max();
                int closest_centroid = -1;
                
                for (size_t c = 0; c < centroids.size(); ++c) {
                    float dist = euclidean_distance_squared(database_vectors[id], centroids[c]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest_centroid = c;
                    }
                }
                
                cluster_counts[closest_centroid]++;
            }
        }
        
        // 找出最频繁的聚类中心
        int most_common_cluster = -1;
        int max_count = 0;
        
        for (const auto& pair : cluster_counts) {
            if (pair.second > max_count) {
                max_count = pair.second;
                most_common_cluster = pair.first;
            }
        }
        
        topk_clusters[i] = most_common_cluster;
    }
    
    // 比较并返回匹配结果
    std::vector<std::pair<int, int>> matches(queries.size());
    for (size_t i = 0; i < queries.size(); ++i) {
        matches[i] = {gt_clusters[i], topk_clusters[i]};
    }
    
    return matches;
}

// KNN搜索测试 - 修正ID匹配逻辑
std::pair<double, double> test_knn_search(
    PGconn* conn,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<int>>& groundtruth,  // 修改为int类型
    const std::vector<std::vector<float>>& original_vectors,
    const std::vector<std::vector<float>>& original_query_vectors,
    const std::vector<std::vector<float>>& centroids,  // 新增：聚类中心
    int k,
    int nprob,
    int thread_count = std::thread::hardware_concurrency(),
    bool analyze_clusters = false) {  // 新增：是否分析聚类
    
    if (thread_count <= 0) thread_count = 1;
    const size_t query_count = queries.size();
    const size_t queries_per_thread = (query_count + thread_count - 1) / thread_count;
    
    std::vector<std::thread> threads;
    std::vector<double> thread_recalls(thread_count, 0.0);
    std::vector<double> thread_times(thread_count, 0.0);
    
    // 跟踪有召回的查询数量
    std::vector<int> thread_hits(thread_count, 0);
    std::vector<int> thread_tests(thread_count, 0);
    
    double total_recall = 0.0;
    
    // 存储每个查询的TopK结果（用于聚类分析）
    std::vector<std::vector<int>> all_topk_results(query_count);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    // 创建线程执行查询
    for (int t = 0; t < thread_count; ++t) {
        threads.emplace_back([&, t]() {
            size_t start_idx = t * queries_per_thread;
            size_t end_idx = std::min(start_idx + queries_per_thread, query_count);
            
            // 每个线程创建自己的数据库连接
            PGconn* thread_conn = connect_to_db();
            if (!thread_conn) {
                std::cerr << "Thread " << t << " failed to connect to database!" << std::endl;
                return;
            }
            
            // 设置每个线程的探测参数
            execute_sql(thread_conn, "SET ivfflat.probes = " + std::to_string(nprob) + ";");
            
            double local_recall = 0.0;
            double local_time = 0.0;
            int local_hits = 0;
            int local_tests = 0;
            
            for (size_t i = start_idx; i < end_idx; ++i) {
                std::string vec_str = "'[";
                for (size_t j = 0; j < queries[i].size(); ++j) {
                    vec_str += (j > 0 ? "," : "") + std::to_string(queries[i][j]);
                }
                vec_str += "]'::vector(" + std::to_string(queries[i].size()) + ")";
                
                std::string sql = "SELECT id, (vector <-> " + vec_str + ") as distance FROM sift_data "
                                  "ORDER BY vector <-> " + vec_str + " LIMIT " + std::to_string(5*k) + ";";
                
                auto query_start = std::chrono::high_resolution_clock::now();
                PGresult* res = PQexec(thread_conn, sql.c_str());
                auto query_end = std::chrono::high_resolution_clock::now();
                
                if (PQresultStatus(res) != PGRES_TUPLES_OK) {
                    std::cerr << "Query " << i << " failed: " << PQerrorMessage(thread_conn) << std::endl;
                    PQclear(res);
                    continue;
                }
                
                std::vector<int> candidate_ids;
                for (int r = 0; r < PQntuples(res); ++r) {
                    int db_id = atoi(PQgetvalue(res, r, 0));
                    candidate_ids.push_back(db_id);
                }
                PQclear(res);
                
                // 重排序（关键步骤）
                std::vector<int> final_results;
                if (!original_vectors.empty() && !original_query_vectors.empty()) {
                    // 使用原始高维向量进行重排序
                    final_results = rerank_results(
                        original_vectors,        // 原始高维向量
                        original_query_vectors[i],     // 当前查询的原始高维向量
                        candidate_ids,           // 低维检索的候选结果
                        k                        // 需要返回的结果数
                    );
                } else {
                    // 如果没有提供原始向量，则直接使用低维结果
                    final_results = candidate_ids.size() > k ? 
                                   std::vector<int>(candidate_ids.begin(), candidate_ids.begin() + k) : 
                                   candidate_ids;
                }
                
                // 保存TopK结果用于聚类分析
                if (analyze_clusters) {
                    all_topk_results[i] = final_results;
                }
                
                // 计算召回率
                int correct = 0;
                for (int id : final_results) {
                    // 检查是否在真实最近邻列表中
                    if (std::find(groundtruth[i].begin(), groundtruth[i].end(), id) != groundtruth[i].end()) {
                        correct++;
                    }
                }
                
                // 更新统计信息
                local_recall += static_cast<double>(correct) / k;
                local_time += std::chrono::duration<double>(query_end - query_start).count();
                local_tests++;
                if (correct > 0) local_hits++;
            }
            
            thread_recalls[t] = local_recall;
            thread_times[t] = local_time;
            thread_hits[t] = local_hits;
            thread_tests[t] = local_tests;
            
            // 关闭线程的数据库连接
            PQfinish(thread_conn);
        });
    }
    
    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count() + pre_process_time;
    
    // 汇总结果
    for (double r : thread_recalls) {
        total_recall += r;
    }
    
    double recall = total_recall / query_count;
    double throughput = query_count / total_time;
    
    // 输出基本统计信息
        // 输出基本统计信息
    int total_hits = std::accumulate(thread_hits.begin(), thread_hits.end(), 0);
    int total_tests = std::accumulate(thread_tests.begin(), thread_tests.end(), 0);
    
    std::cout << "nprob=" << nprob << ", recall@" << k << "=" << recall 
              << ", throughput=" << throughput << " qps" << std::endl;
    std::cout << "Queries with non-zero recall: " << total_hits << " / " << total_tests 
              << " (" << (total_tests > 0 ? (static_cast<double>(total_hits) / total_tests * 100.0) : 0.0) 
              << "%)" << std::endl;
    
    // 执行聚类分析
    std::vector<std::pair<int, int>> cluster_matches;
    if (analyze_clusters && !centroids.empty()) {
        cluster_matches = analyze_cluster_matches(
            queries, groundtruth, all_topk_results, centroids, original_vectors);
        
        // 计算聚类匹配率
        int match_count = 0;
        int valid_count = 0;
        
        for (const auto& match : cluster_matches) {
            if (match.first != -1 && match.second != -1) {
                valid_count++;
                if (match.first == match.second) {
                    match_count++;
                }
            }
        }
        
        double cluster_match_rate = valid_count > 0 ? 
            static_cast<double>(match_count) / valid_count : 0.0;
        
        std::cout << "Cluster analysis: " << match_count << " / " << valid_count 
                  << " queries matched (" << cluster_match_rate * 100.0 << "%)" << std::endl;
        
        // 输出一些不匹配的例子
        int examples_shown = 0;
        for (size_t i = 0; i < cluster_matches.size() && examples_shown < 5; ++i) {
            const auto& match = cluster_matches[i];
            if (match.first != -1 && match.second != -1 && match.first != match.second) {
                std::cout << "Example query " << i << ": "
                          << "Groundtruth cluster=" << match.first 
                          << ", TopK cluster=" << match.second << std::endl;
                
                // 输出groundtruth和TopK结果的距离分布
                const auto& gt_ids = groundtruth[i];
                std::cout << "  Groundtruth distances:" << std::endl;
                for (int j = 0; j < std::min(3, static_cast<int>(gt_ids.size())); ++j) {
                    if (gt_ids[j] >= 0 && gt_ids[j] < original_vectors.size()) {
                        float dist = std::sqrt(euclidean_distance_squared(
                            original_query_vectors[i], original_vectors[gt_ids[j]]));
                        std::cout << "    ID=" << gt_ids[j] << ", dist=" << dist 
                                  << ", cluster=" << match.first << std::endl;
                    }
                }
                
                std::cout << "  TopK results distances:" << std::endl;
                for (int j = 0; j < std::min(3, static_cast<int>(all_topk_results[i].size())); ++j) {
                    int id = all_topk_results[i][j];
                    if (id >= 0 && id < original_vectors.size()) {
                        float dist = std::sqrt(euclidean_distance_squared(
                            original_query_vectors[i], original_vectors[id]));
                        // 计算该结果属于哪个聚类中心
                        int res_cluster = -1;
                        float min_dist = std::numeric_limits<float>::max();
                        for (size_t c = 0; c < centroids.size(); ++c) {
                            float c_dist = euclidean_distance_squared(
                                original_vectors[id], centroids[c]);
                            if (c_dist < min_dist) {
                                min_dist = c_dist;
                                res_cluster = c;
                            }
                        }
                        std::cout << "    ID=" << id << ", dist=" << dist 
                                  << ", cluster=" << res_cluster << std::endl;
                    }
                }
                
                examples_shown++;
            }
        }
    }
    
    return {recall, throughput};
}

int main(int argc, char* argv[]) {
    int target_dim = 64;        // 目标降维维度
    int n_lists = 100;          // IVFFlat索引的聚类数量
    int k = 10;                 // KNN的K值
    int thread_count = std::thread::hardware_concurrency(); // 线程数
    bool use_sjlt = false;      // 是否使用SJLT降维
    int sparsity = 3;           // SJLT稀疏度
    std::vector<int> nprob_values = {1, 5, 10, 20}; // 不同的探测参数
    
    // 解析命令行参数
    int opt;
    while ((opt = getopt(argc, argv, "d:l:k:t:s:p:h")) != -1) {
        switch (opt) {
            case 'd':
                target_dim = std::stoi(optarg);
                break;
            case 'l':
                n_lists = std::stoi(optarg);
                break;
            case 'k':
                k = std::stoi(optarg);
                break;
            case 't':
                thread_count = std::stoi(optarg);
                break;
            case 's':
                use_sjlt = true;
                sparsity = std::stoi(optarg);
                break;
            case 'p': {
                std::stringstream ss(optarg);
                std::string token;
                nprob_values.clear();
                while (std::getline(ss, token, ',')) {
                    nprob_values.push_back(std::stoi(token));
                }
                break;
            }
            case 'h':
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "  -d <dim>       Target dimension for JL reduction (default: 64)\n"
                          << "  -l <lists>     Number of lists for IVFFlat index (default: 100)\n"
                          << "  -k <k>         Number of nearest neighbors to search (default: 10)\n"
                          << "  -t <threads>   Number of threads (default: hardware concurrency)\n"
                          << "  -s <sparsity>  Use SJLT with given sparsity (default: 3)\n"
                          << "  -p <probes>    Comma-separated list of nprob values (default: 1,5,10,20)\n"
                          << "  -h             Show this help message\n";
                return 0;
        }
    }
    
    std::cout << "Starting IVFFlat with JL/SJLT experiment..." << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Target dimension: " << target_dim << std::endl;
    std::cout << "  Number of lists: " << n_lists << std::endl;
    std::cout << "  k: " << k << std::endl;
    std::cout << "  Threads: " << thread_count << std::endl;
    std::cout << "  Reduction method: " << (use_sjlt ? "SJLT" : "JL") << std::endl;
    if (use_sjlt) {
        std::cout << "  SJLT sparsity: " << sparsity << std::endl;
    }
    std::cout << "  nprob values: ";
    for (int p : nprob_values) {
        std::cout << p << " ";
    }
    std::cout << std::endl;
    
    // 记录预处理时间
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    
    // 加载数据
    std::cout << "\nLoading data..." << std::endl;
    auto database = load_data("../sift/sift_base.fvecs", 128);
    auto queries = load_data("../sift/sift_query.fvecs", 128);
    auto groundtruth = load_groundtruth("../sift/sift_groundtruth.ivecs", 100);
    
    // 验证groundtruth
    validate_groundtruth(database, queries, groundtruth);
    
    // 标准化数据
    std::cout << "\nNormalizing data..." << std::endl;
    auto database_normalized = normalize_dataset(database);
    auto queries_normalized = normalize_dataset(queries);
    
    // 降维
    std::cout << "\nPerforming dimensionality reduction..." << std::endl;
    std::vector<std::vector<float>> database_reduced;
    std::vector<std::vector<float>> queries_reduced;
    
    if (use_sjlt) {
        std::cout << "Using SJLT with sparsity " << sparsity << "..." << std::endl;
        database_reduced = perform_sjlt_reduction(database_normalized, target_dim, sparsity);
        queries_reduced = perform_sjlt_reduction(queries_normalized, target_dim, sparsity);
    } else {
        std::cout << "Using JL projection..." << std::endl;
        database_reduced = perform_jl_reduction(database_normalized, target_dim);
        queries_reduced = perform_jl_reduction(queries_normalized, target_dim);
    }
    
    std::cout << "Original dimension: " << database[0].size() 
              << ", Reduced dimension: " << database_reduced[0].size() << std::endl;
    
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    pre_process_time = std::chrono::duration<double>(preprocess_end - preprocess_start).count();
    std::cout << "Preprocessing time: " << pre_process_time << " seconds" << std::endl;
    
    // 连接数据库
    PGconn* conn = connect_to_db();
    if (!conn) return 1;
    
    // 创建表和索引
    std::cout << "\nCreating table..." << std::endl;
    create_table_and_load_data(conn, database_reduced, target_dim);
    
    std::cout << "Creating index with " << n_lists << " lists..." << std::endl;
    create_ivfflat_index(conn, n_lists);
    
    // 获取聚类中心
    std::cout << "\nFetching IVFFlat centroids..." << std::endl;
    auto centroids = get_ivfflat_centroids(conn, n_lists, target_dim);
    std::cout << "Successfully retrieved " << centroids.size() << " centroids of dimension " 
              << (centroids.empty() ? 0 : centroids[0].size()) << std::endl;
    
    // 执行实验
    std::cout << "\nPerforming cluster analysis experiment..." << std::endl;
    std::vector<BenchmarkResult> results;
    
    for (int nprob : nprob_values) {
        std::cout << "\nTesting with nprob = " << nprob << std::endl;
        auto [recall, throughput] = test_knn_search(
            conn, queries_reduced, groundtruth, database, queries, centroids, k, nprob, thread_count, true);
        
        results.emplace_back(nprob, recall, throughput);
    }
    
    // 输出汇总结果
    std::cout << "\nSummary:" << std::endl;
    std::cout << "nprob\tRecall@" << k << "\tThroughput (qps)" << std::endl;
    for (const auto& res : results) {
        std::cout << res.nprob << "\t" << res.recall << "\t\t" << res.throughput << std::endl;
    }
    
    // 清理资源
    execute_sql(conn, "DROP TABLE IF EXISTS sift_data;");
    PQfinish(conn);
    
    return 0;
}