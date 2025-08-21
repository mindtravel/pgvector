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
#include <algorithm>
#include <numeric>
#include <random>
#include <iterator>
#include <set>
#include <iomanip>
#include <stdexcept>

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

std::vector<float> database_norms;    // 原始数据库向量的模长
std::vector<float> query_norms;       // 原始查询向量的模长

// 新增：计算向量集合的模长并存储
void precompute_norms(const std::vector<std::vector<float>>& vectors, std::vector<float>& norms) {
    norms.reserve(vectors.size());
    for (const auto& vec : vectors) {
        norms.push_back(vector_norm(vec));  // 复用已有的vector_norm函数
    }
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

// 新增：基于预计算模长的距离平方计算（重排序专用）
float optimized_distance_squared(
    const std::vector<float>& a,          // 查询向量（原始高维）
    const std::vector<float>& b,          // 数据库向量（原始高维）
    float a_norm,                         // 查询向量的模长（预计算）
    float b_norm                          // 数据库向量的模长（预计算）
) {
    // 计算点积 a·b
    float dot_product = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
    }
    // 应用公式计算距离平方
    return a_norm * a_norm + b_norm * b_norm - 2 * dot_product;
}

std::vector<std::vector<float>> simple_matrix_multiply(
    const std::vector<std::vector<float>>& A,
    const std::vector<std::vector<float>>& B) {
    
    // 检查矩阵维度是否匹配：A的列数必须等于B的行数
    int m = A.size();
    int k = (m > 0) ? A[0].size() : 0; // A的列数
    int n = (B.size() > 0) ? B[0].size() : 0; // B的列数
    
    std::cout<<"A rows: " << m << ", A cols: " << k 
        << ", B rows: " << B.size() << ", B cols: " << n << std::endl;
    if (k == 0 || B.size() != k) {
        throw std::invalid_argument("Matrix dimensions do not match: A.cols != B.rows");
    }

    // 初始化结果矩阵C（m行n列）
    std::vector<std::vector<float>> C(m, std::vector<float>(n, 0.0f));

    // 核心三重循环实现矩阵乘法
    for (int i = 0; i < m; ++i) {        // 遍历A的行
        for (int p = 0; p < k; ++p) {    // 遍历A的列（同时是B的行）
            float a_ip = A[i][p];        // 缓存A的当前元素
            for (int j = 0; j < n; ++j) { // 遍历B的列
                C[i][j] += a_ip * B[p][j]; // 累加乘积
            }
        }
    }

    return C;
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

// std::vector<std::vector<float>> optimized_matrix_multiply(
//     const std::vector<std::vector<float>>& A,
//     const std::vector<std::vector<float>>& B,
//     int batch_size = 8,
//     int block_size = 32,
//     int num_threads = std::thread::hardware_concurrency()) {
    
//     int m = A.size();
//     int n = B[0].size();
//     int k = B.size();
    
//     std::vector<std::vector<float>> C(m, std::vector<float>(n, 0.0f));
    
//     // 每个线程处理的批次数
//     int batches_per_thread = (m + batch_size * num_threads - 1) / (batch_size * num_threads);
    
//     auto worker = [&](int thread_id) {
//         int start_batch = thread_id * batches_per_thread;
//         int end_batch = std::min(start_batch + batches_per_thread, (m + batch_size - 1) / batch_size);
        
//         for (int b = start_batch; b < end_batch; ++b) {
//             int start_i = b * batch_size;
//             int end_i = std::min(start_i + batch_size, m);
            
//             for (int i = start_i; i < end_i; ++i) {
//                 for (int j = 0; j < n; j += 8) { // AVX向量化
//                     __m256 sum = _mm256_setzero_ps();
                    
//                     for (int p = 0; p < k; p += block_size) { // 分块
//                         int end_p = std::min(p + block_size, k);
                        
//                         for (int pb = p; pb < end_p; ++pb) {
//                             __m256 a_val = _mm256_set1_ps(A[i][pb]);
//                             __m256 b_vals = _mm256_loadu_ps(&B[pb][j]);
//                             sum = _mm256_fmadd_ps(a_val, b_vals, sum);
//                         }
//                     }
                    
//                     _mm256_storeu_ps(&C[i][j], sum);
//                 }
                
//                 // 处理剩余的列（如果n不是8的倍数）
//                 if (n % 8 != 0) {
//                     int j_start = (n / 8) * 8;
//                     for (int j = j_start; j < n; ++j) {
//                         float sum = 0.0f;
//                         for (int p = 0; p < k; ++p) {
//                             sum += A[i][p] * B[p][j];
//                         }
//                         C[i][j] = sum;
//                     }
//                 }
//             }
//         }
//     };
    
//     std::vector<std::thread> threads;
//     for (int t = 0; t < num_threads; ++t) {
//         threads.emplace_back(worker, t);
//     }
    
//     for (auto& t : threads) {
//         t.join();
//     }
    
//     return C;
// }

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

// 生成JL随机投影矩阵
std::vector<std::vector<float>> generate_jl_projection_matrix(int original_dim, int target_dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    std::vector<std::vector<float>> projection(original_dim, std::vector<float>(target_dim));
    
    for (int i = 0; i < original_dim; ++i) {
        for (int j = 0; j < target_dim; ++j) {
            float r = dis(gen);
            if (r < 0.1667f) {  // 1/6的概率为+1
            projection[i][j] = 1.0f;
        } else if (r < 0.3333f) {  // 1/6的概率为-1
            projection[i][j] = -1.0f;
        } else {  // 2/3的概率为0
            projection[i][j] = 0.0f;
        }
        }
    }
    
    return projection;
}

// // 执行JL降维
// std::vector<std::vector<float>> perform_jl_reduction(
//     const std::vector<std::vector<float>>& data,
//     int target_dim) {
    
//     int original_dim = data[0].size();
//     int num_vectors = data.size();
    
//     // 生成随机投影矩阵
//     auto projection = generate_jl_projection_matrix(original_dim, target_dim);
    
//     // 转置数据以便于矩阵乘法
//     std::vector<std::vector<float>> data_t(original_dim, std::vector<float>(num_vectors));
//     for (int i = 0; i < num_vectors; ++i) {
//         for (int j = 0; j < original_dim; ++j) {
//             data_t[j][i] = data[i][j];
//         }
//     }
    
//     // 使用优化的矩阵乘法执行降维
//     auto reduced_t = simple_matrix_multiply(projection, data_t);
    
//     // 转置结果回原始格式
//     std::vector<std::vector<float>> reduced(num_vectors, std::vector<float>(target_dim));
//     for (int i = 0; i < target_dim; ++i) {
//         for (int j = 0; j < num_vectors; ++j) {
//             reduced[j][i] = reduced_t[i][j];
//         }
//     }
    
//     return reduced;
// }

// // 执行JL降维
// std::vector<std::vector<float>> perform_jl_reduction(
//     const std::vector<std::vector<float>>& data,
//     int target_dim) {
    
//     int original_dim = data[0].size();
//     int num_vectors = data.size();
    
//     // 生成随机投影矩阵
//     auto projection = generate_jl_projection_matrix(original_dim, target_dim);
    
//     // 使用优化的矩阵乘法执行降维
//     auto reduced = simple_matrix_multiply(data, projection);
    
//     return reduced;
// }

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
    std::vector<std::vector<int>> col_indices(original_dim);    // 每一行的列索引
    std::vector<std::vector<float>> values(original_dim);       // 每一行的非零值
    
    float scale = std::sqrt(sparsity / static_cast<float>(target_dim));
    
    for (int i = 0; i < original_dim; ++i) {
        for (int j = 0; j < target_dim; ++j) {
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
    SparseMatrix& projection,
    int target_dim) {
    
    int original_dim = data[0].size();
    int num_vectors = data.size();
    
    auto col_indices = projection.first;  // 稀疏矩阵的列索引
    auto values = projection.second;       // 稀疏矩阵的非零值

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
    const std::vector<std::vector<float>>& groundtruth,
    int sample_size = 10) {
    
    std::cout << "\nValidating groundtruth..." << std::endl;
    
    for (int i = 0; i < sample_size; ++i) {
        const auto& query = queries[i];
        std::cout << "Query " << i << ":\n";
        
        // 计算与groundtruth中前5个ID的距离
        for (int j = 0; j < std::min(5, static_cast<int>(groundtruth[i].size())); ++j) {
            int gt_id = static_cast<int>(groundtruth[i][j]);
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

// KNN搜索测试 - 修正ID匹配逻辑
std::pair<double, double> test_knn_search(
    PGconn* conn,
    const std::vector<std::vector<float>>& queries,
    const std::vector<std::vector<float>>& groundtruth,
    int k,
    int nprob,
    int thread_count = std::thread::hardware_concurrency()
) {
    if (thread_count <= 0) thread_count = 1;
    const size_t query_count = queries.size();
    const size_t queries_per_thread = (query_count + thread_count - 1) / thread_count;
    
    std::vector<std::thread> threads;
    std::vector<double> thread_recalls(thread_count, 0.0);
    std::vector<double> thread_times(thread_count, 0.0);
    
    // 跟踪有召回的查询数量
    std::vector<int> thread_hits(thread_count, 0);
    std::vector<int> thread_tests(thread_count, 0);
    
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
                                  "ORDER BY vector <-> " + vec_str + " LIMIT " + std::to_string(k) + ";";
                
                auto query_start = std::chrono::high_resolution_clock::now();
                PGresult* res = PQexec(thread_conn, sql.c_str());
                auto query_end = std::chrono::high_resolution_clock::now();
                
                if (PQresultStatus(res) != PGRES_TUPLES_OK) {
                    std::cerr << "Thread " << t << ", Query " << i << " failed: " 
                              << PQerrorMessage(thread_conn) << std::endl;
                    PQclear(res);
                    continue;
                }
                
                // 收集查询结果ID和距离
                std::vector<int> results;
                std::vector<float> distances;
                int rows = PQntuples(res);
                for (int r = 0; r < rows; ++r) {
                    int db_id = atoi(PQgetvalue(res, r, 0));
                    float distance = atof(PQgetvalue(res, r, 1));
                    results.push_back(db_id);
                    distances.push_back(distance);
                }
                
                // 获取当前查询的groundtruth，并将ID加1以匹配数据库ID
                std::vector<int> gt;
                for (float val : groundtruth[i]) {
                    // 将groundtruth中的ID加1，使其与数据库ID匹配
                    gt.push_back(static_cast<int>(val) + 1);
                }
                
                // 输出样本数据用于调试
                if (i == start_idx || (i % 1000 == 0 && i < 10000)) {
                    std::cout << "Thread " << t << ", Query " << i << " sample:" << std::endl;
                    std::cout << "  Query results (first 5):";
                    for (int j = 0; j < std::min(5, static_cast<int>(results.size())); ++j) {
                        std::cout << " ID=" << results[j] << "(dist=" << distances[j] << ")";
                    }
                    std::cout << std::endl;
                    
                    std::cout << "  Adjusted Groundtruth (first 5):";
                    for (int j = 0; j < std::min(5, static_cast<int>(gt.size())); ++j) {
                        std::cout << " " << gt[j];
                    }
                    std::cout << std::endl;
                }
                
                // 计算交集
                std::sort(results.begin(), results.end());
                std::sort(gt.begin(), gt.end());
                
                std::vector<int> intersection;
                std::set_intersection(
                    results.begin(), results.end(),
                    gt.begin(), gt.end(),
                    std::back_inserter(intersection)
                );
                
                double query_recall = static_cast<double>(intersection.size()) / k;
                local_recall += query_recall;
                local_time += std::chrono::duration<double>(query_end - query_start).count();
                local_tests++;
                
                if (intersection.size() > 0) {
                    local_hits++;
                    if (i % 1000 == 0) {
                        std::cout << "HIT! Thread " << t << ", Query " << i 
                                  << " recall: " << query_recall 
                                  << ", intersection size: " << intersection.size() << std::endl;
                    }
                }
                
                PQclear(res);
            }
            
            thread_recalls[t] = local_recall;
            thread_times[t] = local_time;
            thread_hits[t] = local_hits;
            thread_tests[t] = local_tests;
            
            // 输出线程汇总信息
            std::cout << "Thread " << t << " completed: " 
                      << local_tests << " queries, "
                      << local_hits << " had non-zero recall" << std::endl;
            
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
    double total_recall = std::accumulate(thread_recalls.begin(), thread_recalls.end(), 0.0);
    int total_hits = std::accumulate(thread_hits.begin(), thread_hits.end(), 0);
    int total_tests = std::accumulate(thread_tests.begin(), thread_tests.end(), 0);
    
    std::cout << "Total queries: " << total_tests << std::endl;
    std::cout << "Queries with non-zero recall: " << total_hits 
              << " (" << (double)total_hits/total_tests*100.0 << "%)" << std::endl;
    
    return {
        total_recall / query_count,
        query_count / total_time
    };
}

// 新增：带重排序的KNN搜索测试
std::pair<double, double> test_knn_search_with_reordering(
    PGconn* conn,
    const std::vector<std::vector<float>>& queries,                  // 原始高维查询向量（用于重排序）
    const std::vector<std::vector<float>>& queries_reduced,          // 降维后的查询向量（用于初始检索）
    const std::vector<std::vector<float>>& database,                 // 原始高维数据库向量（用于重排序）
    const std::vector<std::vector<float>>& groundtruth,
    int k,                  // 最终返回的近邻数
    int n,                  // 候选向量倍数（候选数 = k * n）
    int nprob,
    int thread_count = std::thread::hardware_concurrency()
) {
    if (thread_count <= 0) thread_count = 1;
    const size_t query_count = queries.size();
    const size_t queries_per_thread = (query_count + thread_count - 1) / thread_count;
    const int candidate_k = k * n;  // 候选向量数量
    
    std::vector<std::thread> threads;
    std::vector<double> thread_recalls(thread_count, 0.0);
    std::vector<double> thread_times(thread_count, 0.0);
    
    // 跟踪有召回的查询数量
    std::vector<int> thread_hits(thread_count, 0);
    std::vector<int> thread_tests(thread_count, 0);
    
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
                // 1. 构建查询语句，获取k*n个候选向量
                std::string vec_str = "'[";
                for (size_t j = 0; j < queries_reduced[i].size(); ++j) {
                    vec_str += (j > 0 ? "," : "") + std::to_string(queries_reduced[i][j]);
                }
                vec_str += "]'::vector(" + std::to_string(queries_reduced[i].size()) + ")";
                
                std::string sql = "SELECT id, (vector <-> " + vec_str + ") as distance FROM sift_data "
                                  "ORDER BY vector <-> " + vec_str + " LIMIT " + std::to_string(candidate_k) + ";";
                
                auto query_start = std::chrono::high_resolution_clock::now();
                PGresult* res = PQexec(thread_conn, sql.c_str());
                auto query_end = std::chrono::high_resolution_clock::now();
                
                if (PQresultStatus(res) != PGRES_TUPLES_OK) {
                    std::cerr << "Thread " << t << ", Query " << i << " failed: " 
                              << PQerrorMessage(thread_conn) << std::endl;
                    PQclear(res);
                    continue;
                }
                
                // 2. 收集候选结果的ID（数据库ID是从1开始的，需要转换为0基索引）
                std::vector<std::pair<int, float>> candidates;  // (数据库原始索引, 降维距离)
                int rows = PQntuples(res);
                for (int r = 0; r < rows; ++r) {
                    int db_id = atoi(PQgetvalue(res, r, 0));  // 数据库中的ID（1基）
                    int original_idx = db_id - 1;  // 转换为原始数据集中的索引（0基）
                    
                    // 验证索引有效性
                    if (original_idx >= 0 && original_idx < database.size()) {
                        float distance = atof(PQgetvalue(res, r, 1));
                        candidates.emplace_back(original_idx, distance);
                    }
                }
                PQclear(res);
                
                // 3. 使用原始高维向量重新计算距离并排序（重排序核心步骤）
                // const auto& original_query = queries[i];  // 原始高维查询向量
                // std::vector<std::pair<int, float>> reordered;  // (原始索引, 高维距离)
                
                // for (const auto& [idx, _] : candidates) {
                //     // 计算原始高维空间中的欧氏距离平方
                //     float dist_sq = euclidean_distance_squared(original_query, database[idx]);
                //     reordered.emplace_back(idx, dist_sq);
                // }
                
                const auto& original_query = queries[i];  // 原始高维查询向量
                float query_norm = query_norms[i];        // 预计算的查询向量模长
                std::vector<std::pair<int, float>> reordered;  // (原始索引, 高维距离平方)
                
                for (const auto& [idx, _] : candidates) {
                    // 从预计算的模长数组中获取数据库向量的模长
                    float db_norm = database_norms[idx];
                    // 使用优化的距离公式计算（避免重复计算模长）
                    float dist_sq = optimized_distance_squared(
                        original_query, database[idx], query_norm, db_norm
                    );
                    reordered.emplace_back(idx, dist_sq);
                }

                // 按高维距离升序排序
                std::sort(reordered.begin(), reordered.end(), 
                          [](const auto& a, const auto& b) { 
                              return a.second < b.second; 
                          });
                
                // 4. 取前k个结果
                std::vector<int> top_k_results;
                for (size_t j = 0; j < std::min(k, (int)reordered.size()); ++j) {
                    top_k_results.push_back(reordered[j].first + 1);  // 转回数据库ID（1基）
                }
                
                // 5. 与groundtruth比较计算召回率
                std::vector<int> gt;
                for (float val : groundtruth[i]) {
                    gt.push_back(static_cast<int>(val) + 1);  // 转换为数据库ID（1基）
                }
                
                // 计算交集
                std::sort(top_k_results.begin(), top_k_results.end());
                std::sort(gt.begin(), gt.end());
                
                std::vector<int> intersection;
                std::set_intersection(
                    top_k_results.begin(), top_k_results.end(),
                    gt.begin(), gt.end(),
                    std::back_inserter(intersection)
                );
                
                // 6. 统计结果
                double query_recall = static_cast<double>(intersection.size()) / k;
                local_recall += query_recall;
                local_time += std::chrono::duration<double>(query_end - query_start).count();
                local_tests++;
                
                if (intersection.size() > 0) {
                    local_hits++;
                    if (i % 1000 == 0) {
                        std::cout << "HIT! Thread " << t << ", Query " << i 
                                  << " recall: " << query_recall 
                                  << ", intersection size: " << intersection.size() << std::endl;
                    }
                }
            }
            
            thread_recalls[t] = local_recall;
            thread_times[t] = local_time;
            thread_hits[t] = local_hits;
            thread_tests[t] = local_tests;
            
            // 输出线程汇总信息
            std::cout << "Thread " << t << " completed: " 
                      << local_tests << " queries, "
                      << local_hits << " had non-zero recall" << std::endl;
            
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
    double total_recall = std::accumulate(thread_recalls.begin(), thread_recalls.end(), 0.0);
    int total_hits = std::accumulate(thread_hits.begin(), thread_hits.end(), 0);
    int total_tests = std::accumulate(thread_tests.begin(), thread_tests.end(), 0);
    
    std::cout << "Total queries: " << total_tests << std::endl;
    std::cout << "Queries with non-zero recall: " << total_hits 
              << " (" << (double)total_hits/total_tests*100.0 << "%)" << std::endl;
    
    return {
        total_recall / query_count,
        query_count / total_time
    };
}

int main(int argc, char* argv[]) {
    int max_parallel_workers = 20;
    int target_dim = 64; // 默认目标维度
    int n_lists = 100;   // 默认IVFFlat索引的lists数量
    
    // 解析命令行参数
    struct option long_options[] = {
        {"max_parallel_workers_per_gather", required_argument, 0, 'p'},
        {"target_dim", required_argument, 0, 'd'},
        {"n_lists", required_argument, 0, 'l'},
        {0, 0, 0, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "p:d:l:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'p':
                max_parallel_workers = atoi(optarg);
                break;
            case 'd':
                target_dim = atoi(optarg);
                break;
            case 'l':
                n_lists = atoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] 
                          << " [-p max_parallel_workers_per_gather] [-d target_dim] [-l n_lists]" << std::endl;
                return 1;
        }
    }

    try {
        std::cout << "Loading datasets..." << std::endl;
        auto database = load_data("../../sift/sift_base.fvecs", 128);
        auto queries = load_data("../../sift/sift_query.fvecs", 128);
        // 明确指定groundtruth是ivecs格式
        auto groundtruth = load_data("../../sift/sift_groundtruth.ivecs", 100, true);
        std::cout << "Datasets loaded successfully" << std::endl;
        
        // 输出样本数据用于验证
        if (!database.empty()) {
            std::cout << "Sample database vector[0]: ";
            for (int i = 0; i < std::min(5, static_cast<int>(database[0].size())); ++i) {
                std::cout << database[0][i] << " ";
            }
            std::cout << "\nNorm: " << vector_norm(database[0]) << std::endl;
        }
        
        if (!queries.empty()) {
            std::cout << "Sample query vector[0]: ";
            for (int i = 0; i < std::min(5, static_cast<int>(queries[0].size())); ++i) {
                std::cout << queries[0][i] << " ";
            }
            std::cout << "\nNorm: " << vector_norm(queries[0]) << std::endl;
        }
        
        if (!groundtruth.empty()) {
            std::cout << "Sample groundtruth[0]: ";
            for (int i = 0; i < std::min(5, static_cast<int>(groundtruth[0].size())); ++i) {
                std::cout << groundtruth[0][i] << " ";
            }
            std::cout << std::endl;
        }
        
        // 验证groundtruth
        // validate_groundtruth(database, queries, groundtruth);
        
        // // 向量标准化
        // std::cout << "\nNormalizing vectors..." << std::endl;
        // database = normalize_dataset(database);
        // queries = normalize_dataset(queries);
        
        // 输出标准化后的样本数据用于验证
        // if (!database.empty()) {
        //     std::cout << "Sample normalized database vector[0]: ";
        //     for (int i = 0; i < std::min(5, static_cast<int>(database[0].size())); ++i) {
        //         std::cout << database[0][i] << " ";
        //     }
        //     std::cout << "\nNorm: " << vector_norm(database[0]) << std::endl;
        // }
        
        // if (!queries.empty()) {
        //     std::cout << "Sample normalized query vector[0]: ";
        //     for (int i = 0; i < std::min(5, static_cast<int>(queries[0].size())); ++i) {
        //         std::cout << queries[0][i] << " ";
        //     }
        //     std::cout << "\nNorm: " << vector_norm(queries[0]) << std::endl;
        // }
        
        auto start_process = std::chrono::high_resolution_clock::now();
        // 执行JL降维
        std::cout << "\nPerforming JL dimensionality reduction to " << target_dim << " dimensions..." << std::endl;
        
        int original_dim = database[0].size();
        auto projection = generate_jl_projection_matrix(original_dim, target_dim);
        auto database_reduced = parallel_matrix_multiply(database, projection);
        auto queries_reduced = parallel_matrix_multiply(queries, projection);

        // int sparsity = 3; // 稀疏度参数，通常3-5
        // auto projection = generate_sjlt_projection_matrix(original_dim, target_dim, sparsity);
        // auto database_reduced = perform_sjlt_reduction(database, projection, target_dim);
        // auto queries_reduced = perform_sjlt_reduction(queries, projection, target_dim);

        auto end_process = std::chrono::high_resolution_clock::now();
        pre_process_time = std::chrono::duration<double>(end_process - start_process).count();
        std::cout << "JL dimensionality reduction completed in " 
                  << pre_process_time << " seconds" << std::endl;

        // 降维后再次标准化
        std::cout << "Normalizing reduced vectors..." << std::endl;
        database_reduced = normalize_dataset(database_reduced);
        queries_reduced = normalize_dataset(queries_reduced);
        
        std::cout << "Dimensionality reduction completed" << std::endl;

        std::cout << "\nPrecomputing vector norms for distance optimization..." << std::endl;
        precompute_norms(database, database_norms);  // 计算数据库向量的模长
        precompute_norms(queries, query_norms);      // 计算查询向量的模长
        std::cout << "Norm precomputation completed" << std::endl;
        
        // 输出降维后的样本数据用于验证
        if (!database_reduced.empty()) {
            std::cout << "Sample reduced database vector[0]: ";
            for (int i = 0; i < std::min(5, static_cast<int>(database_reduced[0].size())); ++i) {
                std::cout << database_reduced[0][i] << " ";
            }
            std::cout << "\nNorm: " << vector_norm(database_reduced[0]) << std::endl;
        }
        
        if (!queries_reduced.empty()) {
            std::cout << "Sample reduced query vector[0]: ";
            for (int i = 0; i < std::min(5, static_cast<int>(queries_reduced[0].size())); ++i) {
                std::cout << queries_reduced[0][i] << " ";
            }
            std::cout << "\nNorm: " << vector_norm(queries_reduced[0]) << std::endl;
        }

        PGconn* conn = connect_to_db();
        if (!conn) return 1;

        std::cout << "\nCreating table..." << std::endl;
        create_table_and_load_data(conn, database_reduced, target_dim);

        std::cout << "Creating index with " << n_lists << " lists..." << std::endl;
        create_ivfflat_index(conn, n_lists);

        std::cout << "Setting parallel workers..." << std::endl;
        execute_sql(conn, 
            "SET max_parallel_workers_per_gather = " + 
            std::to_string(max_parallel_workers) + ";");

        std::vector<BenchmarkResult> results;
        int thread_count = 20;
        std::cout << "Using " << thread_count << " threads for parallel queries" << std::endl;

        for (int nprob : {1, 5, 10, 20}) { // 测试更多nprob值
            std::cout << "\nTesting with nprob = " << nprob << std::endl;
            auto [recall, throughput] = f(
                conn, queries_reduced, groundtruth, 10, nprob, thread_count
            );
            results.push_back(BenchmarkResult(nprob, recall, throughput));
            std::cout << "Recall: " << recall 
                      << ", Throughput: " << throughput << " qps\n";
        }

        PQfinish(conn);
        std::cout << "Database connection closed" << std::endl;
        
        // 输出JSON结果
        std::ofstream json_file("jl_results.json");
        // std::ofstream json_file("sjlt_results.json");
    if (json_file.is_open()) {
        json_file << "[";
        for (size_t i = 0; i < results.size(); ++i) {
            json_file << "{"
                      << "\"nprob\":" << results[i].nprob << ","
                      << "\"recall\":" << results[i].recall << ","
                      << "\"throughput\":" << results[i].throughput
                      << "}" << (i < results.size()-1 ? "," : "");
        }
        json_file << "]" << std::endl;
        json_file.close();
        std::cout << "Results saved to jl_results.json" << std::endl;
        // std::cout << "Results saved to sjlt_results.json" << std::endl;
    }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}