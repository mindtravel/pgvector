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

std::vector<std::vector<float>> optimized_matrix_multiply(
    const std::vector<std::vector<float>>& A,
    const std::vector<std::vector<float>>& B,
    int batch_size = 8,
    int block_size = 32,
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
                for (int j = 0; j < n; j += 8) { // AVX向量化
                    __m256 sum = _mm256_setzero_ps();
                    
                    for (int p = 0; p < k; p += block_size) { // 分块
                        int end_p = std::min(p + block_size, k);
                        
                        for (int pb = p; pb < end_p; ++pb) {
                            __m256 a_val = _mm256_set1_ps(A[i][pb]);
                            __m256 b_vals = _mm256_loadu_ps(&B[pb][j]);
                            sum = _mm256_fmadd_ps(a_val, b_vals, sum);
                        }
                    }
                    
                    _mm256_storeu_ps(&C[i][j], sum);
                }
                
                // 处理剩余的列（如果n不是8的倍数）
                if (n % 8 != 0) {
                    int j_start = (n / 8) * 8;
                    for (int j = j_start; j < n; ++j) {
                        float sum = 0.0f;
                        for (int p = 0; p < k; ++p) {
                            sum += A[i][p] * B[p][j];
                        }
                        C[i][j] = sum;
                    }
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

// 生成JL随机投影矩阵
std::vector<std::vector<float>> generate_jl_projection_matrix(int original_dim, int target_dim) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    std::vector<std::vector<float>> projection(target_dim, std::vector<float>(original_dim));
    
    for (int i = 0; i < target_dim; ++i) {
        for (int j = 0; j < original_dim; ++j) {
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

std::vector<std::vector<float>> generate_sjlt_projection_matrix(int original_dim, int target_dim, int sparsity = 3) {
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

// 数据加载函数
std::vector<std::vector<float>> load_data(const std::string& path, int dim) {
    std::ifstream file(path, std::ios::binary);
    std::vector<std::vector<float>> data;
    
    if (!file) {
        std::cerr << "Error opening file: " << path << std::endl;
        return data; // 返回空向量而不是退出程序
    }

    // 读取向量数量
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 计算向量数量
    // fvecs格式: 每个向量前有一个int表示维度，然后是dim个float
    size_t vector_size = (dim * sizeof(float)) + sizeof(int32_t);
    size_t num_vectors = file_size / vector_size;
    
    std::cout << "File size: " << file_size << " bytes" << std::endl;
    std::cout << "Vector size: " << vector_size << " bytes" << std::endl;
    std::cout << "Number of vectors expected: " << num_vectors << std::endl;
    
    // 读取每个向量
    for (size_t i = 0; i < num_vectors; ++i) {
        int32_t actual_dim;
        file.read(reinterpret_cast<char*>(&actual_dim), sizeof(int32_t));
        
        if (actual_dim != dim) {
            std::cerr << "Dimension mismatch at vector " << i 
                      << ": expected " << dim << ", got " << actual_dim << std::endl;
            break;
        }
        
        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        data.push_back(vec);
        
        // 检查读取是否成功
        if (file.gcount() != static_cast<std::streamsize>(dim * sizeof(float))) {
            std::cerr << "Incomplete read at vector " << i << std::endl;
            break;
        }
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
            row << (j > 0 ? "," : "") << data[i][j];
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

// KNN搜索测试
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
            
            for (size_t i = start_idx; i < end_idx; ++i) {
                std::string vec_str = "'[";
                for (size_t j = 0; j < queries[i].size(); ++j) {
                    vec_str += (j > 0 ? "," : "") + std::to_string(queries[i][j]);
                }
                vec_str += "]'::vector";
                
                std::string sql = "SELECT id FROM sift_data ORDER BY vector <-> " + 
                                  vec_str + " LIMIT " + std::to_string(k) + ";";
                
                PGresult* res = PQexec(thread_conn, sql.c_str());
                auto end = std::chrono::high_resolution_clock::now();
                
                if (PQresultStatus(res) != PGRES_TUPLES_OK) {
                    std::cerr << "Thread " << t << ", Query " << i << " failed: " 
                              << PQerrorMessage(thread_conn) << std::endl;
                    PQclear(res);
                    continue;
                }
                
                std::vector<int> results;
                int rows = PQntuples(res);
                for (int r = 0; r < rows; ++r) {
                    results.push_back(atoi(PQgetvalue(res, r, 0)) - 1);
                }
                
                std::vector<int> gt(groundtruth[i].begin(), groundtruth[i].end());
                std::sort(results.begin(), results.end());
                std::sort(gt.begin(), gt.end());
                
                std::vector<int> intersection;
                std::set_intersection(
                    results.begin(), results.end(),
                    gt.begin(), gt.end(),
                    std::back_inserter(intersection)
                );
                
                local_recall += static_cast<double>(intersection.size()) / k;
                PQclear(res);
            }
            
            thread_recalls[t] = local_recall;
            
            // 关闭线程的数据库连接
            PQfinish(thread_conn);
        });
    }
    
    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    // 汇总结果
    double total_recall = std::accumulate(thread_recalls.begin(), thread_recalls.end(), 0.0);
    
    return {
        total_recall / query_count,
        query_count / total_time
    };
}

int main(int argc, char* argv[]) {
    int max_parallel_workers = 20;
    
    // 解析命令行参数
    struct option long_options[] = {
        {"max_parallel_workers_per_gather", required_argument, 0, 'p'},
        {0, 0, 0, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "p:", long_options, NULL)) != -1) {
        if (opt == 'p') max_parallel_workers = atoi(optarg);
    }

    try {
        std::cout << "Loading datasets..." << std::endl;
        auto database = load_data("../sift/sift_base.fvecs", 128);
        auto queries = load_data("../sift/sift_query.fvecs", 128);
        auto groundtruth = load_data("../sift/sift_groundtruth.ivecs", 100);
        std::cout << "Datasets loaded successfully" << std::endl;
        
        // 执行JL降维
        int target_dim = 64; // 目标维度
        std::cout << "Performing JL dimensionality reduction to " << target_dim << " dimensions..." << std::endl;
        
        // auto database_reduced = perform_jl_reduction(database, target_dim);
        // auto queries_reduced = perform_jl_reduction(queries, target_dim);

        int sparsity = 3; // 稀疏度
        auto database_reduced = perform_sjlt_reduction(database, target_dim, sparsity);
        auto queries_reduced = perform_sjlt_reduction(queries, target_dim, sparsity);
        
        std::cout << "Dimensionality reduction completed" << std::endl;
        // 标准化实现 (需自行实现)
        // ...

        PGconn* conn = connect_to_db();
        if (!conn) return 1;

        std::cout << "Creating table..." << std::endl;
        create_table_and_load_data(conn, database_reduced, target_dim);

        std::cout << "Creating index..." << std::endl;
        create_ivfflat_index(conn, 100);

        std::cout << "Setting parallel workers..." << std::endl;
        execute_sql(conn, 
            "SET max_parallel_workers_per_gather = " + 
            std::to_string(max_parallel_workers) + ";");

        std::vector<BenchmarkResult> results;
        int thread_count = 20; // 自动确定线程数
        std::cout << "Using " << thread_count << " threads for parallel queries" << std::endl;

        for (int nprob : {1, 5}) {
            std::cout << "Testing with nprob = " << nprob << std::endl;
            auto [recall, throughput] = test_knn_search(
            conn, queries_reduced, groundtruth, 10, nprob, thread_count
            );
            results.push_back(BenchmarkResult(nprob, recall, throughput));
            std::cout << "Recall: " << recall 
              << ", Throughput: " << throughput << " qps\n";
        }

        PQfinish(conn);
        std::cout << "Database connection closed" << std::endl;
        
        // 输出JSON结果
        std::cout << "[";
        for (size_t i = 0; i < results.size(); ++i) {
            std::cout << "{"
                      << "\"nprob\":" << results[i].nprob << ","
                      << "\"recall\":" << results[i].recall << ","
                      << "\"throughput\":" << results[i].throughput
                      << "}" << (i < results.size()-1 ? "," : "");
        }
        std::cout << "]" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}