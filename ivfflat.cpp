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

        // 标准化实现 (需自行实现)
        // ...

        PGconn* conn = connect_to_db();
        if (!conn) return 1;

        std::cout << "Creating table..." << std::endl;
        create_table_and_load_data(conn, database, 128);

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
            conn, queries, groundtruth, 10, nprob, thread_count
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