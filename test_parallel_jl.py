import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import time
import io
import json
import argparse
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
from psycopg2 import pool

DB_PARAMS = {
    'host': 'localhost',
    'port': '5432',
    'database': 'mydatabase',
    'user': 'postgres',
    'password': '1'
}

# 新增：全局线程安全的连接池
threaded_postgresql_pool = pool.ThreadedConnectionPool(
    minconn=5,
    maxconn=20,
    **DB_PARAMS
)

def load_data(file_path, dim):
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)
        data = data.reshape(-1, dim)
        data = data[:, 1:].astype(float)
    return data

def connect_to_db():
    try:
        return threaded_postgresql_pool.getconn()  # 从连接池获取
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def release_db_connection(conn):
    threaded_postgresql_pool.putconn(conn)  # 放回连接池

def create_table_and_load_data(conn, data, reduced_dim):
    cur = conn.cursor()
    try:
        # 启用vector扩展
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # # 删除已存在的表
        # cur.execute("DROP TABLE IF EXISTS sift_data;")
        
        # # 创建新表
        # cur.execute(f"CREATE TABLE sift_data (id serial, vector vector({reduced_dim}));")
        
        # # 分批次插入数据
        # batch_size = 5000
        # for i in range(0, len(data), batch_size):
        #     batch = data[i:i + batch_size]
        #     for row in batch:
        #         # 将向量转换为字符串格式，确保数值格式正确
        #         vector_str = '[' + ','.join(f"{x:.6f}" for x in row) + ']'
        #         cur.execute(
        #             "INSERT INTO sift_data (vector) VALUES (%s::vector);",
        #             (vector_str,)
        #         )
        #     conn.commit()
        #     print(f"Inserted batch {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}")
        
        # print("Data loading completed successfully")
        
    except psycopg2.Error as e:
        print(f"Error in create_table_and_load_data: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()

def create_ivfflat_index(conn, n_lists):
    cur = conn.cursor()
    try:
        # 检查表是否存在
        cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'sift_data');")
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            raise Exception("Table sift_data does not exist. Please load data first.")
        
        # 增加维护工作内存
        cur.execute("SET maintenance_work_mem = '1GB';")
        
        # 创建索引
        cur.execute(f"""
            CREATE INDEX ON sift_data 
            USING ivfflat (vector vector_l2_ops) 
            WITH (lists = {n_lists});
        """)
        conn.commit()
        print("Index created successfully")
        
    except psycopg2.Error as e:
        print(f"Error in create_ivfflat_index: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()

def single_query(args):
    """包装单个查询的线程任务"""
    query_vec, groundtruth_i, k, nprob = args
    conn = connect_to_db()
    try:
        cur = conn.cursor()
        vector_str = '[' + ','.join(f"{x:.6f}" for x in query_vec) + ']'
        
        # 设置会话级参数
        cur.execute(f"SET ivfflat.probes = {nprob};")
        
        # 执行查询
        start_time = time.time()
        cur.execute(
            f"SELECT id FROM sift_data ORDER BY vector <-> %s::vector LIMIT {k};",
            (vector_str,)
        )
        results = [row[0]-1 for row in cur.fetchall()]
        latency = time.time() - start_time
        
        # 计算召回率
        intersection = len(set(results) & set(groundtruth_i))
        recall = intersection / k
        
        return recall, latency
    finally:
        release_db_connection(conn)

def test_knn_search(conn, query_vectors, groundtruth, k, nprob, max_workers):
    """并行批量查询版本"""
    
    # 准备任务参数
    task_args = [(qv, gt, k, nprob) 
                for qv, gt in zip(query_vectors, groundtruth)]
    
    
    # 使用线程池并行执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        total_start = time.time()
        futures = list(executor.map(single_query, task_args))
        total_end = time.time()
        # 汇总结果
        total_recall = 0.0
        total_latency = 0.0
        n_queries = 0
        
        for recall, latency in futures:
            total_recall += recall
            total_latency += latency
            n_queries += 1
    
    total_time = total_end - total_start
    avg_recall = total_recall / n_queries
    throughput = n_queries / total_time
    
    return avg_recall, throughput

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_parallel_workers_per_gather', type=int, default=20,
                      help='PostgreSQL并行工作线程数')
    parser.add_argument('--max_workers', type=int, default=8,
                      help='应用层线程池大小')  # 新增参数
    args = parser.parse_args()

    try:
        # 加载数据集
        print("Loading datasets...")
        database = load_data('../sift/sift_base.fvecs',129)
        queries = load_data('../sift/sift_query.fvecs',129)
        groundtruth = load_data("../sift/sift_groundtruth.ivecs",101)
        print("Datasets loaded successfully")
        
        # 计算JL降维后的维度
        n_samples = len(database)
        eps = 0.1  # 误差容忍度
        reduced_dim = min(
            johnson_lindenstrauss_min_dim(n_samples, eps=eps),
            database.shape[1] // 2
        )
        print(f"Original dimension: {database.shape[1]}")
        print(f"Reduced dimension: {reduced_dim}")

        # 对数据进行降维
        transformer = GaussianRandomProjection(n_components=reduced_dim)

        # 标准化数据
        scaler = StandardScaler()
        database_scaled = scaler.fit_transform(database)
        queries_scaled = scaler.transform(queries)

        # 应用JL降维
        database_reduced = transformer.fit_transform(database_scaled)
        queries_reduced = transformer.transform(queries_scaled)
        
        # 再次标准化降维后的数据
        database_reduced = scaler.fit_transform(database_reduced)
        queries_reduced = scaler.transform(queries_reduced)
        
        # 连接到数据库
        print("Connecting to database...")
        conn = connect_to_db()
        if conn is None:
            return
        
        # 创建表并加载数据
        print("Creating table and loading data...")
        create_table_and_load_data(conn, database_scaled, 128)
        
        # 创建IVFFlat索引
        print("Creating index...")
        n_lists = 100
        create_ivfflat_index(conn, n_lists)
        # 设置并行参数（注意这需要PostgreSQL 11+版本）

        cur = conn.cursor()
        cur.execute(f"""
            SET max_parallel_workers_per_gather = {args.max_parallel_workers_per_gather};
            SET max_parallel_workers = {args.max_parallel_workers_per_gather * 2};
        """)
        cur.execute("SELECT pg_reload_conf();")  # 重载配置
        conn.commit()
        cur.close()
        
        # 测试不同nprob参数
        print("Testing performance...")
        nprob_values = [1, 5]  # 测试参数
        
        results = []
        for nprob in nprob_values:
            # 修改调用参数
            recall, throughput = test_knn_search(
                conn, queries_scaled, groundtruth, 
                k=10, nprob=nprob, max_workers=args.max_workers
            )
            results.append({
                'nprob': nprob,
                'recall': recall,
                'throughput': throughput
            })
        
        print(json.dumps(results))
        
    finally:
        threaded_postgresql_pool.closeall()  # 关闭连接池

if __name__ == "__main__":
    main()