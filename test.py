import psycopg2
from psycopg2 import sql
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

# 配置数据库连接
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "mydatabase",
    "user": "postgres",
    "password": "1"
}

def get_table_counts(db_config):
    """
    获取数据库中所有表及其记录数
    """
    try:
        # 连接到PostgreSQL数据库
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        
        # 获取所有用户表（排除系统表）
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        
        # 调试：打印原始查询结果
        raw_tables = cur.fetchall()
        print("原始查询结果:", raw_tables)  # 调试用
        
        # 正确处理元组格式的表名
        tables = ["sift_base","sift_learn","sift_query"]
        
        # 获取每个表的记录数
        table_counts = {}
        for table in tables:
            try:
                cur.execute(
                    sql.SQL("SELECT COUNT(*) FROM {}").format(
                        sql.Identifier(table)
                    )
                )
                table_counts[table] = cur.fetchone()
            except Exception as e:
                print(f"查询表 {table} 时出错: {e}")
                table_counts[table] = "查询失败"
        print(table_counts)
        return table_counts
        
    except Exception as e:
        print(f"数据库操作出错: {e}")
        return None
        
    finally:
        if 'conn' in locals():
            conn.close()

def read_fvecs(filename):
    """读取.fvecs文件"""
    with open(filename, 'rb') as f:
        vecs = np.fromfile(f, dtype=np.float32)
    dim = vecs.view(np.int32)[0]
    vecs = vecs.reshape(-1, dim + 1)[:, 1:]
    return vecs

def read_ivecs(filename):
    """读取.ivecs文件"""
    with open(filename, 'rb') as f:
        vecs = np.fromfile(f, dtype=np.int32)
    dim = vecs.view(np.int32)[0]
    return vecs.reshape(-1, dim + 1)[:, 1:]

def main():
    get_table_counts(DB_CONFIG)
    # 连接到PostgreSQL数据库
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    
    # 1. 启用pgvector扩展
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # # 2. 删除所有现有表
    # cur.execute("""
    #     DROP TABLE IF EXISTS sift_base, sift_learn, sift_query, sift_train
    # """)
    
    # # 3. 创建三个表
    # cur.execute("""
    #     CREATE TABLE sift_base (
    #         id SERIAL PRIMARY KEY,
    #         vector vector(128)
    #     )
    # """)
    
    # cur.execute("""
    #     CREATE TABLE sift_learn (
    #         id SERIAL PRIMARY KEY,
    #         vector vector(128)
    #     )
    # """)
    
    # cur.execute("""
    #     CREATE TABLE sift_query (
    #         id SERIAL PRIMARY KEY,
    #         vector vector(128)
    #     )
    # """)
    
    # 4. 读取并插入数据
    print("Loading data...")
    base_vectors = read_fvecs("../sift/sift_base.fvecs")
    query_vectors = read_fvecs("../sift/sift_query.fvecs")
    learn_vectors = read_fvecs("../sift/sift_learn.fvecs")
    ground_truth = read_ivecs("../sift/sift_groundtruth.ivecs")
    
    # print("Inserting base vectors...")
    # for vec in tqdm(base_vectors):
    #     cur.execute("INSERT INTO sift_base (vector) VALUES (%s)", (vec.tolist(),))
    
    # print("Inserting query vectors...")
    # for vec in tqdm(query_vectors):
    #     cur.execute("INSERT INTO sift_query (vector) VALUES (%s)", (vec.tolist(),))
        
    # print("Inserting learn vectors...")
    # for vec in tqdm(learn_vectors):
    #     cur.execute("INSERT INTO sift_learn (vector) VALUES (%s)", (vec.tolist(),))
    
    get_table_counts(DB_CONFIG)
    # 5. 创建IVFFlat索引
    print("Creating IVFFlat index...")
    cur.execute("""
        CREATE INDEX ON sift_base USING ivfflat (vector) 
        WITH (lists = 100)
    """)
    
    # 6. 测试不同nprobe的性能和召回率
    nprobes = [1, 2,3,4,5,10,20]
    throughputs = []
    recalls = []
    k = 10  # 查询的最近邻数量
    
    # 直接从文件读取ground truth
    print("Loading ground truth from file...")
    ground_truth = ground_truth[:, :k]  # 只取前k个最近邻
    
    for nprobe in nprobes:
        print(f"\nTesting with nprobe = {nprobe}")
        cur.execute(f"SET ivfflat.probes = {nprobe}")
        
        # 检查执行计划
        print("\n检查查询执行计划:")
        cur.execute("""
            EXPLAIN ANALYZE 
            SELECT id FROM sift_base 
            ORDER BY vector <-> %s::vector 
            LIMIT %s
        """, (query_vectors[0].tolist(), k))
        
        # 打印执行计划
        for row in cur.fetchall():
            print(row[0])
        
        cur.execute("SELECT pg_get_indexdef(%s::regclass);", ('sift_base_vector_idx10',))
        definition = cur.fetchone()
        print(definition)

        # 测量吞吐量
        start_time = time.time()
        count = 0
        correct = 0
        
        for i, query in enumerate(query_vectors[:100]):
            cur.execute("""
                SELECT id FROM sift_base 
                ORDER BY vector <-> %s::vector
                LIMIT %s
            """, (query.tolist(), k))
            results = [row[0]-1 for row in cur.fetchall()]
            # 计算召回率
            intersection = len(set(results) & set(ground_truth[i]))
            correct += intersection / k
            
            count += 1
        
        duration = time.time() - start_time
        throughput = count / duration
        recall = correct / count
        
        throughputs.append(throughput)
        recalls.append(recall)
        
        print(f"\n测试结果 - nprobe={nprobe}:")
        print(f"吞吐量(Throughput): {throughput:.2f} qps")
        print(f"召回率(Recall): {recall:.4f}")

    # 绘制吞吐量-召回率曲线
    plt.figure(figsize=(10, 5))
    plt.plot(recalls, throughputs, 'o-')
    plt.xlabel('Recall')
    plt.ylabel('Throughput (queries per second)')
    plt.title('pgvector,sift,k=10')
    plt.grid(True)
    
    for i, nprobe in enumerate(nprobes):
        plt.annotate(f'nprobe={nprobe}', (recalls[i], throughputs[i]))
    
    plt.savefig('throughput_recall.png',format='svg', bbox_inches='tight')
    plt.show()
    
    # 关闭连接
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
