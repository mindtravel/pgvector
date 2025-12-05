-- ============================================
-- 向量搜索性能对比测试
-- 比较传统单个查询 vs 批量查询的性能
-- ============================================

CREATE EXTENSION IF NOT EXISTS vector;

\echo '=== 向量搜索性能对比测试 ==='

-- 检查扩展状态
\echo '检查 vector 扩展状态:'
SELECT 
    CASE 
        WHEN EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') 
        THEN 'vector extension loaded' 
        ELSE 'vector extension NOT loaded' 
    END as vector_status;

-- 清理并创建测试表
\echo '创建测试表...'
DROP TABLE IF EXISTS test_vectors_perf CASCADE;
CREATE TABLE test_vectors_perf (
    id SERIAL PRIMARY KEY,
    embedding vector(128),
    category text,
    description text
);

-- 插入测试数据（1000个向量，128维）
\echo '插入测试数据...'
INSERT INTO test_vectors_perf (id, embedding, category, description) 
SELECT 
    i,
    (SELECT array_agg(sin(i + j)) FROM generate_series(1, 128) j)::vector(128),
    CASE 
        WHEN i % 4 = 0 THEN 'A'
        WHEN i % 4 = 1 THEN 'B' 
        WHEN i % 4 = 2 THEN 'C'
        ELSE 'D'
    END as category,
    'test_vector_' || i as description
FROM generate_series(1, 1000) i;

-- 创建索引（使用余弦距离）
\echo '创建 IVFFlat 索引...'
CREATE INDEX test_vectors_perf_ivfflat_idx ON test_vectors_perf 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 50);

-- 分析表
ANALYZE test_vectors_perf;

-- 获取索引 OID
\echo '获取索引 OID:'
SELECT oid as index_oid, relname as index_name
FROM pg_class 
WHERE relname = 'test_vectors_perf_ivfflat_idx';

-- 定义查询向量（确保4个向量互不相同）
\echo '定义4个查询向量...'
\set query1 '(SELECT array_agg(sin(100 + j)) FROM generate_series(1, 128) j)::vector(128)'
\set query2 '(SELECT array_agg(cos(200 + j)) FROM generate_series(1, 128) j)::vector(128)'
\set query3 '(SELECT array_agg(sin(300 + j) + cos(300 + j)) FROM generate_series(1, 128) j)::vector(128)'
\set query4 '(SELECT array_agg(sin(400 + j) * cos(400 + j)) FROM generate_series(1, 128) j)::vector(128)'

-- ============================================
-- 第一部分：传统单个向量查询（4次查询，每次LIMIT 3）
-- ============================================
\echo ''
\echo '=== 第一部分：传统单个向量查询 ==='
\echo '执行4次单个向量查询，每次LIMIT 3...'

\timing on

\echo '查询1:'
-- EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT id, category, description,
       embedding <=> :query1 as distance
FROM test_vectors_perf
ORDER BY embedding <=> :query1
LIMIT 3;

\echo '查询2:'
-- EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT id, category, description,
       embedding <=> :query2 as distance
FROM test_vectors_perf
ORDER BY embedding <=> :query2
LIMIT 3;

\echo '查询3:'
-- EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT id, category, description,
       embedding <=> :query3 as distance
FROM test_vectors_perf
ORDER BY embedding <=> :query3
LIMIT 3;

\echo '查询4:'
-- EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT id, category, description,
       embedding <=> :query4 as distance
FROM test_vectors_perf
ORDER BY embedding <=> :query4
LIMIT 3;

-- ============================================
-- 第二部分：批量向量搜索（1次查询，包含4个向量，LIMIT 3）
-- ============================================
\echo ''
\echo '=== 第二部分：批量向量搜索 ==='

\echo '批量查询结果:'
-- EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM batch_vector_search(
    (SELECT oid FROM pg_class WHERE relname = 'test_vectors_perf_ivfflat_idx'),
    ARRAY[:query1, :query2, :query3, :query4],
    3
);

-- EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM batch_vector_search(
    (SELECT oid FROM pg_class WHERE relname = 'test_vectors_perf_ivfflat_idx'),
    ARRAY[:query1, :query2, :query3, :query4, :query1, :query2, :query3, :query4, :query1, :query2, :query3, :query4],
    3
);
\timing off


-- ============================================
-- 测试完成
-- ============================================
\echo ''
\echo '=== 性能对比测试完成 ==='

