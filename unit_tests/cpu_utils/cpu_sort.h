// ============================================================================
// CPU Reference Implementations
// ============================================================================

/**
 * CPU 参考实现：标准排序
 */
template<typename T, typename IdxT>
void cpu_sort(
    const T* input,
    T* output_vals,
    IdxT* output_idx,
    int n,
    bool ascending)
{
    std::vector<std::pair<T, IdxT>> pairs;
    pairs.reserve(n);
    
    for (int i = 0; i < n; i++) {
        pairs.push_back({input[i], static_cast<IdxT>(i)});
    }
    
    if (ascending) {
        std::sort(pairs.begin(), pairs.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
    } else {
        std::sort(pairs.begin(), pairs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
    }
    
    for (int i = 0; i < n; i++) {
        output_vals[i] = pairs[i].first;
        output_idx[i] = pairs[i].second;
    }
}

 /**
 * CPU 参考实现：选择 top-k
 */
void cpu_select_k(
    const float** input,
    int batch_size,
    int len,
    int k,
    float** output_vals,
    int** output_idx,
    bool select_min
) {
    for (int b = 0; b < batch_size; ++b) {
        std::vector<std::pair<float, int>> candidates;
        for (int i = 0; i < len; ++i) {
            float val = input[b][i];
            if (val < FLT_MAX && val == val) {  // 排除 INF 和 NaN
                candidates.push_back({val, i});
            }
        }
        
        if (select_min) {
            std::sort(candidates.begin(), candidates.end());
        } else {
            std::sort(candidates.begin(), candidates.end(), std::greater<std::pair<float, int>>());
        }
        
        int n_select = std::min(k, (int)candidates.size());
        for (int i = 0; i < n_select; ++i) {
            output_vals[b][i] = candidates[i].first;
            output_idx[b][i] = candidates[i].second;
        }
        // 填充剩余位置为 INF 和 -1
        for (int i = n_select; i < k; ++i) {
            output_vals[b][i] = FLT_MAX;
            output_idx[b][i] = -1;
        }
    }
}
