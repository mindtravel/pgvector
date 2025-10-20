#ifndef METRIC_COLLECTOR_H
#define METRIC_COLLECTOR_H
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

// ============================================================================
// Metrics 收集器 (Metrics Collector)
// ============================================================================

/**
 * Metrics 收集器类
 * 用于在参数扫描过程中收集和展示性能指标
 * 
 * 使用示例：
 * MetricsCollector metrics;
 * metrics.set_columns("batch", "k", "gpu_ms", "cpu_ms", "speedup");
 * 
 * PARAM_2D(batch, (100, 1000), k, (16, 64)) {
 *     test_function(batch, k, &metrics);  // 测试函数内部调用 metrics.add_row()
 * }
 * 
 * metrics.print_table();
 */
class MetricsCollector {
private:
    std::vector<std::string> column_names_;
    std::vector<std::vector<double>> rows_;
    std::vector<int> column_widths_;
    bool columns_set_ = false;
    int num_repeats_ = 10;        /* 默认重复 10 次 */
    int base_seed_ = 42;          /* 基础随机种子 */
    
    /* 计算每列的最大宽度 */
    void calculate_column_widths() {
        column_widths_.resize(column_names_.size());
        
        /* 初始化为列名长度 */
        for (size_t i = 0; i < column_names_.size(); i++) {
            column_widths_[i] = column_names_[i].length();
        }
        
        /* 检查数据行 */
        for (const auto& row : rows_) {
            for (size_t i = 0; i < row.size() && i < column_widths_.size(); i++) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(3) << row[i];
                int width = oss.str().length();
                if (width > column_widths_[i]) {
                    column_widths_[i] = width;
                }
            }
        }
        
        /* 最小宽度为 6 */
        for (auto& w : column_widths_) {
            if (w < 6) w = 6;
        }
    }
    
    /* 打印分隔线 */
    void print_separator() const {
        std::cout << "+";
        for (size_t i = 0; i < column_widths_.size(); i++) {
            std::cout << std::string(column_widths_[i] + 2, '-') << "+";
        }
        std::cout << "\n";
    }
    
public:
    MetricsCollector() = default;
    
    /**
     * 设置列名（可变参数版本）
     */
    template<typename... Args>
    void set_columns(Args&&... names) {
        column_names_ = {std::forward<Args>(names)...};
        columns_set_ = true;
    }
    
    /**
     * 添加一行数据（可变参数版本）- 直接添加
     */
    template<typename... Args>
    void add_row(Args&&... values) {
        if (!columns_set_) {
            std::cerr << "Error: Must call set_columns() before add_row()\n";
            return;
        }
        
        std::vector<double> row;
        row.reserve(sizeof...(values));
        
        /* 使用折叠表达式将所有参数转为 double 并添加到 row */
        (row.push_back(static_cast<double>(values)), ...);
        
        if (row.size() != column_names_.size()) {
            std::cerr << "Error: Row has " << row.size() 
                      << " values but " << column_names_.size() 
                      << " columns expected\n";
            return;
        }
        
        rows_.push_back(std::move(row));
    }
    
    /**
     * 添加一行数据（重复运行取平均值）
     * 
     * 使用示例：
     * metrics.add_row_averaged([&]() -> std::vector<double> {
     *     // 运行测试，返回指标向量
     *     double gpu_ms = benchmark(...);
     *     return {batch, k, gpu_ms, speedup};
     * });
     * 
     * @param test_func 测试函数，返回 std::vector<double> 表示一行指标
     * @return 平均后的指标向量
     */
    template<typename Func>
    std::vector<double> add_row_averaged(Func test_func) {
        if (!columns_set_) {
            std::cerr << "Error: Must call set_columns() before add_row_averaged()\n";
            return {};
        }
        
        std::vector<double> sum_row(column_names_.size(), 0.0);
        int valid_runs = 0;
        
        for (int i = 0; i < num_repeats_; i++) {
            srand(base_seed_ + i);
            
            std::vector<double> row = test_func();
            
            if (row.size() != column_names_.size()) {
                std::cerr << "Warning: Row size mismatch in add_row_averaged (run " << i << ")\n";
                continue;
            }
            
            for (size_t j = 0; j < row.size(); j++) {
                sum_row[j] += row[j];
            }
            valid_runs++;
        }
        
        if (valid_runs == 0) {
            std::cerr << "Error: No valid runs in add_row_averaged\n";
            return sum_row;
        }
        
        for (auto& val : sum_row) {
            val /= valid_runs;
        }
        
        rows_.push_back(sum_row);
        
        return sum_row;
    }
    
    /**
     * 设置重复次数
     */
    void set_num_repeats(int n) {
        if (n > 0) {
            num_repeats_ = n;
        } else {
            std::cerr << "Warning: num_repeats must be positive, using default (10)\n";
        }
    }
    
    /**
     * 设置基础随机种子
     */
    void set_base_seed(int seed) {
        base_seed_ = seed;
    }
    
    /**
     * 获取当前重复次数设置
     */
    int get_num_repeats() const {
        return num_repeats_;
    }
    
    /**
     * 获取最后一行数据（用于从临时 metrics 中提取指标）
     */
    std::vector<double> get_last_row() const {
        if (rows_.empty()) {
            return {};
        }
        return rows_.back();
    }
    
    /**
     * 获取指定行的数据
     */
    std::vector<double> get_row(size_t idx) const {
        if (idx >= rows_.size()) {
            return {};
        }
        return rows_[idx];
    }
    
    /**
     * 打印表格
     */
    void print_table() {
        if (!columns_set_ || column_names_.empty()) {
            std::cout << "No metrics to display.\n";
            return;
        }

        COUT_ENDL("  Metrics: ");
        
        calculate_column_widths();
        
        /* 打印表头 */
        print_separator();
        std::cout << "|";
        for (size_t i = 0; i < column_names_.size(); i++) {
            std::cout << " " << std::left << std::setw(column_widths_[i]) 
                      << column_names_[i] << " |";
        }
        std::cout << "\n";
        print_separator();
        
        /* 打印数据行 */
        for (const auto& row : rows_) {
            std::cout << "|";
            for (size_t i = 0; i < row.size(); i++) {
                std::cout << " " << std::right << std::setw(column_widths_[i]) 
                          << std::fixed << std::setprecision(3) << row[i] << " |";
            }
            std::cout << "\n";
        }
        print_separator();
        
        /* 打印统计信息 */
        std::cout << "Total rows: " << rows_.size();
        if (num_repeats_ > 1) {
            std::cout << " (averaged over " << num_repeats_ << " runs)";
        }
        std::cout << "\n";
    }
    
    /**
     * 清空所有数据（保留列名）
     */
    void clear_rows() {
        rows_.clear();
    }
    
    /**
     * 重置（清空列名和数据）
     */
    void reset() {
        column_names_.clear();
        rows_.clear();
        column_widths_.clear();
        columns_set_ = false;
    }
    
    /**
     * 获取行数
     */
    size_t size() const {
        return rows_.size();
    }
    
    /**
     * 计算某列的平均值
     */
    double mean(size_t col_idx) const {
        if (col_idx >= column_names_.size() || rows_.empty()) {
            return 0.0;
        }
        
        double sum = 0.0;
        for (const auto& row : rows_) {
            if (col_idx < row.size()) {
                sum += row[col_idx];
            }
        }
        return sum / rows_.size();
    }
    
    /**
     * 计算某列的平均值（按列名）
     */
    double mean(const std::string& col_name) const {
        for (size_t i = 0; i < column_names_.size(); i++) {
            if (column_names_[i] == col_name) {
                return mean(i);
            }
        }
        std::cerr << "Error: Column '" << col_name << "' not found\n";
        return 0.0;
    }
    
    /**
     * 导出为 CSV 格式
     */
    void export_csv(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << "\n";
            return;
        }
        
        /* 写入表头 */
        for (size_t i = 0; i < column_names_.size(); i++) {
            file << column_names_[i];
            if (i < column_names_.size() - 1) file << ",";
        }
        file << "\n";
        
        /* 写入数据 */
        for (const auto& row : rows_) {
            for (size_t i = 0; i < row.size(); i++) {
                file << std::fixed << std::setprecision(6) << row[i];
                if (i < row.size() - 1) file << ",";
            }
            file << "\n";
        }
        
        file.close();
        std::cout << "Metrics exported to " << filename << "\n";
    }
};

#endif