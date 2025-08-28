//=== cudatimer.h ===//
#include <iostream>
#include <string>
#include <map>
#include <chrono>
#include <cuda_runtime.h>

class CUDATimer {
private:
    std::string name_;
    bool enable_;
    bool use_cuda_event_;
    cudaEvent_t start_event_, stop_event_;
    std::chrono::high_resolution_clock::time_point start_cpu_, stop_cpu_;
    float gpu_time_ms_;

public:
    // 构造函数：传入计时器名称、是否启用、是否使用CUDA事件（默认为true）
    CUDATimer(const std::string& name, bool enable = true, bool use_cuda_event = true)
        : name_(name), enable_(enable), use_cuda_event_(use_cuda_event), gpu_time_ms_(0.0f) {
        if (!enable_) return;

        if (use_cuda_event_) {
            cudaEventCreate(&start_event_);
            cudaEventCreate(&stop_event_);
            cudaEventRecord(start_event_); // 记录开始事件
            cudaEventSynchronize(start_event_); // 等待事件记录完成，确保时间起点准确
        } else {
            start_cpu_ = std::chrono::high_resolution_clock::now();
        }
    }

    // 析构函数：自动记录结束时间并输出结果
    ~CUDATimer() {
        if (!enable_) return;

        if (use_cuda_event_) {
            cudaEventRecord(stop_event_);
            cudaEventSynchronize(stop_event_); // 等待GPU操作完成
            cudaEventElapsedTime(&gpu_time_ms_, start_event_, stop_event_);
            std::cout << "[CUDA Event Timer] " << name_ << " took: " << gpu_time_ms_ << " ms" << std::endl;
            cudaEventDestroy(start_event_);
            cudaEventDestroy(stop_event_);
        } else {
            stop_cpu_ = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_cpu_ - start_cpu_);
            std::cout << "[CPU Timer] " << name_ << " took: " << duration.count() << " ms" << std::endl;
        }
    }

    // 获取耗时（毫秒），适用于需要获取返回值的情况
    float getElapsedMilliseconds() const {
        if (!enable_) return 0.0f;
        if (use_cuda_event_) {
            return gpu_time_ms_;
        } else {
            auto current = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current - start_cpu_);
            return static_cast<float>(duration.count());
        }
    }

    // 静态成员，控制所有计时器的全局开关
    static bool global_enable;
};

// 静态成员初始化
bool CUDATimer::global_enable = true;