import subprocess
import matplotlib.pyplot as plt
import json
import os

def run_test(script_name, max_parallel_workers):
    """运行测试脚本并返回结果"""
    cmd = [
        "python3",
        script_name,
        "--max_parallel_workers_per_gather",
        str(max_parallel_workers)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}: {result.stderr}")
        return None
    
    # 解析输出获取recall和throughput
    # 这里假设输出格式为JSON
    try:
        output = json.loads(result.stdout)
        return output
    except json.JSONDecodeError:
        print(f"Error parsing output from {script_name}")
        return None

def plot_results(results, title, filename):
    """绘制结果图表"""
    plt.figure(figsize=(10, 6))
    
    for workers, data in results.items():
        recalls = [d['recall'] for d in data]
        throughputs = [d['throughput'] for d in data]
        plt.plot(recalls, throughputs, marker='o', label=f'{workers} workers')
    
    plt.xlabel('Recall')
    plt.ylabel('Throughput (queries/second)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def main():
    # 测试参数
    parallel_workers = [4, 8, 12, 16]
    scripts = ['test_ivfflat.py', 'test_ivfjl.py']
    
    # 存储结果
    results = {
        'ivfflat': {},
        'ivfjl': {}
    }
    
    # 运行测试
    for workers in parallel_workers:
        print(f"\nTesting with {workers} parallel workers")
        
        for script in scripts:
            print(f"\nRunning {script}...")
            result = run_test(script, workers)
            if result:
                if 'ivfflat' in script:
                    results['ivfflat'][workers] = result
                else:
                    results['ivfjl'][workers] = result
    
    # 绘制结果
    plot_results(results['ivfflat'], 'IVFFlat Performance with Different Parallel Workers', 'ivfflat_parallel_performance.png')
    plot_results(results['ivfjl'], 'IVFJL Performance with Different Parallel Workers', 'ivfjl_parallel_performance.png')

if __name__ == "__main__":
    main() 