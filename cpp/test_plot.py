import json
import matplotlib.pyplot as plt
import os
import argparse

def plot_results(flat_file, jl_file, sjlt_file, output_dir='plots'):
    """从JSON文件读取数据并绘制吞吐量和召回率曲线图"""
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取JSON数据
    with open(flat_file, 'r') as f:
        flat_data = json.load(f)
    
    with open(jl_file, 'r') as f:
        jl_data = json.load(f)
    
    with open(sjlt_file, 'r') as f:
        sjlt_data = json.load(f)
    
    # with open(pq_file, 'r') as f:
    #     pq_data = json.load(f)

    # 提取数据
    nprobes = [item['nprob'] for item in flat_data]
    flat_throughput = [item['throughput'] for item in flat_data]
    jl_throughput = [item['throughput'] for item in jl_data]
    sjlt_throughput = [item['throughput'] for item in sjlt_data]
    # pq_throughput = [item['throughput'] for item in pq_data]
    flat_recall = [item['recall'] for item in flat_data]
    jl_recall = [item['recall'] for item in jl_data]
    sjlt_recall = [item['recall'] for item in sjlt_data]
    # pq_recall = [item['recall'] for item in pq_data]
    
    # 绘制吞吐量曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(nprobes, flat_throughput, 'o-', label='Parallel Flat')
    plt.plot(nprobes, jl_throughput, 's-', label='Parallel JL')
    plt.plot(nprobes, sjlt_throughput, '^-', label='Parallel SJLT')
    # plt.plot(nprobes, pq_throughput, 'd-', label='Parallel PQ')
    plt.xlabel('nprobe')
    plt.ylabel('Throughput (qps)')
    plt.title('nprobe vs Throughput')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/throughput.png")
    plt.close()
    
    # 绘制召回率曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(nprobes, flat_recall, 'o-', label='Parallel Flat')
    plt.plot(nprobes, jl_recall, 's-', label='Parallel JL')
    plt.plot(nprobes, sjlt_recall, '^-', label='Parallel SJLT')
    # plt.plot(nprobes, pq_recall, 'd-', label='Parallel PQ')
    plt.xlabel('nprobe')
    plt.ylabel('Recall')
    plt.title('nprobe vs Recall')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/recall.png")
    plt.close()
    
    # 绘制吞吐量-召回率曲线
    plot_throughput_recall(
        [flat_recall, jl_recall, sjlt_recall],
        [flat_throughput, jl_throughput, sjlt_throughput],
        ['Parallel Flat', 'Parallel JL', 'Parallel SJLT'],
        ['o-', 's-', '^-', 'd-'],
        output_dir
    )
    
    print(f"图表已保存到 {output_dir} 目录")

def plot_throughput_recall(recall_list, throughput_list, labels, markers, output_dir):
    """绘制吞吐量-召回率曲线"""
    plt.figure(figsize=(10, 6))
    
    for recall, throughput, label, marker in zip(recall_list, throughput_list, labels, markers):
        plt.plot(recall, throughput, marker, label=label)
        
        # 为每个数据点添加nprobe标签
        for r, t, n in zip(recall, throughput, [1, 5, 10, 20]):
            plt.annotate(f'n={n}', (r, t), textcoords="offset points", 
                         xytext=(0,10), ha='center')
    
    plt.xlabel('Recall')
    plt.ylabel('Throughput (qps)')
    plt.title('Recall vs Throughput')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/throughput_recall.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='绘制KNN性能对比图')
    parser.add_argument('--flat_results', type=str, default='flat_results.json', help='Parallel Flat结果JSON文件')
    parser.add_argument('--jl_results', type=str, default='jl_results.json', help='Parallel JL结果JSON文件')
    parser.add_argument('--sjlt_results', type=str, default='sjlt_results.json', help='Parallel SJLT结果JSON文件')
    # parser.add_argument('--pq_results', type=str, default='pq_results.json', help='Parallel PQ结果JSON文件')
    parser.add_argument('--output', type=str, default='plots', help='图表输出目录')
    args = parser.parse_args()
    
    # 检查文件是否存在
    for file_path in [args.flat_results, args.jl_results, args.sjlt_results]:
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在")
            return
    
    # 绘制结果图
    plot_results(args.flat_results, args.jl_results, args.sjlt_results, args.output)

if __name__ == "__main__":
    main()