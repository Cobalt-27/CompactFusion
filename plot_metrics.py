import re
import matplotlib.pyplot as plt
import numpy as np

# 定义正则表达式模式来提取数据
# Match this: err: 79.794, total_err: 161.547, act=1981.917, delta=137.318, d/a=0.07, dd=117.284, dd/d=0.85
pattern = r'err: ([\d.]+), total_err: ([\d.]+), act=([\d.]+), delta=([\d.]+), d/a=([\d.]+), dd=([\d.]+), dd/d=([\d.]+)'
# pattern = r'err: ([\d.]+), act=([\d.]+), delta=([\d.]+), d/a=([\d.]+)'
sim_pattern = r'act_sim: ([\d.]+), delta_sim: ([\d.]+)'

# 存储数据的列表
k_metrics = {'err': [], 'total_err': [], 'act': [], 'delta': [], 'd/a': [], 'dd': [], 'dd/d': [], 'act_sim': [], 'delta_sim': []}
v_metrics = {'err': [], 'total_err': [], 'act': [], 'delta': [], 'd/a': [], 'dd': [], 'dd/d': [], 'act_sim': [], 'delta_sim': []}

# 读取输出数据
with open('output.txt', 'r') as f:
    content = f.read()

# 分割成k和v的部分
k_sections = re.findall(r'🔵 \[\d+-\d+-k\].*?(?=🔵|$)', content, re.DOTALL)
v_sections = re.findall(r'🔵 \[\d+-\d+-v\].*?(?=🔵|$)', content, re.DOTALL)

# 处理k部分
for section in k_sections:
    match = re.search(pattern, section)
    sim_match = re.search(sim_pattern, section)
    if match:
        k_metrics['err'].append(float(match.group(1)))
        k_metrics['total_err'].append(float(match.group(2)))
        k_metrics['act'].append(float(match.group(3)))
        k_metrics['delta'].append(float(match.group(4)))
        k_metrics['d/a'].append(float(match.group(5)))
        k_metrics['dd'].append(float(match.group(6)))
        k_metrics['dd/d'].append(float(match.group(7)))
    if sim_match:
        k_metrics['act_sim'].append(float(sim_match.group(1)))
        k_metrics['delta_sim'].append(float(sim_match.group(2)))

# 处理v部分
for section in v_sections:
    match = re.search(pattern, section)
    sim_match = re.search(sim_pattern, section)
    if match:
        v_metrics['err'].append(float(match.group(1)))
        v_metrics['total_err'].append(float(match.group(2)))
        v_metrics['act'].append(float(match.group(3)))
        v_metrics['delta'].append(float(match.group(4)))
        v_metrics['d/a'].append(float(match.group(5)))
        v_metrics['dd'].append(float(match.group(6)))
        v_metrics['dd/d'].append(float(match.group(7)))
    if sim_match:
        v_metrics['act_sim'].append(float(sim_match.group(1)))
        v_metrics['delta_sim'].append(float(sim_match.group(2)))

# 创建图表
fig, axes = plt.subplots(5, 2, figsize=(15, 25))
fig.suptitle('Key and Value Metrics Analysis', fontsize=16)

metrics = ['err', 'total_err', 'act', 'delta', 'd/a', 'dd', 'dd/d', 'act_sim', 'delta_sim']
titles = ['Error', 'Total Error', 'Activation', 'Delta', 'Delta/Activation', 'Delta/Delta', 'Delta/Delta/Delta', 'Activation Similarity', 'Delta Similarity']

for i, (metric, title) in enumerate(zip(metrics, titles)):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    
    x = np.arange(len(k_metrics[metric]))
    ax.plot(x, k_metrics[metric], 'b-', label='Key', marker='o')
    ax.plot(x, v_metrics[metric], 'r-', label='Value', marker='s')
    
    ax.set_title(title)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Value')
    ax.grid(True)
    ax.legend()
    
    # 为相似度指标设置y轴范围
    if metric in ['act_sim', 'delta_sim']:
        ax.set_ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig('metrics_analysis.png')
plt.close() 