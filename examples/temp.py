import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

# 读取数据
def read_sensitivity_data(file_path):
    """
    读取敏感性分析数据
    假设txt文件是按列存储，列之间用空格或制表符分隔
    """
    try:
        # 尝试用pandas读取
        data = pd.read_csv(file_path, sep=None, engine='python', header=None)
        print(f"数据形状: {data.shape}")
        print("前5行数据:")
        print(data.head())
        return data.values
    except Exception as e:
        print(f"pandas读取失败，尝试numpy读取: {e}")
        # 备用方案：用numpy读取
        data = np.loadtxt(file_path)
        print(f"数据形状: {data.shape}")
        return data

# 请替换为你的文件路径
file_path = "indices.txt"  # 替换为实际文件路径

# 读取数据
try:
    data = read_sensitivity_data(file_path)
    
    # 检查数据维度
    if data.shape[1] != 4:
        print(f"警告: 检测到 {data.shape[1]} 列，预期4列")
        print("请确认数据格式是否正确")
    
    # 敏感性分析方法名称（按你提供的顺序）
    methods = ['Sobol\'', 'FAST', 'RBD-FAST', 'MARS-SA']
    
    # 生成参数名称
    n_params = data.shape[0]
    parameters = [f'P{i+1}' for i in range(n_params)]
    
    print(f"参数数量: {n_params}")
    print(f"方法数量: {len(methods)}")
    
except FileNotFoundError:
    print(f"文件未找到: {file_path}")
    print("请检查文件路径是否正确")
    # 生成示例数据用于演示
    np.random.seed(42)
    n_params = 21
    data = np.random.rand(n_params, 4) * 0.8 + 0.1
    methods = ['Sobol\'', 'FAST', 'RBD-FAST', 'MARS-SA']
    parameters = [f'P{i+1}' for i in range(n_params)]
    print("使用示例数据进行演示...")

def plot_grouped_sensitivity_analysis(data, parameters, methods, split_rows=2):
    """
    Plot grouped bar chart for sensitivity analysis results
    
    Parameters:
    -----------
    data : array-like
        Sensitivity indices data (n_params × n_methods)
    parameters : list
        Parameter names
    methods : list
        Sensitivity analysis method names
    split_rows : int
        Number of rows to split the display
    """
    # Calculate parameters per row
    n_params = len(parameters)
    params_per_row = n_params // split_rows
    if n_params % split_rows != 0:
        params_per_row += 1
    
    # Create subplots
    fig, axes = plt.subplots(split_rows, 1, figsize=(16, 5*split_rows))
    if split_rows == 1:
        axes = [axes]
    
    # Color scheme for publication quality
    colors = ['#5EABD6', '#03A6A1', '#A7C1A8', '#E14434']
    width = 0.21
    
    for row in range(split_rows):
        ax = axes[row]

        start_idx = row * params_per_row
        end_idx = min((row + 1) * params_per_row, n_params)
        
        current_params = parameters[start_idx:end_idx]
        current_data = data[start_idx:end_idx]
        
        x = np.arange(len(current_params))
        
        for i, method in enumerate(methods):
            bars = ax.bar(x + i*width, current_data[:, i], width, 
                         label=method, color=colors[i],
                         edgecolor='black', linewidth=1.5)
            
        # 添加参数间的竖线分隔
        for i in range(1, len(current_params)):
            ax.axvline(x=i-0.20, color='black', linestyle='--', linewidth=2)
            
        ax.set_xlabel(f'Parameters', 
                     fontsize=25, fontweight='bold')
        ax.set_ylabel('Sensitivity Indices', fontsize=25, fontweight='bold')
        ax.set_yticklabels(np.round(np.arange(0, 0.9, 0.1), 2), fontsize=20, fontweight='bold')
        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(current_params, fontsize=20, fontweight='bold')
        ax.set_xlim(-0.3, 20.5)
        from matplotlib.font_manager import FontProperties
        ax.legend(loc='upper right', frameon=True, shadow=False, 
                 fancybox=True, prop=FontProperties(weight='bold', size = 20))
        # ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_ylim(0, 0.8)
        
        # 设置边框加粗为2
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        # 设置刻度朝内，去掉横坐标刻度
        ax.tick_params(axis='x', direction='in', length=0, width=0)  # 横坐标刻度长度为0
        ax.tick_params(axis='y', direction='in', length=6, width=1.5)  # 纵坐标刻度朝内
         
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.4)
    plt.show()
    
    return fig



# 绘制图表
fig = plot_grouped_sensitivity_analysis(data, parameters, methods, split_rows=1)