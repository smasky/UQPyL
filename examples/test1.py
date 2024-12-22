import numpy as np

# 定义 list 和 linspace
my_list = ['A', 'B', 'C', 'D']
linspace = [0]

# 定义区间边界
bins = [0, 0.25, 0.5, 0.75, 1]

# 使用 np.digitize，将值映射到区间
indices = np.digitize(linspace, bins, right=True) - 1

# 获取对应的 list 值
mapped_values = [my_list[i] for i in indices]

print(mapped_values)