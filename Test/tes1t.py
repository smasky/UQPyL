import numpy as np

# 假设我们有一个 2x4 的 base_sequence
base_sequence = np.array([[1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9, 10]])

# 我们还有一个 2x2 的 saltelli_sequence，初始值全为 0
saltelli_sequence = np.zeros((3, 3))

# 现在我们想要用 base_sequence 的前两列替换 saltelli_sequence 的对角线元素，用 base_sequence 的后两列替换 saltelli_sequence 的非对角线元素
D = 3
index = 0
for i in range(1):  # 假设我们只处理 base_sequence 的第一行
    for k in range(D):
        for j in range(D):
            if k == j:
                saltelli_sequence[index, j] = base_sequence[i, j]
            else:
                saltelli_sequence[index, j] = base_sequence[i, D + j]
        index += 1

print(saltelli_sequence)

print(saltelli_sequence)