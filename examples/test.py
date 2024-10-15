import sys
import os
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from UQPyL.DoE import Sobol_Sequence, FAST_Sequence, Morris_Sequence, Saltelli_Sequence, LHS
from UQPyL.sensibility import Sobol, FAST, Morris, RBD_FAST, RSA

from UQPyL.problems.single_objective import Sphere

from SALib.analyze.rsa import rsa, analyze

problem=Sphere(nInput=5)

sampler=Saltelli_Sequence(scramble=True, skipValue=4, calSecondOrder=True)

sampler=LHS()

x=sampler.sample(1000, problem.nInput, problem)

y=problem.evaluate(x)


sobol=RSA(nRegion=10, verbose=True)

problems ={
    'num_vars': problem.nInput,
    'names' : problem.x_labels
}

res=analyze(problems, np.copy(x), y.ravel(), bins=10)
res.plot()
plt.show()
res2=sobol.analyze(problem, x, y)
# s1=res.Si['S1(First Order)'][1]

# b=np.sum(abs(s1))

# import numpy as np
# import matplotlib.pyplot as plt

# # 假设 sensitivity_matrix 是你获得的敏感性矩阵
# # 例如，3 个参数，5 个区间
# sensitivity_matrix = 1

# # 计算每个参数的平均敏感性和标准差
# mean_sensitivity = np.mean(sensitivity_matrix, axis=1)
# std_sensitivity = np.std(sensitivity_matrix, axis=1)

# # 绘制条形图
# plt.figure(figsize=(8, 6))
# x_labels = [f'Parameter {i+1}' for i in range(sensitivity_matrix.shape[0])]
# plt.bar(x_labels, mean_sensitivity, yerr=std_sensitivity, capsize=5, color='skyblue')
# plt.xlabel('Parameters')
# plt.ylabel('Average Sensitivity')
# plt.title('Parameter Sensitivity Comparison')
# plt.show()

# # 绘制箱线图
# plt.figure(figsize=(8, 6))
# plt.boxplot(sensitivity_matrix.T, labels=x_labels)
# plt.xlabel('Parameters')
# plt.ylabel('Sensitivity')
# plt.title('Sensitivity Distribution Across Intervals')
# plt.show()
