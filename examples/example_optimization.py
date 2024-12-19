import sys
sys.path.append('.')

import os
uqPath=os.path.dirname("../UQPyL/")
sys.path.insert(0, uqPath)

from UQPyL.problems import Sphere, Weierstrass

problem=Sphere(nInput=10)


#-------------ASMO--------------#

# from UQPyL.optimization.single_objective import ASMO
# from UQPyL.surrogates.rbf import RBF
# from UQPyL.optimization.single_objective import GA
# asmo=ASMO(nInit=200, surrogate=RBF(), optimizer=GA(maxFEs=10000), maxFEs=500)
# res=asmo.run(problem)

#-----------EGO---------------#
# from UQPyL.optimization.single_objective import EGO
# from UQPyL.optimization.single_objective import ASMO
# ego=ASMO(nInit=50)
# res=ego.run(problem)

#-----------GA---------------#
# from UQPyL.optimization.single_objective import GA
# ga=GA(saveFlag=True)
# res=ga.run(problem)


from UQPyL.optimization.multi_objective import MOEAD, RVEA, NSGAIII, NSGAII
from UQPyL.problems.multi_objective import DTLZ6

dtlz1=DTLZ6(nInput=15)
moead=NSGAII(maxFEs=10000, nInit=100, nPop=100)
res=moead.run(dtlz1)

Y=res.bestObj

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 创建数据
x = Y[:, 0]
y = Y[:, 1]
z = Y[:, 2]

# 创建图形和轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 添加数据到轴上
ax.scatter(x, y, z)

# 添加标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 显示图形
plt.show()