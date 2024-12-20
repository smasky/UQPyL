import sys
sys.path.append('.')

import os
uqPath=os.path.dirname("../UQPyL/")
sys.path.insert(0, uqPath)

# from UQPyL.problems import Sphere, Weierstrass

# problem=Sphere(nInput=10)


#-------------ASMO--------------#

# from UQPyL.optimization.single_objective import ASMO
# from UQPyL.surrogates.rbf import RBF
# from UQPyL.optimization.single_objective import GA
# asmo=ASMO(nInit=200, surrogate=RBF(), optimizer=GA(maxFEs=10000), maxFEs=500)
# res=asmo.run(problem)

#-----------EGO---------------#
# from UQPyL.optimization.single_objective import EGO
# from UQPyL.optimization.single_objective import ASMO
# ego=EGO(nInit=50)
# res=ego.run(problem)

#-----------GA---------------#
# from UQPyL.optimization.single_objective import GA
# ga=GA()
# res=ga.run(problem)

#-----------ABC---------------#
# from UQPyL.optimization.single_objective import ABC
# abc=ABC()
# res=abc.run(problem)

#-----------CSA---------------#
# from UQPyL.optimization.single_objective import CSA
# csa=CSA()
# res=csa.run(problem)

#-------------DE---------------#
# from UQPyL.optimization.single_objective import DE
# de=DE()
# res=de.run(problem)

#------------PSO----------------#
# from UQPyL.optimization.single_objective import PSO
# pso=PSO()
# res=pso.run(problem)

#--------------SCE-UA--------------#
# from UQPyL.optimization.single_objective import SCE_UA
# sce_ua=SCE_UA()
# res=sce_ua.run(problem)

#----------------ML-SCE-UA------------------#
# from UQPyL.optimization.single_objective import ML_SCE_UA
# ml_sce_ua=ML_SCE_UA()
# res=ml_sce_ua.run(problem)

#Multi-objective Optimization
from UQPyL.optimization.multi_objective import RVEA, NSGAII, NSGAIII, MOEAD, MOASMO
from UQPyL.problems.multi_objective import DTLZ2

dtlz1=DTLZ2(nInput=15)
rvea=MOASMO(maxFEs=1000, nInit=50)
res=rvea.run(dtlz1)

bestObjs=res.bestObj
# from UQPyL.optimization.multi_objective import MOEAD, RVEA, NSGAIII, NSGAII
# from UQPyL.problems.multi_objective import DTLZ6

x=bestObjs[:, 0]
y=bestObjs[:, 1]
z=bestObjs[:, 2]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维散点图
scatter = ax.scatter(x, y, z, alpha=0.8)
# 显示图形
plt.show()