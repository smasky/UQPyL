import sys
sys.path.append('..')

import os
uqPath=os.path.dirname("../UQPyL/")
sys.path.insert(0, uqPath)

# from UQPyL.problems import Sphere, Weierstrass, RosenbrockWithCon, Rosenbrock

# problem=Sphere(nInput=10)

# problem1=RosenbrockWithCon(nInput=10)
# problem2=Rosenbrock(nInput=10)
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

#------------GA---------------#
# from UQPyL.optimization.single_objective import GA
# ga=GA()
# res=ga.run(problem)

#------------ABC---------------#
# from UQPyL.optimization.single_objective import ABC
# abc=ABC(saveFlag=True)
# res=abc.run(problem)

#------------CSA---------------#
# from UQPyL.optimization.single_objective import CSA
# csa=CSA()
# res=csa.run(problem)

#-------------DE---------------#
# from UQPyL.optimization.single_objective import DE
# de=DE()
# res=de.run(problem)

#--------------PSO----------------#
# from UQPyL.optimization.single_objective import PSO
# pso=PSO()
# res=pso.run(problem)

#-----------------SCE-UA-------------------#
# from UQPyL.optimization.single_objective import SCE_UA
# sce_ua=SCE_UA()
# res=sce_ua.run(problem)

#----------------ML-SCE-UA------------------#
# from UQPyL.optimization.single_objective import ML_SCE_UA
# ml_sce_ua=ML_SCE_UA()
# res=ml_sce_ua.run(problem)

#Multi-objective Optimization
# from UQPyL.problems import DTLZ2, DTLZ5
# dtlz5=DTLZ5(nInput = 15)
# dtlz2=DTLZ2(nInput = 15)
#-------------------RVEA----------------------#
# from UQPyL.optimization.multi_objective import RVEA
# rvea=RVEA(nPop = 50, maxFEs = 100)
# res=rvea.run(dtlz2)

#---------------------NSGAII-------------------#
# from UQPyL.optimization.multi_objective import NSGAII
# nsgaii=NSGAII(maxFEs=100)
# res=nsgaii.run(dtlz2)

#------------------NSGAIII--------------------#
from UQPyL.optimization.multi_objective import NSGAIII
nsgaiii=NSGAIII(maxFEs = 10000)
res=nsgaiii.run(dtlz5)

#------------------MOEAD--------------------------#
# from UQPyL.optimization.multi_objective import MOEAD
# moead=MOEAD()
# res=moead.run(dtlz2)

# Y=res.bestObj

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# # 创建数据
# x = Y[:, 0]
# y = Y[:, 1]
# z = Y[:, 2]

# # 创建图形和轴
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 添加数据到轴上
# ax.scatter(x, y, z)

# # 添加标签
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# # 显示图形
# plt.show()