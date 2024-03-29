import sys
import numpy as np
sys.path.append(".")
from UQPyL.Problems import Sphere, ZDT1
from UQPyL.Experiment_Design import LHS
from UQPyL.Optimization import SCE_UA, ASMO, NSGAII
from UQPyL.Surrogates import RBF
from UQPyL.Surrogates.RBF_Kernel import Cubic
import matplotlib.pyplot as plt
import os
os.chdir('./Test')

##SCE_UA
# lhs=LHS('center')
# Problems=Sphere(dim=10, ub=100, lb=-100)
# Algorithm=SCE_UA(Problems.evaluate, Problems.dim, Problems.lb, Problems.ub)
# Algorithm.run()

##ASMO
# rbf=RBF(kernel=Cubic())
# lhs=LHS('center')
# problem=Sphere(dim=10, ub=100, lb=-100)
# algorithm=ASMO(problem.evaluate, rbf, problem.lb, problem.ub, problem.dim)
# algorithm.run(maxFE=100)

##NSGAII
# problem=ZDT1(15,2)
# algorithm=NSGAII(problem.evaluate, problem.dim, problem.NOutput, problem.lb, problem.ub, NInit=500)
# Xpop,Ypop=algorithm.run()
# plt.scatter(Ypop[:,0],Ypop[:,1])
# plt.xlim((0,1))
# plt.ylim((0,1))
# plt.show()

##