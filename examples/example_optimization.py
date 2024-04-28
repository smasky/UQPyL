'''
    Example of using optimization algorithms:
    - 1. Genetic Algorithm (GA) - Single objective
    - 2. SCE-UA - Single objective
    - 3. ASMO - Single-objective
    - 4. NSGA-II - Multi-objective
    - 5. MO_ASMO - Multi-objective
'''
#tmp
import sys
sys.path.append(".")
from scipy.io import loadmat
print(sys.path)
import os
os.chdir('./examples')
#
import numpy as np

################1. Genetic Algorithm (GA) - Single objective################
# print('################1. Genetic Algorithm (GA) - Single objective################')
# from UQPyL.optimization import GA
# from UQPyL.problems import Sphere

# problem=Sphere(n_input=30, ub=100, lb=-100)
# ga=GA(problem, n_samples=50)
# res=ga.run()
# print('Best objective:', res['best_obj'])
# print('Best decisions:', res['best_dec'])
# print('FE:', res['FEs'])

################2. SCE-UA - Single objective################
# print('################2. SCE-UA - Single objective################')
# from UQPyL.optimization import SCE_UA
# problem=Sphere(n_input=30, ub=100, lb=-100)
# sce=SCE_UA(problem)
# res=sce.run()
# print('Best objective:', res['best_obj'])
# print('Best decisions:', res['best_dec'])
# print('FE:', res['FEs'])


################3. ASMO - Single-objective################
# print('################3. ASMO - Single-objective################')
# from UQPyL.optimization import ASMO
# from UQPyL.surrogates import RBF
# from UQPyL.surrogates.rbf_kernels import Cubic
# problem=Sphere(n_input=30, ub=100, lb=-100)
# rbf=RBF(kernel=Cubic())
# asmo=ASMO(problem, rbf, n_init=50)
# res=asmo.run(maxFE=1000)
# print('Best objective:', res['best_obj'])
# print('Best decisions:', res['best_dec'])
# print('FE:', res['FEs'])

#############4. NSGA-II - Multi-objective#################
print('#############4. NSGA-II - Multi-objective#################')
from UQPyL.optimization import NSGAII
from UQPyL.problems import ZDT1
problem=ZDT1(n_input=30)
optimum=problem.get_optimum(100)
nsga=NSGAII(problem, maxFEs=10000, maxIters=1000, n_samples=50)
res=nsga.run()
import matplotlib.pyplot as plt
y=res[1]
# plt.scatter(y[:,0], y[:,1])
# plt.show()

#############5. MO_ASMO - Multi-objective#################
from UQPyL.optimization import MOASMO
from UQPyL.problems import ZDT1
from UQPyL.surrogates import RBF
from UQPyL.surrogates import MO_Surrogates
from UQPyL.surrogates.rbf_kernels import Cubic

rbf1=RBF(kernel=Cubic())
rbf2=RBF(kernel=Cubic())
mo_surr=MO_Surrogates(2,[rbf1, rbf2])
problem=ZDT1(n_input=30)
mo_amso=MOASMO(problem=problem, surrogates=mo_surr, maxFEs=300)
res=mo_amso.run()
y=res[1]
plt.scatter(y[:,0], y[:,1])
plt.show()