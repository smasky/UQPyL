'''
    Example of using optimization algorithms:
    - 1. Genetic Algorithm (GA) - Single objective
    - 2. SCE-UA - Single objective
    - 3. PSO - Single objective
    - 4. ASMO - Single-objective - Surrogate
    - 5. NSGA-II - Multi-objective
    - 6. MOEAD/D - Multi-objective
    - 7. MO_ASMO - Multi-objective - Surrogate
'''
#--------------tmp-------------------#
import sys
sys.path.append(".")
#-----------tmp--------------------#
import numpy as np
import matplotlib.pyplot as plt
from UQPyL.problems import Sphere
################1. Genetic Algorithm (GA) - Single objective################
print('################1. Genetic Algorithm (GA) - Single objective################')
from UQPyL.optimization import GA, PSO, SCE_UA, ML_SCE_UA, CSA
from UQPyL.problems import Sphere, Rastrigin, Rosenbrock

problem=Rastrigin(n_input=30, ub=100, lb=-100)
sce=SCE_UA(ngs=5, verbose=True, maxFEs=50000, maxIterTimes=1000)
pso=GA(nInit=50, nPop=50, maxFEs=50000, maxIterTimes=1000)
csa=CSA(verboseFreq=10)
res=pso.run(problem)

# import matplotlib.pyplot as plt
# FEs_objs=res['history_best_objs']
# FEs=list(FEs_objs.keys())
# objs=list(FEs_objs.values())
# plt.plot(FEs, np.log10(objs))

################2. SCE-UA - Single objective################
# print('################2. SCE-UA - Single objective################')
# from UQPyL.optimization import SCE_UA
# problem=Sphere(n_input=30, ub=100, lb=-100)
# sce=SCE_UA(problem)
# res=sce.run()
# print('Best objective:', res['best_obj'])
# print('Best decisions:', res['best_dec'])
# print('FE:', res['FEs'])

# objs=res['history_best_objs']
# FEs_objs=res['history_best_objs']
# FEs=list(FEs_objs.keys())
# objs=list(FEs_objs.values())
# plt.plot(FEs, objs)
# plt.show()
###############3. PSO - Single objective################
# print('###############3. PSO - Single objective################')
# from UQPyL.optimization import PSO
# from UQPyL.problems import Sphere
# problem=Sphere(n_input=30, ub=100, lb=-100)
# pso=PSO(problem, n_sample=50, w=0.5, c1=1.5, c2=1.5)
# res=pso.run()
# print('Best objective:', res['best_obj'])
# print('Best decisions:', res['best_decs'])
# print('FE:', res['FEs'])

# objs=res['history_best_objs']
# FEs_objs=res['history_best_objs']
# FEs=list(FEs_objs.keys())
# objs=list(FEs_objs.values())
# plt.plot(FEs, np.log10(objs))

# plt.title('Best Objective Over Iterations')
# plt.xlabel('Function Evbaluations')
# plt.ylabel('Best Objective (log10)')
# plt.show()
################3. ASMO - Single-objective - Surrogate-assisted################
# print('################3. ASMO - Single-objective################')
# from UQPyL.optimization import ASMO
# from UQPyL.surrogates import RBF
# from UQPyL.surrogates.rbf_kernels import Cubic
# from UQPyL.problems import Sphere
# problem=Sphere(n_input=30, ub=100, lb=-100)
# rbf=RBF(kernel=Cubic())
# asmo=ASMO(problem, rbf, n_init=50)
# res=asmo.run()
# print('Best objective:', res['best_obj'])
# print('Best decisions:', res['best_dec'])
# print('FE:', res['FEs'])

#############4. NSGA-II - Multi-objective#################
# print('#############4. NSGA-II - Multi-objective#################')
# from UQPyL.optimization import NSGAII
# from UQPyL.problems import ZDT3
# problem=ZDT3(n_input=30)
# optimum=problem.get_optimum(100)
# nsga=NSGAII(problem, maxFEs=10000, maxIters=1000, n_samples=50)
# res=nsga.run()
# import matplotlib.pyplot as plt
# y=res['pareto_y']
# plt.scatter(y[:,0], y[:,1])
# plt.show()

#############5. MOEAD - Multi-objective#################
# print('#############5. MOEAD - Multi-objective#################')
# from UQPyL.optimization import MOEA_D
# from UQPyL.problems import ZDT6
# problem=ZDT6(n_input=30)
# optimum=problem.get_PF()
# moead=MOEA_D(problem, aggregation_type='TCH_M',maxFEs=50000, maxIters=1000)
# _, res=moead.run()
# import matplotlib.pyplot as plt
# plt.scatter(res[:,0], res[:,1])
# plt.plot(optimum[:,0], optimum[:,1], 'r')
# plt.show()

#############5. MO_ASMO - Multi-objective - Surrogate-assisted#################
# from UQPyL.optimization import MOASMO
# from UQPyL.problems import ZDT1
# from UQPyL.surrogates import RBF, KRG
# from UQPyL.surrogates import Mo_Surrogates
# from UQPyL.surrogates.rbf_kernels import Cubic
# from UQPyL.surrogates.krg_kernels import Guass_Kernel
# import matplotlib.pyplot as plt

# rbf1=RBF(kernel=Cubic())
# rbf2=RBF(kernel=Cubic())
# guass1=Guass_Kernel(theta=1e-3, theta_lb=1e-5, theta_ub=1, heterogeneous=True)
# guass2=Guass_Kernel(theta=1e-3, theta_lb=1e-5, theta_ub=1, heterogeneous=True)
# krg1=KRG(kernel=guass1)
# krg2=KRG(kernel=guass2)
# mo_surr=Mo_Surrogates(2,[krg1, krg2])
# problem=ZDT1(n_input=30)
# mo_amso=MOASMO(problem=problem, surrogates=mo_surr, maxFEs=300, advance_infilling=False)
# res=mo_amso.run()
# y=res[1]
# fig, axs = plt.subplots(1, 2) 
# axs[0].scatter(y[:,0], y[:,1], color='b')
# axs[0].set_title('Advanced')
# rbf1=RBF(kernel=Cubic())
# rbf2=RBF(kernel=Cubic())
# mo_surr=Mo_Surrogates(2,[rbf1, rbf2])
# problem=ZDT1(n_input=30)
# mo_amso=MOASMO(problem=problem, surrogates=mo_surr, maxFEs=300, advance_infilling=False)
# res=mo_amso.run()
# yy=res[1]
# axs[1].scatter(yy[:,0], yy[:,1], color='r')
# axs[1].set_title('Origin')
# plt.show()