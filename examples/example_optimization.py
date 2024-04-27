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


################1. Genetic Algorithm (GA) - Single objective################
print('################1. Genetic Algorithm (GA) - Single objective################')
from UQPyL.optimization import GA
from UQPyL.problems import Sphere

problem=Sphere(n_input=30, ub=100, lb=-100)
ga=GA(problem, n_samples=50)
res=ga.run()
print('Best objective:', res['best_obj'])
print('Best decisions:', res['best_decs'])
print('FE:', res['FEs'])

################2. SCE-UA - Single objective################
print('################2. SCE-UA - Single objective################')
from UQPyL.optimization import SCE_UA
problem=Sphere(n_input=30, ub=100, lb=-100)
sce=SCE_UA(problem)
res=sce.run()
print('Best objective:', res['best_obj'])
print('Best decisions:', res['best_decs'])
print('FE:', res['FEs'])


################3. ASMO - Single-objective################
print('################3. ASMO - Single-objective################')
from UQPyL.optimization import ASMO
from UQPyL.surrogates import RBF
from UQPyL.surrogates.rbf_kernels import Cubic
problem=Sphere(n_input=30, ub=100, lb=-100)
rbf=RBF(kernel=Cubic())
asmo=ASMO(problem, rbf, NInit=50)
res=asmo.run(maxFE=1000)