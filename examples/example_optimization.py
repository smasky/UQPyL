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
temp=ga.run()
a=1