import sys
sys.path.insert(0, '.')


import numpy as np
from UQPyL.problems import Problem
# Define Ishigami Function
def objFunc(X):
    objs = np.sin(X[:, 0]) + 7 * np.sin(X[:, 1])**2 + \
                 0.1 * X[:, 2]**4 * np.sin(X[:, 0])
    return objs[:, None]

ishigami = Problem(nInput = 3, nOutput = 1, objFunc = objFunc,
                    ub = np.pi, lb = -1*np.pi, varType = [0, 0, 0],
                    name = "Ishigami")
                    
from UQPyL.sensibility import Sobol

# Instantiate a Sobol sensitivity analysis object
sobol = Sobol()
# N = 512 defines the base sample size; 
# total number of evaluations will be larger due to Sobol' method structure
X = sobol.sample(problem = ishigami, N = 512)

# Evaluate the objective function (i.e., Ishigami function) on the sample points
# Returns an array of function outputs corresponding to each input in X
Obj = ishigami.objFunc(X)

# Perform Sobol' sensitivity analysis
# Inputs:
#   - problem: the problem instance (defines bounds and function)
#   - X: the input samples
#   - Obj: the function evaluations at X
res = sobol.analyze(ishigami, X, Obj)

# Print the results
print(res)