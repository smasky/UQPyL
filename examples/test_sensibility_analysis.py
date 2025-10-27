import sys
sys.path.insert(0, '.')

import numpy as np
from UQPyL.problem import Problem



# ------------------------------------------- #
#              Ishigami Function              # 
# ------------------------------------------- #

# S1 0.314 0.442 0.000
# S2 0 0.244 0
# ST 0.558 0.442 0.244

def objFunc(X):
    
    Y = np.sin(X[:, 0]) + 7* (np.sin(X[:, 1])**2) + 0.1* X[:, 2]**4*np.sin(X[:, 0])
    
    return Y[:, np.newaxis]

setting = {
    'nInput' : 3,
    'nOutput' : 1,
    'ub' : np.pi,
    'lb' : -np.pi,
    'objFunc' : objFunc
}

problem = Problem(**setting)


# ------------------------------------------- #
#     Non-monotonic Sobol G Function          # 
# ------------------------------------------- #

# First-order indices
# S1: 0.7065 0.1791 0.0237 0.0072 0.000 0.0 0.0 0.0 0.0

# def objFunc(X):
    
#     a = np.array([0, 1, 4.5, 9, 99, 99, 99, 99])
#     alpha = np.ones_like(a)
#     Ytemp = np.zeros(X.shape)

#     for i in range(8):
        
#         Ytemp[:, i] = ((1 + alpha[i]) * np.abs(2 * X[:, i] - 1) ** alpha[i] + a[i]) / (1 + a[i])
        
#     Y = Ytemp.prod(axis=1)[:, np.newaxis]
    
#     return Y

# setting = {
#     'nInput' : 8,
#     'nOutput' : 1,
#     'ub' : 1,
#     'lb' : 0,
#     'objFunc' : objFunc
# }

# problem = Problem(**setting)

# ------------------------------------------- #
#                   Sobol                     # 
# ------------------------------------------- #

from UQPyL.analysis import Sobol

sobol = Sobol(saveFlag = True)

X = sobol.sample(problem, 512, secondOrder = True)

Y = problem.objFunc(X)

res = sobol.analyze(problem, X, Y, secondOrder = True)

# ------------------------------------------- #
#                   FAST                      # 
# ------------------------------------------- #

# from UQPyL.analysis import FAST

# fast = FAST(verboseFlag = True, saveFlag = True)

# X = fast.sample(problem, N = 512)

# Y = problem.objFunc(X)

# res = fast.analyze(problem, X, Y)

# ------------------------------------------- #
#                   RBD-FAST                  # 
# ------------------------------------------- #

# from UQPyL.analysis import RBDFAST

# rbd_fast =  RBDFAST(saveFlag = True)

# X = rbd_fast.sample(problem, 500)

# Y = problem.objFunc(X)

# res = rbd_fast.analyze(problem, X, Y)

# ------------------------------------------- #
#                   Morris                    # 
# ------------------------------------------- #

# from UQPyL.analysis import Morris

# morris = Morris(saveFlag = True)

# X = morris.sample(problem, numTrajectory=500)

# Y = problem.objFunc(X)

# res = morris.analyze(problem, X, Y)

# ------------------------------------------- #
#                   RSA                       # 
# ------------------------------------------- #

# from UQPyL.analysis import RSA

# rsa = RSA(saveFlag = True)

# X = rsa.sample(problem, N = 1000)

# Y = problem.objFunc(X)

# res = rsa.analyze(problem, X, Y)

# ------------------------------------------- #
#                   MARS                      # 
# ------------------------------------------- #

# from UQPyL.analysis import MARS

# mars = MARS(saveFlag = True)

# X = mars.sample(problem, N = 1000)

# Y = problem.objFunc(X)

# res = mars.analyze(problem, X, Y)

# ------------------------------------------- #
#                   Delta-Test                # 
# ------------------------------------------- #

# from UQPyL.analysis import DeltaTest

# delta_test = DeltaTest(saveFlag = True)

# X = delta_test.sample(problem, N = 1000 )

# Y = problem.objFunc(X)

# res = delta_test.analyze(problem, X, Y)