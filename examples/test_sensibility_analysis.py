import sys
sys.path.insert(0, '.')

import numpy as np

#Ishigami Function
#S1 = 0.314
#S2 = 0.442
#S3 = 0
#S12 = 0
#S13 = 0.244
#S23 = 0 
#ST1 = 0.558
#ST2 = 0.442
#ST3 = 0.244

# from UQPyL.problems import Problem

# def objFunc(X):
    
#     Y = np.sin(X[:, 0]) + 7* (np.sin(X[:, 1])**2) + 0.1* X[:, 2]**4*np.sin(X[:, 0])
    
#     return Y[:, np.newaxis]

# setting = {
#     'nInput' : 3,
#     'nOutput' : 1,
#     'ub' : np.pi,
#     'lb' : -np.pi,
#     'objFunc' : objFunc
# }

# problem = Problem(**setting)

#Sobol
# from UQPyL.sensibility import Sobol

# sobol = Sobol(calSecondOrder = True)

# X = sobol.sample(problem, 512)

# res = sobol.analyze(problem, X)
# print(res)

#FAST
# from UQPyL.sensibility import FAST
# fast = FAST(verboseFlag = True)

# X = fast.sample(problem)

# res = fast.analyze(problem, X)

# print(res)

#RBD-FAST
# from UQPyL.sensibility import RBD_FAST

# rbd_fast =  RBD_FAST()

# X = rbd_fast.sample(problem, 500)

# res = rbd_fast.analyze(problem, X)

# print(res)


#Non-monotonic Sobol G Function (8 parameters)
# First-order indices
# S1: 0.5065 0.1791 0.0237 0.0072 0.000 0.0 0.0 0.0 0.0

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

# from UQPyL.sensibility import Morris

# morris = Morris()

# X = morris.sample(problem, numTrajectory=500)

# res = morris.analyze(problem, X)
# print(res)


# from UQPyL.sensibility import RSA

# rsa = RSA()

# X = rsa.sample(problem, N = 1000)

# res = rsa.analyze(problem, X)
# print(res)