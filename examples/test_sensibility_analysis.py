import sys
sys.path.insert(0, '.')

import numpy as np

#Ishigami
#S1 = 0.314
#S2 = 0.442
#S3 = 0
#S12 = 0
#S13 = 0.244
#S23 = 0 
#ST1 = 0.558
#ST2 = 0.442
#ST3 = 0.244

from UQPyL.problems import Problem

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

#Sobol
# from UQPyL.sensibility import Sobol

# sobol = Sobol(calSecondOrder = True)
# res = sobol.analyze(problem)
# print(res)

#FAST
# from UQPyL.sensibility import FAST
# fast = FAST()

# res = fast.analyze(problem)
# print(res)

#RBD-FAST

