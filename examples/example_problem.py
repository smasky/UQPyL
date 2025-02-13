import sys
sys.path.append('.')

import numpy as np

from UQPyL.problems import ProblemABC

#-----------------Single Evaluation Separate Mode-----------------#
# @ProblemABC.singleFunc
# def objFunc(x):
    
#     y = (x[0]-1.5)**2 + (x[1]-0.5)**2 + \
#         3*(x[2]-1.1)**2 + (x[3]-2)**2 + \
#         5*(x[4]-3.7)**2
    
#     return y[:, np.newaixs]

# @ProblemABC.singleFunc
# def conFunc(x):
    
#     cv1 = (x[0] - 1)**2 - 0.25
#     cv2 = (x[0] - 1)**2 - 1
    
#     return np.column_stack((CV1, CV2))

#-----------------------Single Evaluation Integration Mode-------------------------------------------#

# @ProblemABC.singleEval
# def evaluate(x):
#     y = (x[0]-1.5)**2 + (x[1]-0.5)**2 + \
#         3*(x[2]-1.1)**2 + (x[3]-2)**2 + \
#         5*(x[4]-3.7)**2
    
#     cv1 = (x[0] - 1)**2 - 0.25
#     cv2 = (x[0] - 1)**2 - 1

#     return {'objs' : y, 'cons' : np.column_stack((CV1, CV2)) }

#----------------Ordinary Evaluation Integration Mode------------------------#
# def evaluate(X):
    
#     y = (X[:, 0]-1.5)**2 + (X[:, 1]-0.5)**2 + \
#          3*(X[:, 2]-1.1)**2 + (X[:, 3]-2)**2 + \
#          5*(X[:, 4]-3.7)**2
    
#     CV1 = (X[:, 0] - 1)**2 - 0.25
#     CV2 = (X[:, 0] - 1)**2 - 1
    
#     return {'objs' : y, 'cons' : np.column_stack((CV1, CV2))}

#----------------Ordinary Evaluation Separate Mode-----------#
def objFunc(X):
    
    y = (X[:, 0]-1.5)**2 + (X[:, 1]-0.5)**2 + \
            3*(X[:, 2]-1.1)**2 + (X[:, 3]-2)**2 + \
                5*(X[:, 4]-3.7)**2
                
    return y[:, np.newaxis]

def conFunc(X):
    
    CV1 = (X[:, 0] - 1)**2 - 0.25
    CV2 = (X[:, 0] - 1)**2 - 1
    
    return np.column_stack((CV1, CV2))
    
from UQPyL.problems import Problem

setting = {
    "name" : "F1",
    "nInput": 5,
    "nOutput": 1,
    "ub": [1.0] * 5,
    "lb": [0.0] * 5,
    # "evaluate" : evaluate,
    'objFunc' : objFunc,
    'conFunc' : conFunc,
    "varType" : [0, 1, 2, 0, 0],
    "varSet" : {2: [2.2, 3.1, 5.5, 6.8]},
    "xLabels" : [ f"x{i}" for i in range(1, 6)]
}

problem = Problem(**setting)

from UQPyL.optimization import GA, PSO

ga = GA()

pso = PSO()

res = ga.run(problem)