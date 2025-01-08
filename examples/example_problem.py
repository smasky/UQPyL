import sys
sys.path.append('.')

import numpy as np



from UQPyL.problems import ProblemABC
#define objFunc
@ProblemABC.singleEval
def objFunc(x):
    
    y = (x[0]-1.5)**2 + (x[1]-0.5)**2 + \
        3*(x[2]-1.1)**2 + (x[3]-2)**2 + \
        5*(x[4]-3.7)**2
    
    return y

@ProblemABC.singleEval
def concFunc(x):
    
    cv1 = (x[0] - 1)**2 - 0.25
    cv2 = (x[0] - 1)**2 - 1
    
    return [cv1, cv2]

from UQPyL.problems import PracticalProblem
setting = {
    "name" : "F1",
    "nInput": 5,
    "nOutput": 1,
    "ub": [1.0] * 10,
    "lb": [0.0] * 10,
    "objFunc": objFunc,
    "concFunc" : concFunc,
    "varType" : [0, 1, 2, 0, 0],
    "varSet" : {2: [2.2, 3.1, 5.5, 6.8]},
    "xLabel" : [ f"x{i}" for i in range(1, 6)]
}

problem = PracticalProblem(**setting)