import sys
sys.path.insert(0, '.')

import numpy as np
from UQPyL.problems import Problem, ProblemABC

#-----------------------------------------------#
#Type1: Objective Function and Constraint Function are separate

#Return np.2darray
def objFunc(X):
    
    y = (X[:, 0]-1.5)**2 + (X[:, 1]-0.5)**2 + \
            3*(X[:, 2]-1.1)**2 + (X[:, 3]-2)**2 + \
                5*(X[:, 4]-3.7)**2
    
    return y[:, np.newaxis]

#Return np.2darray
def conFunc(X):
    
    CV1 = (X[:, 0] - 1)**2 - 0.25
    CV2 = (X[:, 0] - 1)**2 - 1
    
    return np.column_stack((CV1, CV2))

#Basic Information for Problem
setting = {
    "name" : "F1",
    "nInput": 5,
    "nOutput": 1,
    "ub": [1.0] * 5,
    "lb": [0.0] * 5,
    'objFunc' : objFunc,
    'conFunc' : conFunc,
    "varType" : [0, 1, 2, 0, 0],
    "varSet" : {2: [2.2, 3.1, 5.5, 6.8]},
    "xLabels" : [ f"x{i}" for i in range(1, 6)]
}

problem = Problem(**setting)

X = np.random.random((10, 5))

res = problem.evaluate(X)

#Print result
print("-"*10+"Type1"+"-"*10)
for key, value in res.items():
    print(f"{key}:")
    print(value)
print("-"*20)


#-----------------------------------------------#
#Type2: Objective Function and Constraint Function are separate and single evaluation mode

#Use special decorator
#Return np.1darray, float, int or list of them
@ProblemABC.singleFunc
def objFunc(x):
    
    y = (x[0]-1.5)**2 + (x[1]-0.5)**2 + \
            3*(x[2]-1.1)**2 + (x[3]-2)**2 + \
                5*(x[4]-3.7)**2

    return y

#Return np.1darray, float, int or list of them
@ProblemABC.singleFunc
def conFunc(x):
    
    cv1 = (x[0] - 1)**2 - 0.25
    cv2 = (x[0] - 1)**2 - 1
    
    return [cv1, cv2]

#Basic Information for Problem
setting = {
    "name" : "F1",
    "nInput": 5,
    "nOutput": 1,
    "ub": [1.0] * 5,
    "lb": [0.0] * 5,
    'objFunc' : objFunc,
    'conFunc' : conFunc,
    "varType" : [0, 1, 2, 0, 0],
    "varSet" : {2: [2.2, 3.1, 5.5, 6.8]},
    "xLabels" : [ f"x{i}" for i in range(1, 6)]
}

problem = Problem(**setting)

X = np.random.random((10, 5))

res = problem.evaluate(X)


#Print result
print("-"*10+"Type2"+"-"*10)
for key, value in res.items():
    print(f"{key}:")
    print(value)
print("-"*20)


#-----------------------------------------------#
#Type3: Use Evaluate Function to integrate Objective and Constraint Function

def evaluate(X):
    
    Y = (X[:, 0]-1.5)**2 + (X[:, 1]-0.5)**2 + \
            3*(X[:, 2]-1.1)**2 + (X[:, 3]-2)**2 + \
                5*(X[:, 4]-3.7)**2
    
    CV1 = (X[:, 0] - 1)**2 - 0.25
    CV2 = (X[:, 0] - 1)**2 - 1
    
    return {'objs' : Y, 'cons' : np.column_stack((CV1, CV2))}

#Basic Information for Problem
setting = {
    "name" : "F1",
    "nInput": 5,
    "nOutput": 1,
    "ub": [1.0] * 5,
    "lb": [0.0] * 5,
    "evaluate" : evaluate,
    "varType" : [0, 1, 2, 0, 0],
    "varSet" : {2: [2.2, 3.1, 5.5, 6.8]},
    "xLabels" : [ f"x{i}" for i in range(1, 6)]
}

X = np.random.random((10, 5))

res = problem.evaluate(X)

#Print result
print("-"*10+"Type3"+"-"*10)
for key, value in res.items():
    print(f"{key}:")
    print(value)
print("-"*20)

#-------------------------------------------#
#Type3: Under single evaluation, Use Evaluate Function to integrate Objective and Constraint Function

@ProblemABC.singleEval
def evaluate(x):
    
    y = (x[0]-1.5)**2 + (x[1]-0.5)**2 + \
            3*(x[2]-1.1)**2 + (x[3]-2)**2 + \
                5*(x[4]-3.7)**2
    
    cv1 = (x[0] - 1)**2 - 0.25
    cv2 = (x[0] - 1)**2 - 1
    
    return { 'objs' : y, 'cons': [cv1, cv2]}

#Basic Information for Problem
setting = {
    "name" : "F1",
    "nInput": 5,
    "nOutput": 1,
    "ub": [1.0] * 5,
    "lb": [0.0] * 5,
    "evaluate" : evaluate,
    "varType" : [0, 1, 2, 0, 0],
    "varSet" : {2: [2.2, 3.1, 5.5, 6.8]},
    "xLabels" : [ f"x{i}" for i in range(1, 6)]
}

X = np.random.random((10, 5))

res = problem.evaluate(X)

#Print result
print("-"*10+"Type4"+"-"*10)
for key, value in res.items():
    print(f"{key}:")
    print(value)
print("-"*20)
