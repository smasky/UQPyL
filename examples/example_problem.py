import numpy as np
from UQPyL.problem import Problem, ProblemABC
from UQPyL.doe import LHS

#---------------Problem Construction---------------#

# Type1: Objective Function and Constraint Function are integrated in the evaluate function
# Tip1 : Return a dictionary with keys 'objs' and 'cons'
# Tip2 :CV < 0 denotes the feasible solution
# Tip3 : The shape of 'objs' and 'cons' should be (N, M) and (N, K) respectively
#        N is the number of samples, M is the number of objectives, K is the number of constraints
def evaluate(X):
    
    y = (X[:, 0]-1.5)**2 + (X[:, 1]-0.5)**2 + \
            3*(X[:, 2]-1.1)**2 + (X[:, 3]-2)**2 + \
                5*(X[:, 4]-3.7)**2
    
    CV1 = (X[:, 0] - 1)**2 - 0.25
    CV2 = (X[:, 1] - 1)**2 - 1
    
    return {'objs' : y[:, np.newaxis], 'cons' : np.column_stack((CV1, CV2))}

setting = {
    "name" : "F1",
    "nInput": 5,
    "nOutput": 1,
    "ub": [1.0] * 5,
    "lb": [0.0] * 5,
    'evaluate' : evaluate,
    "varType" : [0, 1, 2, 0, 0],
    "varSet" : {2: [2.2, 3.1, 5.5, 6.8]},
    "xLabels" : [ f"x{i}" for i in range(1, 6)]
}

problem = Problem(**setting)

X = LHS(criterion = 'classic').sample(problem, 10)

res = problem.evaluate(X)

objs = res['objs']
cons = res['cons']

#Type2: Objective Function and Constraint Function are separate

#Return np.2darray
def objFunc(X):
    
    y = (X[:, 0]-1.5)**2 + (X[:, 1]-0.5)**2 + \
            3*(X[:, 2]-1.1)**2 + (X[:, 3]-2)**2 + \
                5*(X[:, 4]-3.7)**2
                
    # Keep the shape of y as (N, M)
    # N is the number of samples, M is the number of objectives
    return y[:, np.newaxis]

# Return np.2darray
# CV < 0 denotes the feasible solution
def conFunc(X):
    
    CV1 = (X[:, 0] - 1)**2 - 0.25
    CV2 = (X[:, 1] - 1)**2 - 1
    
    # integrate CV1 and CV2 into a 2d array
    # Column 0 is CV1, Column 1 is CV2
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

X = LHS(criterion = 'classic').sample(problem, 10)

# Call the evaluate method to get the result
res = problem.evaluate(X)
objs = res['objs']
cons = res['cons']

# Call the objFunc and conFunc method to get the result
objs = problem.objFunc(X)
cons = problem.conFunc(X)


#Type3: Objective Function and Constraint Function are integrated in the evaluate function and single evaluation mode

# Tip1 : Use python decorator to enable single evaluation mode
#        For evaluate function, using problemABC.singleEval
# Tip2 : the input of evaluate function would be np.1darray to denote a single sample
# Tip3 : The return of evaluate function should be a dictionary with keys 'objs' and 'cons'
#        objs should be np.1darray, float, int or list of them
#        cons should be np.1darray, float, int or list of them
@ProblemABC.singleEval
def evaluate(x):
    
    y = (x[0]-1.5)**2 + (x[1]-0.5)**2 + \
            3*(x[2]-1.1)**2 + (x[3]-2)**2 + \
                5*(x[4]-3.7)**2
    
    CV1 = (x[0] - 1)**2 - 0.25
    CV2 = (x[1] - 1)**2 - 1
    
    return {'objs' : y, 'cons' : [CV1, CV2]}


setting = {
    "name" : "F1",
    "nInput": 5,
    "nOutput": 1,
    "ub": [1.0] * 5,
    "lb": [0.0] * 5,
    'evaluate' : evaluate,
    "varType" : [0, 1, 2, 0, 0],
    "varSet" : {2: [2.2, 3.1, 5.5, 6.8]},
    "xLabels" : [ f"x{i}" for i in range(1, 6)]
}

problem = Problem(**setting)

X = LHS(criterion = 'classic').sample(problem, 10)

res = problem.evaluate(X)

objs = res['objs']
cons = res['cons']


#Type4: Objective Function and Constraint Function are separate and single evaluation mode

#Use python decorator to enable single evaluation mode
# Tip1 : For objFunc and conFunc, using problemABC.singleFunc
# Tip2 : the input of objFunc and conFunc would be np.1darray to denote a single sample
# Tip3 : The return of objFunc and conFunc should be np.1darray, float, int or list of them

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

X = LHS(criterion = 'classic').sample(problem, 10)

res = problem.evaluate(X)
objs = res['objs']
cons = res['cons']