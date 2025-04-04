import sys
sys.path.insert(0, '.')

import numpy as np
from UQPyL.problems import Problem

# Step 2: define objFunc Function
# Here, X is default to a numpy 2-dimensional matrix. 
# Each row in X represents a candidate solution (i.e., an individual in the population) 
# and each column corresponds to a decision variable (i.e., a feature or parameter to be optimized).
#
# The objective function (objFunc) needs to return the objective values (objs) for each solution, which should also be a 2-dimensional matrix.
# The number of rows in objs should match the number of rows in X (i.e., the number of candidate solutions),
#and the number of columns should match the number of objectives.
# For a single-objective problem, objs will have shape (N, 1);
# For a multi-objective problem with M objectives, objs will have shape (N, M),
# Where N is the number of candidate solutions (rows of X), and M is the number of objectives.
def objFunc(X):
    N, D = X.shape 
    #This is a vectorized operation over the input matrix X.
    objs =100 * (X[:, 2] - X[:, 1]**2)**2+ 100 * (X[:, 1] - X[:, 0]**2)**2  + \
            (1 - X[:, 1])**2 + (1 - X[:, 0])**2 
    return objs[:, None]

# Another way to define objFunc Function
# For problems that involve using computational models, the UQPyL package provides a decorator 
# to enable a "single run mode", which means that the objective function will be evaluated 
# for one solution at a time, rather than processing multiple solutions in a batch.
# The decorator @singleFunc ensures that the function operates on a single solution (i.e., one row from X) 
# for each call, which is particularly useful in scenarios where each evaluation is computationally expensive
# or when the model is designed to handle one solution at a time.
# Therefore, the input X is a numpy 1-dimensional array.
# In this example, objFunc_ calculates the objective for a single solution X (with two variables). 
# The function returns the objective value corresponding to this solution.
from UQPyL.problems import singleFunc

@singleFunc
def objFunc_(X):
    #This is element-wise operation on the 1-dimensional individual of X.
    obj = 100 * (X[2] - X[1]**2)**2 + 100 * (X[1] - X[0]**2)**2 + \
            (1 - X[1])**2 + (1 - X[0])**2 
    return obj

# Step 3: Define concFunc Function
# Similar to the objFunc function, there are also two ways to define the constraint function.
# Note that the return value of concFunc reflects how much the constraints are violated.
# - If the return value is less than 0, it indicates that the constraint is violated. 
#   The smaller the value, the more severely the constraint is violated.
# - If the return value is greater than 0, it means the constraint is satisfied (normal solution).
#Therefore, users may need to modify or re-formulate the constraint functions.

# Matrix Mode
def conFunc(X):
    cons = X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2 -4 
    return cons[:, None]

# Single Run Mode
@singleFunc
def conFunc(X):
    con = X[0]**2 + X[1]**2 + X[2]**2 -4 
    return con

# Step 4: describe the properties of X

nInput = 3 # number of input variables (X), here it's 3 inputs.
nOutput = 1 # number of outputs (objective functions), here it's 1 objective.

#Upper bound of X.
ub = [0, 0, 0] # It can be a float, int, list, or numpy array. 
# In this case, both input variables have an upper bound of 0. 

# Lower bound of X.
lb = [10, 10, 10] # It can also be a float, int, list, or numpy array. 
# In this case, both input variables have a lower bound of 10.

# Types of variables.
# type 0 for continuous, 1 for integer, and 2 for discrete.
varType = [0, 1, 2]  
# varType[0] = 0: The first input (X[0]) is a float.
# varType[1] = 1: The second input (X[1]) is an integer variable.
# varType[2] = 2: The second input (X[2]) is a discrete variable.

# The set of possible values for discrete variables.
varSet = {2: [2, 3.4, 5.1, 7]} 
# varSet is a dictionary where the key indicates the index of the variable (2 refers to the third variable, X[2]). It follows Python's zero-based indexing.
# The value associated with key 2 specifies the set of possible values for X[2]: [2, 3.4, 5.1, 7].
# This means that X[2] can only take one of these four values: 2, 3.4, 5.1, or 7.

# The optimization type: 'min' for minimization, 'max' for maximization.
optType = 'min'

# Names (or labels) for the input variables.
xLabels = ['x1', 'x2', 'x3'] 
# If the optimization problem has named variables, you can set them here.
# Otherwise, default names like 'x1', 'x2', 'x3', etc., can be used.

# Names (or labels) for the objective functions.
yLabels = ['obj1'] # Similar to xLabel, if your objective(s) have specific names, you can set them here.
# Otherwise, use default labels like 'obj1', 'obj2', etc.

# Name of the optimization problem
name = 'Rosenbrock'
# Useful for identifying the problem instance, organizing results, saving files, etc.

#Step 5: Initialize the problem instance
problem = Problem(nInput = nInput, nOutput = nOutput, objFunc = objFunc, conFunc = conFunc,
                    ub = ub, lb = lb, varType = varType, varSet = varSet,
                        xLabels = xLabels, yLabels = yLabels, name = name)

# Step 6: Use optimization methods from UQPyL
# All methods and algorithms in UQPyL operate by reading the 'problem' object
# In this example, we are using the Genetic Algorithm (GA) for optimization
from UQPyL.optimization.single_objective import GA

# Create an instance of the Genetic Algorithm (GA). By default, GA will output optimization history
# and final results in the command line.
ga = GA()

# Run the Genetic Algorithm optimization by passing the defined 'problem' object
ga.run(problem = problem)