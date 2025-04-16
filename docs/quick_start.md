## Quick Start

To effectively use UQPyL, the **first step** is to define the problem you solve:

1. The **basic information** of  the problem, e.g., the dimension, range, value type  (float, int, or discrete) of each variable, name of the problem, decisions, objectives.

2. The **function** from input variables `x` to output objective `obj` named `objFunc` in UQPyL, i.e., how the output `obj` is obtained from the inputs `x`, which could be an analytical function, or include running computational model, external black-box process. If necessary, the constraint functions named `concFunc` should be implemented.

Following problem is a variant of the Rosenbrock function, which adds additional constraint functions ( $x_1^2+x_2^2+x_3^2 \ge 4$ ) and changes the variable types, from the origin `continuous` and `float` to `int` ( $x_2$ ) and `discrete` ( $x_3$ ). Take this as an example to illustrate the specific steps of problem definition. 

<p align="center"><img src="../pic/Problem1.svg" width=700/></p>

UQPyL provide a python class named `Problem` to simplify the workflow of defining problems.


<a href="https://nbviewer.org/github/smasky/UQPyL/blob/dev/notebooks/defefine_problem.ipynb" target="_blank">📘 View Jupyter Notebook example online </a>


```python
# Step 1: import Problem class from UQPyL's problem module
from UQPyL.problems import Problem

# Step 2: define objFunc Function
# The 'objFunc' function is one that accepts a 2D numpy array 'X' as input and returns a 2D numpy array 'objs'. Specifically:
# The rows of the 2D array 'X' represent a set of decision variables, while the columns correspond to different values of the same variable.
# The returned 2D array 'objs' should have the same number of rows as 'X', and the number of columns should equal the number of objectives in the problem.
# For a single-objective problem, the shape of the 2D array 'objs' should be (N, 1), where 'N' is the number of input variables.
# For a multi-objective problem, the shape of the 2D array 'objs' should be (N, M), where 'M' is the number of objective functions.
# Users should ensure that the returned 2D array 'objs' satisfies the above shape requirements.

def objFunc(X):
    # If possible, advise vectorizing operations on matrix X to improve computational efficiency.
    objs =100 * (X[:, 2] - X[:, 1]**2)**2+ 100 * (X[:, 1] - X[:, 0]**2)**2  + \
            (1 - X[:, 1])**2 + (1 - X[:, 0])**2 
    return objs[:, None] # Although UQPyL performs further checks, please ensure the returned `objs` is a 2D array.

# UQPyL also supports an alternative way of defining the objFunc function.
# For problems involving numerical simulation models, it's often not feasible to vectorize operations on matrix 'X'.
# To address this, UQPyL provides a decorator function @singleFunc that enables single running mode.
# In single running mode, The 'objFunc' function only accepts a Python list or 1D numpy array as input.
# It processes one decision variable combination at a time, making it suitable for complex or non-vectorizable objective functions.

# First, import the decorator that enables singleton mode from UQPyL
from UQPyL.problems import singleFunc

@singleFunc
def objFunc_(X):  # Input X should be a 1D numpy array or Python list
     # Perform calculations for each element in X
    obj = 100 * (X[2] - X[1]**2)**2 + 100 * (X[1] - X[0]**2)**2 + \
            (1 - X[1])**2 + (1 - X[0])**2 
    return obj # Return the objective value: a scalar for single-objective, or a 1D array/list for multi-objective

# Step 3: Define concFunc Function
# Similar to objFunc, the conFunc function supports two definition modes.
# Note: The return value of conFunc indicates whether the constraints are satisfied:
# - A negative value indicates a violation of the constraint — the smaller the value, the more severe the violation.
# - A positive value indicates the constraint is satisfied, i.e., the solution is feasible.
# As a result, users may need to reformulate their original constraint expressions to follow this convention.

# Matrix Mode
def conFunc(X):
    cons = X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2 - 4 
    return cons[:, None] 

# Single Running Mode
@singleFunc
def conFunc(X):
    con = X[0]**2 + X[1]**2 + X[2]**2 - 4 
    return con

# Step 4: describe the properties of X

nInput = 3 # number of input variables (X), here it's 3 inputs.
nOutput = 1 # number of outputs (objective functions), here it's 1 objective.

#Upper bound of X.
ub = [10, 10, 10] # It can be a float, int, list, or numpy array. 
# In this case, both input variables have an upper bound of 10. 

# Lower bound of X.
lb = [0, 0, 0] # It can also be a float, int, list, or numpy array. 
# In this case, both input variables have a lower bound of 0.

# Types of variables.
# type 0 for continuous, 1 for integer, and 2 for discrete.
varType = [0, 1, 2]  
# varType[0] = 0: The first input (X[0]) is a float.
# varType[1] = 1: The second input (X[1]) is an integer variable.
# varType[2] = 2: The second input (X[2]) is a discrete variable.

# The set of possible values for discrete variables.
varSet = {2: [2, 3.4, 5.1, 7]} 
# varSet is a dictionary where the key indicates the index of the variable (2 refers to the third variable, x3). It follows Python's zero-based indexing.
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
problem = Problem(
    nInput=nInput,
    nOutput=nOutput,
    objFunc=objFunc,
    conFunc=conFunc,
    ub=ub,
    lb=lb,
    varType=varType,
    varSet=varSet,
    xLabels=xLabels,
    yLabels=yLabels,
    name=name
)

# Step 6: Use optimization methods from UQPyL
# All methods and algorithms in UQPyL operate by reading the 'problem' object
# In this example, we are using the Genetic Algorithm (GA) for optimization
from UQPyL.optimization.single_objective import GA

# Create an instance of the Genetic Algorithm (GA). By default, GA will output optimization history
# and final results in the command line.
ga = GA()

# Run the Genetic Algorithm optimization by passing the defined 'problem' object
ga.run(problem = problem)
# Output:
# Time:  0.0 day | 0.0 hour | 0.0 minute |  1.17 second
# Used FEs:    50000  |  Iters:  999
# Best Objs and Best Decision with the FEs
# +-------------------+-------------------+-------------------+-------------------+
# |        FEs        |       Iters       |      OptType      |      Feasible     |
# +-------------------+-------------------+-------------------+-------------------+
# |         50        |         0         |        min        |        True       |
# +-------------------+-------------------+-------------------+-------------------+
# +-------------------+-------------------+-------------------+-------------------+
# |        obj1       |         x1        |         x2        |         x3        |
# +-------------------+-------------------+-------------------+-------------------+
# |      4.0e+02      |       0.000       |       0.000       |       2.000       |
# +-------------------+-------------------+-------------------+-------------------+
```

### Benchmark Problems
UQPyL provides built-in benchmark problems (inheriting from `Problem` class) to test methods.
```python
from UQPyL.problems.single_objective import Sphere, Ackley
from UQPyL.problems.multi_objective import ZDT1, DTLZ1

# Create benchmark problems for testing algorithms
# You can easily customize the input dimension and variable bounds as needed

# Single-objective benchmark problems
problem1 = Sphere(nInput=10, ub=100, lb=-100)  # Sphere function with 10 dimensions, bounds [-100, 100]
problem2 = Ackley(nInput=10, ub=np.ones(10)*100, lb=np.ones(10)*-100)  # Ackley function with 10 dimensions

# Multi-objective benchmark problems
problem3 = ZDT1(nInput=5)   # ZDT1 problem with 5 decision variables
problem4 = DTLZ1(nInput=15) # DTLZ1 problem with 15 decision variables

# UQPyL provides ready-to-use benchmark problems for both single and multi-objective optimization.
# You can easily adjust input dimensions and variable bounds to suit your testing needs.
```



### Sensitivity Analysis

Here, use Ishigami Function as example.
<p align="center"><img src=../pic/Problem2.svg" width=500 /></p>

The theoretical sensitivity indices of the Ishigami function are as follows:
First-order sensitivity indices: x1 = 0.314, x2 = 0.442, x3 = 0.000
Total-order sensitivity indices: x1 = 0.558, x2 = 0.442, x3 = 0.244

<a href="https://nbviewer.org/github/smasky/UQPyL/blob/dev/notebooks/sensitivity_analysis.ipynb" target="_blank">📘 View Jupyter Notebook example online </a>

```python
import numpy as np
from UQPyL.problems import Problem
# Define Ishigami Function
def objFunc(X):
    objs = np.sin(X[:, 0]) + 7 * np.sin(X[:, 1])**2 + \
                 0.1 * X[:, 2]**4 * np.sin(X[:, 0])
    return objs[:, None]

Ishigami = Problem(nInput = 3, nOutput = 1, objFunc = objFunc,
                    ub = np.pi, lb = -1*np.pi, varType = [0, 0, 0],
                    name = "Ishigami")
                    
from UQPyL.sensibility import Sobol

# Instantiate a Sobol sensitivity analysis object
sobol = Sobol()
# N = 512 defines the base sample size; 
# total number of evaluations will be larger due to Sobol' method structure
X = sobol.sample(problem = Ishigami, N = 512)

# Evaluate the objective function (i.e., Ishigami function) on the sample points
# Returns an array of function outputs corresponding to each input in X
Obj = problem.objFunc(X)

# Perform Sobol' sensitivity analysis
# Inputs:
#   - problem: the problem instance (defines bounds and function)
#   - X: the input samples
#   - Obj: the function evaluations at X
sobol.analyze(problem, X, Obj)

# By default, the following results will be obtained:
# =======================Attribute=======================
# First Order Sensitivity: True
# Second Order Sensitivity: False
# Total Order Sensitivity: True
# ======================Conclusion=============================
# --------------------------S1---------------------------------
# +-------------------+-------------------+-------------------+
# |        x_1        |        x_2        |        x_3        |
# +-------------------+-------------------+-------------------+
# |       0.3222      |       0.4531      |       0.0175      |
# +-------------------+-------------------+-------------------+
# --------------------------ST---------------------------------
# +-------------------+-------------------+-------------------+
# |        x_1        |        x_2        |        x_3        |
# +-------------------+-------------------+-------------------+
# |       0.5436      |       0.4306      |       0.2416      |
# +-------------------+-------------------+-------------------+

```
### Optimization

Here's an example using SCE-UA to optimize the Sphere function

<a href="https://nbviewer.org/github/smasky/UQPyL/blob/dev/notebooks/optimization.ipynb" target="_blank">📘 View Jupyter Notebook example online </a>

```python

# Import the Sphere benchmark function
from UQPyL.problems.single_objective import Sphere

# Instantiate the problem with 10 input dimensions; other settings use defaults
sphere = Sphere(nInput = 10) #Other settings use default

# Import the SCE-UA optimization algorithm
from UQPyL.optimization.single_objective import SCE_UA

# Instantiate the optimizer with default settings
sce = SCE_UA()

# Run the optimization on the Sphere problem
res = sce.run(sphere)

# Extract the best decision variables and objective values
bestDecs = res.bestDecs
bestObjs = res.bestObjs

# The program would display the optimization history in the command line
# =========Conclusion================================= 
# Time:  0.0 day | 0.0 hour | 0.0 minute |  5.32 second
# Used FEs:    24356  |  Iters:  1000
# Best Objs and Best Decision with the FEs
# +-----------------+-----------------+-----------------+-----------------+-----------------+
# |       FEs       |      Iters      |     OptType     |     Feasible    |       y_1       |
# +-----------------+-----------------+-----------------+-----------------+-----------------+
# |      24122      |       990       |       min       |       True      |     4.6e-12     |
# +-----------------+-----------------+-----------------+-----------------+-----------------+
# +-----------------+-----------------+-----------------+-----------------+-----------------+
# |       x_1       |       x_2       |       x_3       |       x_4       |       x_5       |
# +-----------------+-----------------+-----------------+-----------------+-----------------+
# |      -0.000     |      -0.000     |      0.000      |      0.000      |      -0.000     |
# +-----------------+-----------------+-----------------+-----------------+-----------------+
# +-----------------+-----------------+-----------------+-----------------+-----------------+
# |       x_6       |       x_7       |       x_8       |       x_9       |       x_10      |
# +-----------------+-----------------+-----------------+-----------------+-----------------+
# |      -0.000     |      -0.000     |      0.000      |      -0.000     |      -0.000     |
# +-----------------+-----------------+-----------------+-----------------+-----------------+
```

### Surrogate Modeling

Use RBF model to predict Sphere Function as an example

<a href="https://nbviewer.org/github/smasky/UQPyL/blob/dev/notebooks/surrogate_modelling.ipynb" target="_blank">📘 View Jupyter Notebook example online </a>

```python
from UQPyL.problems import Sphere
#Instantiate the problem
sphere = Sphere(nInput = 10)

# Import Latin Hypercube Sampling (LHS) for generating design of experiments
from UQPyL.DoE import LHS

# Generate 200 training samples in the input space using LHS
lhs = LHS()
xTrain = lhs.sample(200, problem.nInput, problem = sphere)

# Evaluate the true objective function at training points
yTrain = sphere.objFunc(xTrain)

# Generate 50 test samples for model validation
xTest = lhs.sample(50, problem.nInput, problem = sphere)
# Evaluate the true function at test points
yTest = sphere.objFunc(xTest)

# Import Radial Basis Function (RBF) surrogate model
from UQPyL.surrogate.rbf import RBF

# Initialize and fit the RBF surrogate model
rbf = RBF()
rbf.fit(xTrain, yTrain)

# Use the trained RBF model to predict outputs on test inputs
yPred = rbf.predict(xTest)

# Import R² metric to evaluate surrogate model performance
from UQPyL.utility.metric import r_square
# Compute R² score between true and predicted test outputs
r2 = r_square(yTest, yPred)
print(r2)
```