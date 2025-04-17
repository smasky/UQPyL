## Problem Define

---

### 1. Use `evaluate` Function

In practical applications, it may be difficult to separately define the `objFunc` and `conFunc` functions for a given problem. In such cases, UQPyL recommends using the `evaluate` function of the Problem class instead.
Still, take this problem as example:

<p align="center"><img src="./pic/Problem1.svg" width=550/></p>

```python

# both the objective and constraint computations are handled within the `evaluate` function.
def evaluate(X):

    # objective
    objs =100 * (X[:, 2] - X[:, 1]**2)**2+ 100 * (X[:, 1] - X[:, 0]**2)**2  + \
            (1 - X[:, 1])**2 + (1 - X[:, 0])**2 
    
    # constraint
    cons = cons = X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2 - 4 

    # different with `objFunc` and `conFunc`, `evaluate` function should return a python dict, which contain two fixed keywords: `objs` and `cons`.

    # still keep return np.2d-array.

    return {'objs' : objs[:, None], 'cons' : cons[:, None]}

#support single running mode
from UQPyL.problems import singleEval

@singleEval
def evaluate(x):

    # objective
    obj = 100 * (X[2] - X[1]**2)**2 + 100 * (X[1] - X[0]**2)**2 + \
            (1 - X[1])**2 + (1 - X[0])**2 

    # constraint

    con = X[0]**2 + X[1]**2 + X[2]**2 - 4

    # still return a python dict with two keywords, `objs` and `cons`
    # obj and con should be int, float or np.1d-array

    return {'objs' : obj, 'cons' : con}

# remaining is same as the quick start example.

nInput = 3 
nOutput = 1 

ub = [10, 10, 10]
lb = [0, 0, 0]

varType = [0, 1, 2] 
varSet = {2: [2, 3.4, 5.1, 7]} 

optType = 'min'

xLabels = ['x1', 'x2', 'x3'] 
yLabels = ['obj1']

name = 'Rosenbrock'

# use `evaluate` keyword to replace `objFunc` and `conFunc`
problem = Problem(
    nInput = nInput,
    nOutput = nOutput,
    evaluate = evaluate,
    ub = ub,
    lb = lb,
    varType = varType,
    varSet = varSet,
    xLabels = xLabels,
    yLabels = yLabels,
    name = name
)

```

When using the `Problem.evaluate` function, the return value is a Python dictionary. To access the objective values and constraints, use the keys 'objs' and 'cons', respectively — for example: `res['objs']` and `res['cons']`.


### 2. Inherit from `ProblemABC` class

UQPyL allows users to define custom problem classes by extending the built-in `Problem` framework. To do so, simply inherit from the abstract base class `ProblemABC`.

```python
from UQPyL.problems import ProblemABC

class NewProblem(ProblemABC):

    def __init__(nInput, nOutput, ub, lb):

        # Call the initializer of ProblemABC in the constructor
        super().__init__(nInput, nOutput, ub, lb)

        pass
    
    # You should override either `evaluate`, `objFunc`, or `conFunc`.
    # It is not recommended to override all of them, as it may cause untested side effects and unexpected interactions.
    def evaluate(X):
        pass

    def objFunc(X):
        pass

    def conFunc(X)
        pass
```

Once defined, the `NewProblem` class can be seamlessly used with all optimization methods and algorithms available in UQPyL.

## Optimization

### 