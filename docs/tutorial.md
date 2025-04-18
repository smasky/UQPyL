# Tutorial

## 🥕 Overview of UQPyL

**UQPyL** is a Python package that provides a comprehensive set of methods for supporting various workflows, including design of experiments, sensitivity analysis, optimization, data mining, and their integration. To facilitate these workflows, UQPyL is organized into several modules: a. **DoE** (Design of Experiments); b. **problems**; c. **sensibility**; d. **optimization**; e. **surrogates**. 

<br>

All methods and algorithms are implemented within a unified and consistent framework.
As a result, UQPyL enables the construction of complete workflows such as:

<figure align="center">
  <img src="./pic/workflows.svg" width="800"/>
  <figcaption>Full Workflow of UQPyL</figcaption>
</figure>

First, users should define solved problem. This problem definition acts as a interface to UQPyL.

Once the problem is defined, users can proceed through a variety of workflows:

1. **Design of Experiments (DOE):** All sampling contain `sample` function. So use `DoE.sample()` to efficiently explore the input space and generate data for analysis or modeling.

2. **Surrogate Modelling:** Based on the sampled data, surrogate models (e.g., polynomial regression, Gaussian processes) can be trained using `Surrogate.fit()`. These models provide fast approximations of the original, potentially expensive simulations.

3. **Sensitivity analysis:** use `SA.analyze()` can quantify the impact of input variables on the outputs. It can either use the original model evaluations or the trained surrogate model via `Surrogate.predict()` for efficiency.

4. **Optimization:** With the original model or surrogate in place, optimization tasks can be carried out using `Optimization.run()`, enabling design improvement or calibration.

5. **Integration:** These modules are seamlessly connected. For instance, surrogate models can be used to accelerate both sensitivity analysis and optimization. The entire process is iterative and modular, allowing for re-training or re-analysis based on new data or insights.

💡**Note:**  For detailed usage examples, please refer to the [example collections](./examples.md) provided in the documentation.

## 🌶️ Define Problem

---

### 1. Overview of Problem

**Problem** is the foundational component of UQPyL. It encapsulates the core elements required to define and solve a computational problem. Specifically, a problem instance should include:

1. **Decision Variables:** - Information about the decision dimension, variable names, ranges, and types.

2. **Problem Function:** - Python functions that maps decision variables to output values, including objectives and/or constraints. 

For convenience, the `UQPyL.problems`  provide the `Problem` class, served as a container for all essential information required to define a problem instance.

### 2. How to define an instance of `Problem`



### 3. Use `evaluate` function to replace `objFunc` and `conFunc`
Some practical problems  may be difficult to separately define the `objFunc` and `conFunc`. UQPyL recommends using `evaluate` function of the Problem class, instead.

Still, take this problem as example:

<p align="center"><img src="./pic/Problem1.svg" width=550/></p>

```python

# both computations of objective and constraint are handled within `evaluate` function.
def evaluate(X):

    # compute objective
    objs =100 * (X[:, 2] - X[:, 1]**2)**2+ 100 * (X[:, 1] - X[:, 0]**2)**2  + \
            (1 - X[:, 1])**2 + (1 - X[:, 0])**2 
    
    # compute constraint
    cons = cons = X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2 - 4 

    # `evaluate` function should return a python dict, 
    # which contain two fixed keywords: `objs` and `cons`.

    return {'objs' : objs[:, None], 'cons' : cons[:, None]} # ensure 2D numpy array

# still support single-run mode using `@singleEval`
from UQPyL.problems import singleEval

@singleEval
def evaluate(x):

    # objective
    obj = 100 * (X[2] - X[1]**2)**2 + 100 * (X[1] - X[0]**2)**2 + \
            (1 - X[1])**2 + (1 - X[0])**2 

    # constraint
    con = X[0]**2 + X[1]**2 + X[2]**2 - 4

    # return a python dict with two keywords, `objs` and `cons`
    # `obj` and `con` should be scaler value (int, float) for single-objective
    # python list or np.1d-array for multi-objective

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

# use `evaluate` keyword to replace origin `objFunc` and `conFunc`
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

💡**Note:** # Note: When calling `Problem.evaluate`, the return value is a Python dictionary. Use the keys `'objs'` and `'cons'` to access the objective values and constraints, respectively — e.g., `res['objs']`, `res['cons']`.


### 4. Implement `NewProblem` class, which inherits from `ProblemABC` base class

UQPyL allows users to customize problem-based classes by extending the built-in `Problem` class. To do so, simply inherit from the abstract base class `ProblemABC`.

```python
from UQPyL.problems import ProblemABC

class NewProblem(ProblemABC):

    def __init__(nInput, nOutput, ub, lb):

        # Initialize the base class (ProblemABC)
        # `ProblemABC` accepts more arguments 
        # than shown here—check the API docs for full usage.

        super().__init__(nInput, nOutput, ub, lb)

        pass
    
    # Override one of `evaluate`, `objFunc`, or `conFunc`.
    # Overriding all is discouraged, as it may lead to 
    # untested side effects or unexpected behavior.
    def evaluate(X):
        pass

    def objFunc(X):
        pass

    def conFunc(X)
        pass
```

Once defined, the `NewProblem` class can be seamlessly used with all optimization methods and algorithms available in UQPyL.

## 🥬 Sensibility Analysis

1. Overview of SA methods





