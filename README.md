
# UQPyL:  Uncertainty Quantification Python Lab
<p align="center"><img src="./docs/UQ.svg" width="400"/></p>

[![PyPI version](https://badge.fury.io/py/UQPyL.svg?icon=si%3Apython&icon_color=%2331aadd)](https://badge.fury.io/py/UQPyL) ![PyPI - Downloads](https://img.shields.io/pypi/dm/UQPyL) ![PyPI - License](https://img.shields.io/pypi/l/UQPyL) ![GitHub last commit](https://img.shields.io/github/last-commit/smasky/UQPyL) ![Static Badge](https://img.shields.io/badge/Author-wmtSky-orange) ![Static Badge](https://img.shields.io/badge/Contact-wmtsmasky%40gmail.com-blue)

**UQPyL** is a Python package for **Uncertainty Quantification** and **Optimization** of computational models and their associated problems (e.g., model calibration, resource scheduling, product design). It includes a wide range of methods and algorithms for Design of Experiments, Sensitivity Analysis, Optimization (Single- and Multi-objective). Additionally, **Surrogate Models** are built-in for solving computationally expensive problems.

## Main Features
1. **Comprehensive Sensitivity Analysis and Optimization**: Implements widely used sensitivity analysis methodologies and optimization algorithms.
2. **Running Display and Result Save**: Enable users to track and save the history and results of their running.
3. **Advanced Surrogate Modeling**: Integrates diverse surrogate models and auto-tunning tool to enhance these model performances.
4. **Rich Application Resources**: Provides a comprehensive suite of benchmark problems and practical case studies, enabling users to get started quickly. (👉**Recent Planing:** For water science research, we plan to customize the interface to integrate water-related models with UQPyL, enhancing usability and functionality, like [SWAT-UQ](https://github.com/smasky/SWAT-UQ). So, **if you have interest, please contact us to collaborate.**).
5. **Modular and Extensible Architecture**: Encourages and facilitates the development of novel methods or algorithms by users, aligning with our commitment to openness and collaboration. (**We appreciate and welcome contributions**)

## Installation

 ![Static Badge](https://img.shields.io/badge/Python-3.6%2C%203.7%2C%203.8%2C%203.9%2C%203.10%2C%203.11%2C%203.12-blue) ![Static Badge](https://img.shields.io/badge/OS-Windows%2C%20Linux-orange)

**Recommended (PyPi or Conda):**

```bash
pip install -U UQPyL
```

```bash
conda install UQPyL --upgrade
```

Alternatively:

```bash
git clone https://github.com/smasky/UQPyL.git 
cd UQPyL
pip install .
```

## Useful Links

- **Website**: [UQPyL Official Site](http://www.uq-pyl.com) (**#TODO**: Needs update)
- **Source Code**: [GitHub Repository](https://github.com/smasky/UQPyL/)
- **Documentation**: [ReadTheDocs](https://uqpyl.readthedocs.io/en/latest/) (**#TODO**: Being updating )
- **Citation Infos**: [UQPyL 2.0](**#TODO**: Needs update), [UQPyL 1.0](https://www.sciencedirect.com/science/article/pii/S1364815215300955)

---

## Overview of Methods, Algorithms and Problems

### Sensitivity Analysis

| Abbreviation | Full Name | References |
| -------|------------|----------|
| Sobol' | \ |[Sobol(2010)](https://www.sciencedirect.com/science/article/pii/S0378475400002706), [Saltelli (2002)](https://www.sciencedirect.com/science/article/pii/S0010465502002801)|
| DT| Delta Test| [Eirola et al. (2008)](https://www.semanticscholar.org/paper/Using-the-Delta-Test-for-Variable-Selection-Eirola-Liiti%C3%A4inen/fa131898bbd99e848e706837f4072a310e1109e5?p2df)|
| FAST | Fourier Amplitude Sensitivity Test | [Cukier et al. (1973)](https://pubs.aip.org/aip/jcp/article-abstract/59/8/3873/533535/Study-of-the-sensitivity-of-coupled-reaction), [Saltelli et al. (1999)](https://amstat.tandfonline.com/doi/abs/10.1080/00401706.1999.10485594)|
| RBD-FAST| Random Balance Designs Fourier Amplitude Sensitivity Test | [Tarantola et al. (2006)](https://www.sciencedirect.com/science/article/pii/S0951832005001444), [Tissot, Prieur (2012)](https://www.sciencedirect.com/science/article/pii/S0951832012001159)
|MARS-SA|  Multivariate Adaptive Regression Splines for Sensibility Analysis |[Friedman, (1991)](https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-1/Multivariate-Adaptive-Regression-Splines/10.1214/aos/1176347963.full)|
|Morris| \ |[Morris, (2012)](https://www.tandfonline.com/doi/abs/10.1080/00401706.1991.10484804)|
|RSA| Regional Sensitivity Analysis | [Hornberger, Spear, (1981)](https://www.osti.gov/biblio/6396608), [Pianosi (2016)](https://www.sciencedirect.com/science/article/pii/S1364815216300287) |  

🎉 **some ideas of above methods refer to [SALib](https://github.com/SALib/SALib).** and 🚀 **All methods now support for using surrogate models.**

### Optimization Algorithms

| Abbreviation | Full Name |   Optimization Label   |   References  |
|--------------|-----------| ----------|---------------|
| SCE-UA | Shuffled Complex Evolution| Single | [Duan et al. (1992)](https://link.springer.com/article/10.1007/BF00939380)|
| ML-SCE-UA| M&L Shuffled Complex Evolution| Single | [Muttil, Liong (2006)](https://www.worldscientific.com/doi/abs/10.1142/9789812707208_0036) |
| GA | Genetic Algorithm| Single | [Holland (1992)](https://direct.mit.edu/books/monograph/2574/Adaptation-in-Natural-and-Artificial-SystemsAn)|
| CSA | Cooperation Search Algorithm | Single | [Feng et al. (2021)](https://www.sciencedirect.com/science/article/pii/S1568494620306724) |
| PSO | Particle Swarm Optimization | Single | [Kennedy and Eberhart (1995)](https://ieeexplore.ieee.org/abstract/document/488968/) |
| DE | Differential Evolution | Single | [Storn and Price (1997)](https://link.springer.com/article/10.1023/a:1008202821328) |
| ABC |Artificial Bee Colony | Single | [Karaboga (2005)](https://abc.erciyes.edu.tr/pub/tr06_2005.pdf) |
| ASMO | Adaptive Surrogate Modelling based Optimization | Single, Surrogate | [Wang et al.(2014)](https://www.sciencedirect.com/science/article/pii/S1364815214001698) |
| EGO | Efficient Global Optimization | Single, Surrogate | [Jones (1998)](https://link.springer.com/article/10.1023/A:1008306431147)
| MOEA/D | Multi-objective Evolutionary Algorithm based on Decomposition | Multiple | [Zhang, Li (2007)](https://ieeexplore.ieee.org/document/4358754)|
| NSGA-II| Nondominated Sorting Genetic Algorithm II | Multiple | [Deb et al. (2002)](https://ieeexplore.ieee.org/document/996017)|
| NSGA-III| Nondominated Sorting Genetic Algorithm III| Multiple | [Deb, Jain (2014)](https://ieeexplore.ieee.org/document/6600851)|
| RVEA | Reference Vector guided Evolutionary Algorithm | Multiple | [Cheng et al. (2016)](https://ieeexplore.ieee.org/document/7386636)|
|MO-ASMO|Multi-Objective Adaptive Surrogate Modelling-based Optimization| Multiple, Surrogate | [Gong et al. (2015)](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2015WR018230)|  

(The label `Surrogate` indicates solving computational expensive optimization problem)

👀 **This modular is still being updated. If you need other algorithms, please contact us**

### Surrogate Models

| Abbreviation | Full Name | Features |
|--------------|-----------|----------|
| KRG | Kriging | Support `guass`, `cubic`, `exp` kernel functions |
| GP | Gaussian Process | Support `const`, `rbf`, `dot`, `matern`, `rq` kernel functions |
| LR | Linear Regression | Support `origin`, `ridge`, `lasso` loss functions|
| PR | Polynomial Regression | Support `origin`, `ridge`, `lasso` loss functions|
| RBF | Radial Basis Function |Support `cubic`, `guass`, `linear`, `mq`, `tps` kernel functions and their corresponding hyper-parameters|
| SVM | Support Vector Machine | Use [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) as the core library |
| MARS | Multivariate Adaptive Regression Splines | Use [Earth](http://www.milbo.users.sonic.net/earth/) package as the core library |

🫡 **Here, we provide the Auto-tuning tool to optimally build surrogate models, so you don't need to worry about hyper-parameters.**  

### Single-objective Problems

| Name | Formula | Optimal Solution | Optima | 
|------|---------|------------------|--------|
|Sphere| <img src="./docs/pic/Sphere.svg" /> | ( 0, 0, 0 ... 0 ) | 0.0 |
|Schwefel_2_22| <img src="./docs/pic/Schwefel_2_22.svg" /> | ( 0, 0, 0 ... 0 ) | 0.0 |
|Schwefel_1_22| <img src="./docs/pic/Schwefel_1_22.svg" /> | ( 0, 0, 0 ... 0 ) | 0.0 |
|Schwefel_2_21| <img src="./docs/pic/Schwefel_2_21.svg" /> | ( 0, 0, 0 ... 0 ) | 0.0 |
|Schwefel_2_26 | <img src="./docs/pic/Schwefel_2_26.svg" /> | (420.9687 ... 420.9687) | -12569.5 |
| Rosenbrock | <img src="./docs/pic/Rosenbrock.svg" /> | ( 0, 0, 0 ... 0 ) | 0.0 |
| Step | <img src="./docs/pic/Step.svg" /> | ( 1, 1, 1 ... 1) | 0.0 |
| Quartic | <img src="./docs/pic/Quartic.svg" /> | ( 1, 1, 1 ... 1) | 0.0 |
| Rastrigin | <img src="./docs/pic/Rastrigin.svg" /> | ( 0, 0, 0 ... 0 ) | 0.0 |
| Ackley | <img src="./docs/pic/Ackley.svg" /> | ( 0, 0, 0 ... 0 ) | 0.0 |
| Griewank | <img src="./docs/pic/Griewank.svg" /> | ( 0, 0, 0 ... 0) | 0.0 |
| Trid | <img src="./docs/pic/Trid.svg" /> | <img src="./docs/pic/Trid_solution.svg">| `-D(D+4)(D-1)/6` |
| Bent_Cigar | <img src="./docs/pic/Bent_Cigar.svg" /> |(0, 0, 0 ... 0) | 0.0 |
| Discus | <img src="./docs/pic/Discus.svg" /> | (0, 0, 0 ... 0) | 0.0 |
| Weierstrass | <img src="./docs/pic/Weierstrass.svg" /> | (0, 0, 0 ... 0) | 0.0 |

### Multi-objective Problems

| Name | Num. of Objective | Shape of the Pareto Front | Feature |
|------|-------------------|---------------------------|---------|
| ZDT1 |         2         |           Line            | Convex  |
| ZDT2 |         2         |           Line            | Concave |
| ZDT3 |         2         |           Line            | Disconnected |
| ZDT4 |         2         |           Line            | Convex |
| ZDT6 |         2         |           Line            | Concave |
| DTLZ1 | >=3 (user define) |         Surface          | Multimodal |
| DTLZ2 | >=3 (user define) |         Surface          | Single-peaked |
| DTLZ3 | >=3 (user define) |         Surface          | Multimodal|
| DTLZ4 | >=3 (user define) |         Surface          | Multimodal|
| DTLZ5 | >=3 (user define) |         Line         | Multimodal|
| DTLZ6 | >=3 (user define) |         Line         | Multimodal|
| DTLZ7 | >=3 (user define) | Discrete Surface        | Multimodal|

### Practical Problems

**#TODO:** We are planning to incorporate some common hydrological model calibration (like SWAT, SAC...) or related water resource optimization cases into UQPyL.

---

## Quick Start

### Define User Problems
To effectively use UQPyL, the first is to define the problem you solve, which should contain following properties:  
1. The information of input decisions, e.g., the dimension, range, value type  (float, int, or discrete) of each variable.
2. The function `objFunc` from input variables `x` to output objective, i.e., how the output `obj` is obtained from the inputs `x`, which could be an analytical function, computational model, or external black-box process. If necessary, it also includes the constraint functions `concFunc`.

Following problem is a variant of the Rosenbrock function, which adds additional constraint functions ($x_1^2+x_2^2+x_3^2 \ge 4$) and changes the variable types, from the origin `continuous` and `float` to `int` ($x_2$) and `discrete` ($x_3$).

<p align="center"><img src="./docs/pic/Problem1.svg" width=500/></p>

UQPyL provide a python class named `Problem` to simplify the workflow of defining problems.

```python
# Step 1: import Problem class from UQPyL's problem module
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
    #This is a vectorized operation over the input matrix X.
    objs =100 * (X[:, 2] - X[:, 1]**2)**2+ 100 * (X[:, 1] - X[:, 0]**2)**2  + \
            (1 - X[:, 1])**2 + (1 - X[:, 0])**2 
    return objs[:, None] #keep 2-dimension matrix

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
    cons = X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2 - 4 
    return cons[:, None] #keep 2-dimension matrix

# Single Run Mode
@singleFunc
def conFunc(X):
    con = X[0]**2 + X[1]**2 + X[2]**2 - 4 
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
problem = Problem(nInput = nInput, nOutput = nOutput, objFunc = objFunc, concFunc = concFunc,
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

```python
from UQPyL.sensibility import Sobol

sobol = Sobol()  # Instantiate and set hyper-parameters
sobol.analyze(problem)
```

### Optimization

```python
from UQPyL.optimization.single_objective import SCE_UA

sce = SCE_UA()
res = sce.run(problem)
bestDec = res.bestDec
bestObj = res.bestObj
```

### Surrogate Modeling

```python
from UQPyL.DoE import LHS

lhs = LHS(problem)
xTrain = lhs.sample(200, problem.nInput)
yTrain = problem.evaluate(xTrain)

xTest = lhs.sample(50, problem.nInput)
yTest = problem.evaluate(xTest)

from UQPyL.surrogate.rbf import RBF

rbf = RBF()
rbf.fit(xTrain, yTrain)
yPred = rbf.predict(xTest)

from UQPyL.utility.metric import r_square
r2 = r_square(yTest, yPred)
```

For more advanced usage, please refer to the documentation (**#TODO**).

---

## Call for Contributions

We welcome contributions to expand our library with more sophisticated UQ methods, optimization algorithms and engineering problems.

---

## Contact

For any inquiries or contributions, please contact:

**wmtSky**  
Email: [wmtsky@hhu.edu.cn](mailto:wmtsky@hhu.edu.cn), [wmtsmasky@gmail.com](mailto:wmtsmasky@gmail.com)

---

*This project is licensed under the MIT License - see the [LICENSE](https://github.com/smasky/UQPyL/LICENSE) file for details.*

![GitHub Stars](https://img.shields.io/github/stars/smasky/UQPyL?style=social)
![GitHub Forks](https://img.shields.io/github/forks/smasky/UQPyL?style=social)



