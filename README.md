
# UQPyL:  Uncertainty Quantification Python Lab
<p align="center"><img src="./docs/UQ.svg" width="400"/></p>

[![PyPI version](https://badge.fury.io/py/UQPyL.svg?icon=si%3Apython&icon_color=%2331aadd)](https://badge.fury.io/py/UQPyL) ![PyPI - Downloads](https://img.shields.io/pypi/dm/UQPyL) ![PyPI - License](https://img.shields.io/pypi/l/UQPyL) ![GitHub last commit](https://img.shields.io/github/last-commit/smasky/UQPyL) ![Static Badge](https://img.shields.io/badge/Author-wmtSky-orange) ![Static Badge](https://img.shields.io/badge/Contact-wmtsmasky%40gmail.com-blue)

**UQPyL** is a Python package for **Uncertainty Quantification** and **Optimization** of computational models and their associated problems (e.g., model calibration, resource scheduling, product design). It includes a wide range of methods and algorithms for Design of Experiments, Sensitivity Analysis, Optimization (Single- and Multi-objective). Additionally, **Surrogate Models** are built-in for solving computationally expensive problems.

## Main Features
1. **Comprehensive Sensitivity Analysis and Optimization**: Implements widely used sensitivity analysis methodologies and optimization algorithms.
2. **Running Display and Result Save**: Enable users to track and save the history and results of their running.
3. **Advanced Surrogate Modeling**: Integrates diverse surrogate models and auto-tunning technique to enhance these model performances.
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

## Overview of Methods and Algorithms

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

🎉 **All above methods support for using surrogate models**

### Optimization Algorithms

(* indicates solving computational expensive optimization problem)
- **Single Objective Optimization**: SCE-UA, ML-SCE-UA, GA, CSA, PSO, DE, ABC, ASMO*, EGO*
- **Multi-Objective Optimization**: MOEA/D, NSGA-II, RVEA, NSGA-III, MOASMO*

👀 **This modular is still being updated. If you need other algorithms, please contact us**

### Surrogate Models

- Kriging (KRG)
- Gaussian Process (GP)
- Linear Regression (LR)
- Polynomial Regression (PR)
- Radial Basis Function (RBF)
- Support Vector Machine (SVM)
- Multivariate Adaptive Regression Splines (MARS)

---

## Quick Start

To use UQPyL, define the problem you want to solve. The problem usually contains three important properties:
1. `func` (the mapping from X to Y)
2. The dimensions of decisions and outputs
3. The bounds of decisions (ub, lb)

### Benchmark Problems

```python
from UQPyL.problems.single_objective import Sphere

problem = Sphere(nInput=10, ub=100, lb=-100)
problem = Sphere(nInput=10, ub=np.ones(10)*100, lb=np.ones(10)*-100)
```

### Practical Problems

Define the evaluation function:

```python
from UQPyL.problems import PracticalProblem

def func(X):
    Y = np.sum(X, axis=1).reshape(-1, 1)
    return Y

problem = PracticalProblem(func=func, nInput=10, nOutput=1, ub=100, lb=-100, name="Sphere")
```

**Note:** The `func` needs to accept a matrix of X and return a matrix of Y, with columns equal to dimensions and rows equal to samples. X and Y should be np.ndarray.

After defining the problem, you can use any methods in UQPyL.

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



