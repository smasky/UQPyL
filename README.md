
# Uncertainty Quantification Python Laboratory (UQPyL)

**UQPyL:** The **Uncertainty Quantification Python Laboratory** provides comprehensive workflows tailored for **Uncertainty Quantification** and **Optimization** of computational models and their associated applications (e.g., model calibration, resource scheduling, product design). 

## Main Characteristics

1. **Comprehensive Sensitivity Analysis and Optimization Algorithm**: Implements widely used sensitivity analysis methodologies and optimization algorithms.
2. **Advanced Surrogate Modeling**: Integrates diverse surrogate models equipped to solve computationally expensive problems.
3. **Rich Application Resources**: Provides a comprehensive suite of benchmark problems and practical case studies, enabling users to get started quickly.
4. **Modular and Extensible Architecture**: Encourages and facilitates the development of novel methods or algorithms by users, aligning with our commitment to openness and collaboration. (**We appreciate and welcome contributions**)

## Quick Links

- **Website**: [UQPyL Official Site](http://www.uq-pyl.com) (**#TODO**: Needs update)
- **Source Code**: [GitHub Repository](https://github.com/smasky/UQPyL/)
- **Documentation**: [ReadTheDocs](https://uqpyl.readthedocs.io/en/latest/)(**#TODO**: Being updating )
- **Citation Info**: [UQPyL 2.0](**#TODO**: Needs update), [UQPyL 1.0](https://www.sciencedirect.com/science/article/pii/S1364815215300955)

---

## Included Methods and Algorithms

### Sensitivity Analysis

(All methods support surrogate models)
- Sobol'
- Delta Test (DT)
- Extended Fourier Amplitude Sensitivity Test (eFAST)
- Random Balance Designs - Fourier Amplitude Sensitivity Test (RBD-FAST)
- Multivariate Adaptive Regression Splines-Sensitivity Analysis (MARS-SA)
- Morris
- Regional Sensitivity Analysis (RSA)

### Optimization Algorithms

(* indicates solving computational expensive optimization problem)
- **Single Objective Optimization**: SCE-UA, ML-SCE-UA, GA, CSA, PSO, DE, ABC, ASMO*, EGO*
- **Multi-Objective Optimization**: MOEA/D, NSGA-II, RVEA, NSGA-III, MOASMO*

*Note: The library is still being updated. If you need other algorithms, please contact us.*

### Surrogate Models

- Fully Connected Neural Network (FCNN)
- Kriging (KRG)
- Gaussian Process (GP)
- Linear Regression (LR)
- Polynomial Regression (PR)
- Radial Basis Function (RBF)
- Support Vector Machine (SVM)
- Multivariate Adaptive Regression Splines (MARS)

---

## Installation

Recommended (PyPi or Conda):

```bash
pip install UQPyL
```

```bash
conda install UQPyL
```

Alternatively:

```bash
git clone https://github.com/smasky/UQPyL.git 
cd UQPyL
pip install .
```

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



