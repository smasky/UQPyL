# Uncertainty Quantification Python Laboratory <br> (UQPyL)

**UQPyL:** The **Uncertainty Quantification Python Laboratory** provide comprehensive workflows tailored to the **Uncertainty Quantification** and **Optimization** for computational models and their associated applications (e.g. model calibration, resource scheduling, product design). 

The **main characteristics** of UQPyL includes:

1. Implementation of widely used sensitivity analysis methodologies and optimization algorithms.

2. Integration of diverse surrogate models equipped with tunable to solving computational expensive problems.

3. Provision of a comprehensive suite of benchmark problems and practical case studies, enabling users to quick start.

4. A modular and extensible architecture that encourages and facilitates the development of novel methods or algorithms by users, aligning with our commitment to openness and collaboration. (**We appreciate and welcome contributions**)

 **Website:** http://www.uq-pyl.com/ (**#TODO** it need to update now.) <br>
  **Source Code:** https://github.com/smasky/UQPyL/ <br> 
  **Documentation:** **#TODO** <br>
  **Citing in your work:** **#TODO** <br>

# Included Methods and Algorithms
**Sensibility Analysis:** (all methods support for surrogate models)
- Sobol'
- Delta_test (DT)
- extended Fourier Amplitude Sensitivity Test (eFAST)
- Random Balance Designs - Fourier Amplitude Sensitivity Test
- Multivariate Adaptive Regression Splines-Sensibility Analysis (MARS-SA)
- Morris
- Regional Sensitivity Analysis (RSA)

**Optimization Algorithms:** (* indicates the use of surrogate models)
- Single Objective Optimization: SCE-UA, ML-SCE-UA, GA, CSA, PSO, DE, ABC, ASMO*, EGO*
- Multi Objective Optimization: MOEA/D, NSGA-II, RVEA, MOASMO

Noted: It is still being updated, and if you need other algorithms, please contact me.

**Surrogate Models:**
- Full connect neural network (FNN)
- Kriging (KRG)
- Gaussian Process (GP)
- Linear Regression (LR)
- Polynomial Regression (PR)
- Radial Basis Function (RBF)
- Support Vector Machine (SVM)
- Multivariate Adaptive Regression Splines (MARS)

# Installation

Recommend (PyPi or Conda):

```
pip install UQPyL

conda install UQPyL
```

And also:

```
git clone https://github.com/smasky/UQPyL.git 
pip install . 
```

# Quick Start
For users, we should define the problem you want to solved firstly. The problem usually contains three important properties:
a. func (the mapping from X to Y); b. the dimensions of decisions and outputs; c. the bound of decisions (ub, lb).

For benchmark problems, we can import them from **UQPyL.problems** and instantiation:

```
from UQPyL.problems.single_objective import Sphere

problem=Sphere(nInput=10, ub=100, lb=-100)
problem=Sphere(nInput=10, ub=np.ones(10)*100, lb=np.ones(10)*-100)
```

For practical problems, we should define the evaluation function in addition, like:
```
from UQPyL.problems import PracticalProblem
def func(X):
  Y=np.sum(X, axis=1).reshape(-1, 1)
  return Y

problem=PracticalProblem(func=func, ub=100, lb=-100, name="Sphere")
```
**Please noted that**, the func need receive the matrix of X and return the matrix of Y. Columns=Dimensions / Rows=Samples

After defining problem you solved, you can use any methods in UQPyL.
**Sensibility:**
```
from UQPyL.sensibility import Sobol

sobol=Sobol() #instantiation and set hyper-parameters
sobol.analyze(problem)
```

**Optimization:**
```
from UQPyL.optimization.single_objective import SCE_UA

sce=SCE_UA()
res=sce.run(problem)
bestDec=res.bestDec; bestObj=res.bestObj
```

**Surrogate:**
```
from UQPyL.DoE import LHS

lhs=LHS(problem)
xTrain=lhs.sample(200, problem.nInput)
yTrain=problem.evaluate(xTrain)

xTest=lhs.sample(50, problem.nInput)
yTest=problem.evaluate(xTest)

from UQPyL.surrogate.rbf import RBF

rbf=RBF()
rbf.fit(xTrain, yTrain)
yPred=rbf.predict(xTest)

from UQPyL.utility.metric import r_square
r2=r_square(yTest, yPred)
```

The above is a quick start. For more advanced usage, please refer to the documentation (#TODO).

# Call for Contributions
We appreciate and welcome contributions. Because, we only set up standard workflows here. More advanced quantification methods and optimization algorithms are waited for pulling to this project.

---
# Contact:

wmtSky, <wmtsky@hhu.edu.cn> 





