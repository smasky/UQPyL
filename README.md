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
- SCE-UA
- Genetic Algorithm (GA)
- Non-dominated Sorting Genetic Algorithm-II (NSGA-II)
- AMSMO*
- MO_ASMO*
- MASTO* #TODO
- AMSMO* #TODO

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


# Call for Contributions
We appreciate and welcome contributions. Because, we only set up standard workflows here. More advanced quantification methods and optimization algorithms are waited for pulling to this project.

---
# Contact:

wmtSky, <wmtsky@hhu.edu.cn> 





