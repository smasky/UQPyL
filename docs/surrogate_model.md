# Surrogate Model

---

## What is the Surrogate Model

**Surrogate Model** is the approximate model used to mimic the behavior of a more complex and expensive-to-evaluate function or model. In sensitivity analysis and optimization, surrogate models are often employed to reduce the number of costly evaluations (e.g., simulations, real-world experiments) by providing fast predictions of objective or constraint values. 

<br>

Surrogate models include Gaussian Processes (or called Kriging), Radial Basis Function Networks (RBF), Support Vector Regression (SVR), and so on. The common characteristics are:

- **Fast prediction speed** – once trained, surrogate models can evaluate new inputs quickly compared to the original expensive function.

- **Good approximation capability** – they can capture complex, nonlinear relationships from a limited set of data samples.

- **Uncertainty estimation** – some models (Kriging) provide not only predictions but also confidence intervals, which is useful for exploration-exploitation trade-offs in optimization.

Overall, surrogate models show great potential in expensive-to-evaluate problems. And involving surrogate modelling has become the most distinctive and important features of UQPyL.

## Overview of `UQPyL.surrogates`

The `surrogates` module provides a collection of carefully implemented surrogate models designed to seamlessly support sensitivity analysis and parameter optimization tasks:

| Abbreviation | Full Name | Features |
|--------------|-----------|----------|
| KRG | Kriging | Support `guass`, `cubic`, `exp` kernel functions |
| GP | Gaussian Process | Support `const`, `rbf`, `dot`, `matern`, `rq` kernel functions |
| LR | Linear Regression | Support `origin`, `ridge`, `lasso` loss functions|
| PR | Polynomial Regression | Support `origin`, `ridge`, `lasso` loss functions|
| RBF | Radial Basis Function |Support `cubic`, `guass`, `linear`, `mq`, `tps` kernel functions and their corresponding hyper-parameters|
| SVM | Support Vector Machine | Use [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) as the core library |
| MARS | Multivariate Adaptive Regression Splines | Use [Earth](http://www.milbo.users.sonic.net/earth/) package as the core library |

All models above inherits from the `surrogateABC` class. In this class, the `fit` and `predict` functions are fixed. Now, we give its API reference:

<br>

**class `surrogateABC`:**
```
__init__

- Description:

Initializes a surrogate model.

Note: there also exist two parts of hyper-parameters: fixed and special parameters.

- Fixed parameters for all models:
    - scalers (Tuple(Scaler, Scaler)) : A Python tuple specifying the scalers to be used for normalization. If provided, the method will normalize the input `X` and output `Y` during analysis. Each `Scaler` should be an instance of the class from `UQPyL.utility`. `scalers[0]` is used for normalizing `X`, and `scalers[1]` is used for normalizing `Y`.

    - polyFeature (PolynomialFeatures) : A class from UQPyL.utility, used to perform polynomial transform to `X`.

Other hyper-parameters vary between different models. Please check API reference.
```

**Method**

```
fit

- Description:
Accepts training data to build the surrogate model. This method fits the model to the given inputs and corresponding outputs, allowing it to approximate the underlying function or model.

- Parameters:
    - xTrain(np.2darray) : A 2D Numpy array for input decisions
    - yTrain(np.2darray) : A 2D Numpy array for output values corresponding to each decisions. `Y` should have only one column, although `Y` belong to 2D Numpy.
```

```
predict

- Description:
Accepts input data `X` and returns the predicted output values based on the trained surrogate model.

- Parameters:  
  - xPred(np.ndarray):  
    A 2D NumPy array of shape (n_samples, n_features), where each row represents an input sample for which predictions are to be made.

- Returns:  
  - `np.ndarray`:  
    A 2D NumPy array of shape (n_samples, 1), containing the predicted output values corresponding to each input sample in `X`.
```

## How to import surrogate model

Each surrogate model in UQPyL is implemented in a separate submodule for better modularity and ease of use.

<br>

Take the Radial Basis Function as example:
```python
# Import the Radial Basis Function surrogate model
from UQPyL.surrogates.rbf import RBF

# Import the Cubic kernel used by the RBF model
from UQPyL.surrogates.rbf.kernel import Cubic
# Optional kernels: Cubic, Guassian, Linear, Multiquadric, ThinPlateSpline

# Create an instance of the RBF surrogate model with the Cubic kernel
rbf = RBF(kernel = Cubic())

# Note: `Cubic` is a Python class and must be instantiated before being passed as a kernel
```

💡 **Noted:** Except RBF model, the Gaussian Process, Kriging model also contain kernel functions. And the usage of these kernel are same with RBF model. Each kernel class contain hyper-parameters, please check API reference.

## Fitting and predicting with surrogate models

Use an RBF model with a cubic kernel to predict the output of the Sphere function.

```python
from UQPyL.problems.single_objective import Sphere

# Define the problem: Sphere function with 15 inputs
problem = Sphere(nInput = 15, ub = 100, lb = -100)

# Generate training data using Latin Hypercube Sampling (LHS)
from UQPyL.DoE import LHS
lhs = LHS()

# 300 training samples
trainX = lhs.sample(nt = 300, problem = problem)

# Evaluate training outputs
trainY = problem.objFunc(trainX)

# Generate testing data
predictX = lhs.sample(nt = 50, problem = problem)

# Create and configure the RBF surrogate model
from UQPyL.surrogates.rbf import RBF
from UQPyL.surrogate.rbf.kernel import Cubic

# Instantiate a Cubic kernel
kernel = Cubic()

# Instantiate RBF model with Cubic kernel
model = RBF(kernel = kernel)

# Train the surrogate model
model.fit(trainX, trainY)

# Predict outputs for new inputs
predictY = model.predict(predictX)

# Evaluate model performance using R² metric
from UQPyL.utility.metric import R_square

R2 = R_square(trainY, predictY)

print(R2)
```

## Use auto-tuner tool to build the surrogate model

The construction of surrogate models is a crucial step prior to their deployment. To facilitate this, UQPyL offers an auto-tuning utility named `AutoTuner`, available in the `UQPyL.surrogates` module. This tool assists in identifying optimal hyperparameter settings for surrogate models.

<br>

`AutoTuner` supports two search strategies for identifying hyper-parameters:

- **Grid Search**  
   This method evaluates all possible combinations of specified hyper-parameters.  
   It is simple, reliable, and time-efficient for small search spaces, but it may not always locate the global optimum.

- **Evolution Search**  
   This approach uses evolutionary algorithms to explore the hyperparameter space.  
   It is more suitable for complex or large search spaces and has a higher chance of finding global optima.  
   However, it can be computationally more demanding than Grid Search.











