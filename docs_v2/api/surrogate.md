# Surrogate API

## `UQPyL.surrogate`

The `surrogate` module trains predictive models for expensive simulations, objectives, or intermediate response surfaces.

### Import

```python
from UQPyL.surrogate.rbf import RBF
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate import MinMaxScaler, StandardScaler, KFold, AutoTuner
```

### Public Objects

Top-level objects:

| Object | Role |
|---|---|
| `SurrogateABC` | Base class for surrogate models. |
| `MultiSurrogate` | Container for one surrogate per output column. |
| `AutoTuner` | Hyper-parameter tuning helper. |
| `PolyFeature` | Polynomial feature expansion. |
| `KFold` | K-fold index splitter. |
| `RandSelect` | Random train/test splitter. |
| `Scaler` | Base scaler interface. |
| `MinMaxScaler` | Min-max scaler. |
| `StandardScaler` | Standard scaler. |

Model subpackages:

| Subpackage | Import path | Main objects |
|---|---|---|
| RBF | `UQPyL.surrogate.rbf` | `RBF`, `Cubic`, `Linear`, `Multiquadric`, `ThinPlateSpline`, `Gaussian` |
| Gaussian process | `UQPyL.surrogate.gp` | `GPR` |
| Kriging | `UQPyL.surrogate.kriging` | `KRG` |
| Regression | `UQPyL.surrogate.regression` | `LinearRegression`, `PolynomialRegression` |
| MARS | `UQPyL.surrogate.mars` | `MARS` |
| SVR | `UQPyL.surrogate.svr` | `SVR` |

`mars` and `svr` may be unavailable when optional compiled dependencies are not installed.

## Fit and Predict

All surrogate models follow the same high-level protocol:

```python
model.fit(xTrain, yTrain)
yPred = model.predict(xPred)
```

Input and output arrays are normalized to 2D internally. `yTrain` may be 1D or 2D.

Example:

```python
import numpy as np

from UQPyL.surrogate.rbf import RBF


X = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
Y = np.sin(2 * np.pi * X)

model = RBF()
model.fit(X, Y)

pred = model.predict([[0.25], [0.75]])
print(pred)
```

Uncertainty output uses the common flags:

```python
mean = model.predict(X)
mean, std = model.predict(X, returnStd=True)
mean, var = model.predict(X, returnVar=True)
```

Only models with `supportsUncertainty=True` support `returnStd` or `returnVar`.

## `SurrogateABC`

Base class for surrogate models.

```python
SurrogateABC(scalers=(None, None), polyFeature=None)
```

| Parameter | Meaning |
|---|---|
| `scalers` | Pair of optional `(xScaler, yScaler)`. |
| `polyFeature` | Optional `PolyFeature` applied after input scaling. |

Common methods:

| Method | Returns | Meaning |
|---|---|---|
| `fit(xTrain, yTrain)` | model | Prepare data, fit hyper-parameters, and fit the model. |
| `predict(xPred, returnStd=False, returnVar=False)` | `np.ndarray` or tuple | Predict output values, optionally with uncertainty. |
| `prepareTrainingData(xTrain, yTrain)` | `(X, Y)` | Validate, scale, and transform training data. |
| `storeTrainingData(xTrain, yTrain)` | model | Store prepared training data. |
| `requireFitted(*stateKeys)` | model | Raise if the model is not fitted. |
| `getParaList()` | `list` | Return active tunable parameter names. |
| `applyParameterValues(paraList, values, ignoreInactive=True)` | model | Apply tuned parameter values. |
| `getParameterValues(*args, ignoreInactive=False)` | value or tuple | Return current parameter values. |

## `MultiSurrogate`

Container for multi-output surrogate prediction.

```python
MultiSurrogate(n_surrogates, models_list=[])
```

| Method | Returns | Meaning |
|---|---|---|
| `append(model)` | `None` | Append one `SurrogateABC` model. |
| `fit(trainX, trainY)` | `None` | Fit one model per output column. |
| `predict(testX)` | `np.ndarray` | Horizontally stack predictions from all models. |

`trainY.shape[1]` and `len(models_list)` must match `n_surrogates`.

## Models

### `RBF`

Radial basis function surrogate.

```python
RBF(
    scalers=(None, None),
    polyFeature=None,
    kernel=Cubic(),
    C_smooth=0.0,
    C_smooth_attr={...},
)
```

| Parameter | Meaning |
|---|---|
| `kernel` | RBF kernel object. |
| `C_smooth` | Smoothing parameter. |
| `C_smooth_attr` | Tuning metadata for `C_smooth`. |

Additional methods:

| Method | Meaning |
|---|---|
| `setKernel(kernel)` | Replace the active kernel. |
| `setKernelChoices(kernels)` | Register tunable kernel choices. |

Available RBF kernels:

| Kernel |
|---|
| `Cubic` |
| `Linear` |
| `Multiquadric` |
| `ThinPlateSpline` |
| `Gaussian` |

### `GPR`

Gaussian process regression surrogate.

```python
GPR(
    scalers=(None, None),
    polyFeature=None,
    kernel=RBFKernel(),
    optimizer="Boxmin",
    nRestartTimes=5,
    C=1e-9,
    C_attr={...},
)
```

| Parameter | Meaning |
|---|---|
| `kernel` | Gaussian process kernel. |
| `optimizer` | Hyper-parameter optimizer. |
| `nRestartTimes` | Number of hyper-parameter optimization restarts. |
| `C` | Numerical regularization parameter. |
| `C_attr` | Tuning metadata for `C`. |

Additional methods:

| Method | Meaning |
|---|---|
| `setKernel(kernel)` | Replace the active kernel. |
| `setKernelChoices(kernels)` | Register tunable kernel choices. |

`GPR` supports uncertainty output.

### `KRG`

Kriging surrogate.

```python
KRG(
    scalers=(None, None),
    polyFeature=None,
    kernel=Guass(),
    regression="poly0",
    optimizer="Boxmin",
    nRestartTimes=5,
)
```

| Parameter | Meaning |
|---|---|
| `kernel` | Kriging correlation kernel. |
| `regression` | Regression trend. One of `"poly0"`, `"poly1"`, or `"poly2"`. |
| `optimizer` | Hyper-parameter optimizer. |
| `nRestartTimes` | Number of hyper-parameter optimization restarts. |

Additional methods:

| Method | Meaning |
|---|---|
| `setKernel(kernel)` | Replace the active kernel. |
| `setKernelChoices(kernels)` | Register tunable kernel choices. |

`KRG` supports uncertainty output.

### `LinearRegression`

Linear regression surrogate.

```python
LinearRegression(
    scalers=(None, None),
    polyFeature=None,
    lossType="Origin",
    fitIntercept=True,
    C=0.1,
    C_attr={...},
    maxIter=100,
    maxEpoch=500000.0,
    tolerance=0.001,
    p0=10,
)
```

| Parameter | Meaning |
|---|---|
| `lossType` | Regression loss. One of `"Origin"`, `"Ridge"`, or `"Lasso"`. |
| `fitIntercept` | Whether to fit an intercept term. |
| `C` | Regularization parameter. |
| `maxIter` | Maximum outer iterations. |
| `maxEpoch` | Maximum Lasso epochs. |
| `tolerance` | Convergence tolerance. |
| `p0` | Lasso working-set parameter. |

### `PolynomialRegression`

Polynomial regression surrogate.

```python
PolynomialRegression(
    scalers=(None, None),
    degree=2,
    degree_attr={...},
    onlyInteraction=False,
    lossType="Origin",
    fitIntercept=True,
    C=0.1,
    C_attr={...},
    maxIter=100,
    maxEpoch=500000.0,
    tolerance=0.001,
    p0=10,
)
```

| Parameter | Meaning |
|---|---|
| `degree` | Polynomial degree. |
| `onlyInteraction` | Whether to use interaction-only polynomial terms. |
| `lossType` | Regression loss. One of `"Origin"`, `"Ridge"`, or `"Lasso"`. |

### `MARS`

Multivariate adaptive regression splines surrogate.

```python
MARS(
    scalers=(None, None),
    polyFeature=None,
    max_terms=400,
    max_degree=1,
    penalty=3.0,
    endspan_alpha=0.05,
    endspan=-1,
    minspan_alpha=0.05,
    minspan=-1,
    thresh=0.001,
    zero_tol=1e-12,
    min_search_points=100,
    check_every=-1,
    allow_linear=True,
    use_fast=False,
    fast_K=5,
    fast_h=1,
    smooth=False,
    enable_pruning=True,
    feature_importance_type="gcv",
)
```

| Parameter | Meaning |
|---|---|
| `max_terms` | Maximum number of basis terms. |
| `max_degree` | Maximum interaction degree. |
| `penalty` | Generalized cross-validation penalty. |
| `enable_pruning` | Whether to prune basis terms. |
| `feature_importance_type` | Feature-importance calculation mode. |

### `SVR`

Support vector regression surrogate.

```python
SVR(
    scalers=(None, None),
    polyFeature=None,
    symbol="epsilon-SVR",
    kernel="rbf",
    nu=0.5,
    C=0.1,
    epsilon=0.1,
    gamma=1.0,
    coe0=0.1,
    degree=3,
    maxIter=100000.0,
    eps=0.001,
)
```

| Parameter | Meaning |
|---|---|
| `symbol` | SVR type. One of `"epsilon-SVR"` or `"nu-SVR"`. |
| `kernel` | Kernel. One of `"linear"`, `"rbf"`, `"sigmoid"`, or `"polynomial"`. |
| `nu` | Nu-SVR parameter. |
| `C` | Regularization parameter. |
| `epsilon` | Epsilon-SVR tube width. |
| `gamma` | Kernel gamma. |
| `coe0` | Kernel coefficient. |
| `degree` | Polynomial kernel degree. |
| `maxIter` | Maximum solver iterations. |
| `eps` | Solver tolerance. |

## Scaling and Features

### `MinMaxScaler`

```python
MinMaxScaler(min_=0, max_=1)
```

Maps each feature to `[min_, max_]`.

### `StandardScaler`

```python
StandardScaler(muX=0, sitaX=1)
```

Maps each feature to the requested mean and standard deviation scale.

Scaler methods:

| Method | Meaning |
|---|---|
| `fit(trainX)` | Fit scaler statistics. |
| `transform(trainX)` | Transform data. |
| `fit_transform(trainX)` | Fit and transform data. |
| `inverse_transform(trainX)` | Reverse the transformation. |

### `PolyFeature`

```python
PolyFeature(degree=2, includeBias=False, onlyInteraction=False)
```

| Method | Returns | Meaning |
|---|---|---|
| `transform(trainX)` | `np.ndarray` | Expand polynomial features. |

## Splitting and Metrics

### `KFold`

```python
KFold(n_splits=5)
```

| Method | Returns | Meaning |
|---|---|---|
| `split(X, mode="full")` | `(train, test)` | Return fold indices. `mode` is `"full"` or `"single"`. |

### `RandSelect`

```python
RandSelect(pTest=5)
```

| Method | Returns | Meaning |
|---|---|---|
| `split(X)` | `(train, test)` | Return one random train/test split. `pTest` is a percentage. |

Metrics:

| Function | Meaning |
|---|---|
| `r_square(true_Y, pre_Y)` | R-squared score. |
| `nse(true_Y, pre_Y)` | Nash-Sutcliffe efficiency. |
| `mse(true_Y, pre_Y)` | Mean squared error. |
| `rank_score(true_Y, pre_Y)` | Kendall-style rank score. |
| `sort_score(true_Y, pre_Y)` | Sorted-index distance score. |

## `AutoTuner`

Hyper-parameter tuning helper for surrogate models.

```python
AutoTuner(model, optimizer=None)
```

| Method | Returns | Meaning |
|---|---|---|
| `optTune(xData, yData, paraList=None, owner=None, obj="mse", split=None, tuneMode="fit")` | model | Tune parameters with an optimizer. |
| `gridTune(xData, yData, paraGrid=None, obj="mse", split=None, tuneMode="fit")` | model | Tune parameters by grid search. |

`paraList` defaults to active model parameters. `paraGrid` maps parameter names to candidate values.
