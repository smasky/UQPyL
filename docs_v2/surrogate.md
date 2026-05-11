# Surrogate Modeling

The `surrogate` module trains cheap predictive models from evaluated input-output data.

Use a surrogate when the original model is expensive and you already have a table of inputs and outputs. A surrogate can help with quick prediction, validation, response-surface plotting, screening, or surrogate-assisted optimization.

The core workflow is:

```text
sample or collect X -> evaluate Y -> model.fit(X, Y) -> model.predict(Xnew)
```

## What a Surrogate Needs

A surrogate does not call your simulation by itself. It learns from data you already computed.

| Object | Shape | Meaning |
|---|---|---|
| `X` | `(n_samples, n_input)` | Input table. Each row is one evaluated parameter vector. |
| `Y` | `(n_samples, n_output)` | Output table. Each row must match the same row in `X`. |
| `Xnew` | `(n_new, n_input)` | New input rows where you want surrogate predictions. |
| `pred` | `(n_new, n_output)` | Predicted output rows. |

For one output, use a column such as `(n_samples, 1)`. Most models can also accept one-dimensional arrays and reshape them internally, but writing two-dimensional data makes examples and debugging clearer.

## Basic Fit and Predict

This example trains an `RBF` surrogate for:

```text
y = sin(2*pi*x)
```

```python
import numpy as np

from UQPyL.surrogate.rbf import RBF

np.set_printoptions(precision=4, suppress=True)

X = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
Y = np.sin(2 * np.pi * X)

model = RBF()
model.fit(X, Y)

pred = model.predict([[0.25], [0.75]])

print(X.shape, Y.shape)
print(pred.shape)
print(pred)
```

Example output:

```text
(8, 1) (8, 1)
(2, 1)
[[ 0.9989]
 [-0.9989]]
```

Read this as:

| Output | Meaning |
|---|---|
| `(8, 1)` | Eight training points and one input variable. |
| `(2, 1)` | Two prediction points and one predicted output. |
| `0.9989`, `-0.9989` | Predictions near `sin(pi/2)=1` and `sin(3*pi/2)=-1`. |

## Keep Rows Aligned

`X` and `Y` must describe the same model runs in the same order.

| Row | Input | Output |
|---:|---|---|
| `0` | `X[0, :]` | `Y[0, :]` |
| `1` | `X[1, :]` | `Y[1, :]` |
| `2` | `X[2, :]` | `Y[2, :]` |

Do not shuffle `X` and `Y` separately. If you split data into training and testing sets, apply the same indices to both arrays.

Although UQPyL can reshape simple one-dimensional data, prediction inputs are easiest to read when written as a list of rows:

```text
model.predict([[0.25], [0.75]])  # two rows, one variable
model.predict([[0.2, 0.8]])      # one row, two variables
```

## Choose a Model

Start with the question you need the surrogate to answer.

| Model | Import path | Good first use |
|---|---|---|
| `RBF` | `UQPyL.surrogate.rbf` | Default response-surface model for smooth data. |
| `GPR` | `UQPyL.surrogate.gp` | Gaussian process regression with uncertainty output. |
| `KRG` | `UQPyL.surrogate.kriging` | Kriging-style model with trend choices and uncertainty output. |
| `LinearRegression` | `UQPyL.surrogate.regression` | Baseline when the response is close to linear. |
| `PolynomialRegression` | `UQPyL.surrogate.regression` | Low-order smooth polynomial behavior. |
| `MARS` | `UQPyL.surrogate.mars` | Piecewise regression splines when optional compiled dependencies are available. |
| `SVR` | `UQPyL.surrogate.svr` | Support vector regression when optional compiled dependencies are available. |

Practical default: use `RBF` first. Use `GPR` or `KRG` when you need predictive uncertainty. Use a regression model as a baseline to check whether a simpler model is already enough.

For constructor parameters and kernels, see [Surrogate API](api/surrogate.md).

## Validate Prediction Quality

Always check predictions on held-out data before trusting a surrogate.

```python
import numpy as np

from UQPyL.surrogate import RandSelect, mse, r_square
from UQPyL.surrogate.rbf import RBF

np.set_printoptions(precision=4, suppress=True)
np.random.seed(123)

X = np.linspace(0.0, 1.0, 24).reshape(-1, 1)
Y = np.sin(2 * np.pi * X) + 0.2 * X

trainIdx, testIdx = RandSelect(pTest=25).split(X)

model = RBF()
model.fit(X[trainIdx], Y[trainIdx])

pred = model.predict(X[testIdx])

print(trainIdx.shape, testIdx.shape)
print(np.round(r_square(Y[testIdx], pred), 4))
print(np.round(mse(Y[testIdx], pred), 6))
print(pred[:3])
```

Example output:

```text
(18,) (6,)
1.0
[0.]
[[ 1.0222]
 [-0.556 ]
 [-0.3361]]
```

Metrics:

| Metric | Direction | Meaning |
|---|---|---|
| `r_square` | Higher is better; `1.0` is perfect on checked data. | Fraction of output variation explained by predictions. |
| `nse` | Higher is better; `1.0` is perfect. | Nash-Sutcliffe efficiency, often used in hydrology. |
| `mse` | Lower is better; `0.0` is perfect. | Mean squared prediction error. |
| `rank_score` | Higher is better. | Whether predictions preserve the ordering of samples. |
| `sort_score` | Lower is better. | Distance between true and predicted sorted indexes. |

The example is intentionally easy and nearly interpolated. Real expensive models usually need more samples and a less perfect validation score.

## Scale Data

Scaling is useful when variables have different units or magnitudes, such as rainfall in millimeters and a coefficient between `0` and `1`.

Pass a pair of scalers:

```text
scalers=(xScaler, yScaler)
```

The first scaler transforms inputs. The second scaler transforms outputs during training and automatically inverts predictions back to the original output scale.

```python
import numpy as np

from UQPyL.surrogate import StandardScaler
from UQPyL.surrogate.rbf import RBF

np.set_printoptions(precision=4, suppress=True)

X = np.linspace(0.0, 100.0, 12).reshape(-1, 1)
Y = np.cos(X / 20.0)

model = RBF(scalers=(StandardScaler(0, 1), StandardScaler(0, 1)))
model.fit(X, Y)

pred = model.predict([[50.0], [75.0]])

print(pred.shape)
print(pred)
print(model.xTrain.mean(axis=0), model.xTrain.std(axis=0, ddof=1))
```

Example output:

```text
(2, 1)
[[-0.801 ]
 [-0.8206]]
[0.] [1.]
```

The last line shows that the internally stored training input was standardized to mean `0` and standard deviation `1`. Predictions are still returned on the original `Y` scale.

## Predict With Uncertainty

Only models with uncertainty support can return standard deviation or variance. In the current public models, use `GPR` or `KRG` for this.

```python
import numpy as np

from UQPyL.surrogate import StandardScaler
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF as RBFKernel

np.set_printoptions(precision=4, suppress=True)

X = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
Y = X**2

model = GPR(scalers=(StandardScaler(0, 1), StandardScaler(0, 1)), kernel=RBFKernel(), nRestartTimes=1)
model.rng = np.random.default_rng(123)
model.fit(X, Y)

mean, std = model.predict([[0.3], [0.7]], returnStd=True)

print(mean.shape, std.shape)
print(mean)
print(std)
```

Example output:

```text
(2, 1) (2, 1)
[[0.1752]
 [0.575 ]]
[[0.]
 [0.]]
```

Use:

```text
model.predict(Xnew)                 -> mean only
model.predict(Xnew, returnStd=True) -> mean, standard deviation
model.predict(Xnew, returnVar=True) -> mean, variance
```

`returnStd=True` and `returnVar=True` cannot both be true. Calling them on a model such as `RBF` raises an error because `RBF` does not provide uncertainty output.

## Multi-Output Surrogates

For multiple outputs, use `MultiSurrogate` to train one model per output column.

```python
import numpy as np

from UQPyL.surrogate import MultiSurrogate
from UQPyL.surrogate.rbf import RBF

np.set_printoptions(precision=4, suppress=True)

X = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
Y = np.hstack([np.sin(2 * np.pi * X), X**2])

model = MultiSurrogate(2, models_list=[RBF(), RBF()])
model.fit(X, Y)

pred = model.predict([[0.25], [0.75]])

print(Y.shape)
print(pred.shape)
print(pred)
```

Example output:

```text
(10, 2)
(2, 2)
[[ 0.9996  0.0626]
 [-0.9996  0.5626]]
```

`n_surrogates` must match `Y.shape[1]`, and the number of models in `models_list` must match `n_surrogates`.

## Tune Hyper-Parameters

`AutoTuner` searches parameter values with a validation split. `gridTune()` is a good starting point because it is explicit and easy to understand.

```python
import numpy as np

from UQPyL.surrogate import AutoTuner, StandardScaler
from UQPyL.surrogate.rbf import RBF

np.set_printoptions(precision=4, suppress=True)
np.random.seed(123)

X = np.linspace(0.0, 1.0, 16).reshape(-1, 1)
Y = X**2 + 0.1

model = RBF(
    scalers=(StandardScaler(0, 1), StandardScaler(0, 1)),
    C_smooth_attr={"ub": 0.1, "lb": 0.0, "type": "float", "log": False},
)
tuner = AutoTuner(model=model)

bestParams, bestScore = tuner.gridTune(X, Y, paraGrid={"C_smooth": [0.0, 1e-6, 1e-4]}, ratio=25)

print(bestParams)
print(round(float(bestScore), 4))
print(model.predict([[0.5]]))
print(model.getParameterValues("C_smooth"))
```

Example output:

```text
0.0
0.9996
[[0.35]]
0.0
```

After tuning, `AutoTuner` applies the best parameter values to `model` and refits it on the full dataset.

Here `bestScore` is an `r_square` validation score. The exact split depends on NumPy's random state, so set `np.random.seed(...)` when you need reproducible tuning examples.

## Use Data From a `Problem`

Surrogate training data often comes from DOE samples evaluated by a `Problem`.

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.problem import Problem
from UQPyL.surrogate.rbf import RBF

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.5 * X[:, 1] ** 2
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D")

X = LHS("classic").sample(problem, nSamples=20, seed=123)
Y = problem.evaluate(X).objs

model = RBF()
model.fit(X, Y)

pred = model.predict([[0.0, 0.0], [0.5, 0.5]])

print(X.shape, Y.shape)
print(pred)
```

Example output:

```text
(20, 2) (20, 1)
[[-0.0001]
 [ 0.3749]]
```

This is the usual bridge from expensive real evaluations to a cheap surrogate model.

## Use a Surrogate Inside Optimization

After fitting, `model.predict()` can be wrapped as a new `Problem`. This lets an optimizer search the surrogate cheaply.

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem
from UQPyL.surrogate.rbf import RBF

np.set_printoptions(precision=4, suppress=True)


def realObjFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.5 * X[:, 1] ** 2
    return y.reshape(-1, 1)


realProblem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=realObjFunc, optType="min", name="RealModel")
X = LHS("classic").sample(realProblem, nSamples=20, seed=123)
Y = realProblem.evaluate(X).objs

model = RBF()
model.fit(X, Y)


def surrogateObjFunc(X):
    return model.predict(X)


surrogateProblem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=surrogateObjFunc, optType="min", name="SurrogateModel")
result = GA(nPop=8, maxFEs=32, maxIters=4, verboseFlag=False, logFlag=False, saveFlag=False).run(surrogateProblem, seed=123)

print(result.bestDecs)
print(result.bestObjs)
print(realProblem.evaluate(result.bestDecs).objs)
```

Example output:

```text
[[-0.0233 -0.0817]]
[[0.0039]]
[[0.0039]]
```

The surrogate objective is cheap, but it is still an approximation. Always evaluate the final candidate with the real `Problem`, as shown in the last line.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Training with mismatched rows in `X` and `Y`. | Keep row order aligned from sampling through evaluation and splitting. |
| Passing one row as `[0.2, 0.8]` when the model has one input variable. | For one input variable and two rows, use `[[0.2], [0.8]]`. |
| Trusting training accuracy only. | Use held-out validation or cross-validation. |
| Expecting `RBF` to return uncertainty. | Use `GPR` or `KRG` for `returnStd` or `returnVar`. |
| Forgetting that predictions are approximations. | Check important predicted candidates with the original model. |
| Scaling only `X` manually and not `Xnew`. | Prefer model scalers so training and prediction use the same transformation. |
| Using too few samples for a high-dimensional problem. | Increase DOE size or reduce input dimension before trusting the surrogate. |
| Treating a tuned validation score as universal truth. | Re-check on another split or with domain-specific test points. |

## Next Steps

| Goal | Read |
|---|---|
| Look up model constructors, kernels, scalers, and tuner APIs | [Surrogate API](api/surrogate.md) |
| Define the evaluated system | [Problem](problem.md) |
| Generate training samples | [Design of Experiment](doe.md) |
| Optimize a fitted surrogate | [Optimization](optimization.md) |
| See end-to-end examples | [Examples](examples.md) |
