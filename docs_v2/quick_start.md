# Quick Start

This page shows the shortest complete path through UQPyL.

Most workflows follow the same shape:

```text
Problem / ModelProblem -> Method -> Result
```

Use this page when you want to confirm that UQPyL is installed correctly and understand how the main modules connect.

## 0. Install UQPyL

UQPyL supports Python 3.10 and later. In most cases, install it into your project environment with:

```bash
python -m pip install UQPyL
```

If you want to run examples with plots, install the visualization extras:

```bash
python -m pip install "UQPyL[viz]"
```

If you are developing UQPyL locally or want to use the latest repository code:

```bash
git clone https://github.com/smasky/UQPyL.git
cd UQPyL
python -m pip install -e .
```

Installing from source may need a compiled-extension toolchain. If the build fails, check that the environment has `Cython`, `pybind11`, `numpy`, `scipy`, and a working C/C++ compiler.

After installation, run this minimal check:

```python
import UQPyL
from UQPyL.problem import Problem
from UQPyL.doe import LHS

print("UQPyL is ready")
```

Example output:

```text
UQPyL is ready
```

Common installation issues:

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'UQPyL'` | UQPyL is not installed in the Python environment that runs your script or notebook. Run `python -m pip install UQPyL` in that same environment. |
| Import works in the terminal but not in a notebook | The notebook kernel is using a different Python environment. Switch kernels or install UQPyL into the kernel environment. |
| Source installation fails with C/C++ compiler errors | Prefer the published package when possible; otherwise install a compiler plus `Cython`, `pybind11`, and the build dependencies. |

## 1. Choose a Problem Type

UQPyL now has two parallel problem entry points:

| Type | Main flow | Use when |
|---|---|---|
| `Problem` | `X -> objs/cons` | Objectives and constraints are computed directly from inputs |
| `ModelProblem` | `X -> sim -> objs/cons` | A simulation runs first, then objectives or constraints are derived from simulation outputs |

Start with `Problem` for ordinary optimization, analysis, and inference.\
Start with `ModelProblem` for calibration, time-series simulation, or multi-series simulation workflows.

## 2. Define a `Problem`

A `Problem` defines the input bounds, objective function, and optimization direction. Other modules use this same object.

This toy problem has two inputs:

```text
y = x1^2 + 0.2*x2^2
```

```python
import numpy as np

from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.2 * X[:, 1] ** 2
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D", xLabels=["x1", "x2"])

res = problem.evaluate([[0.2, 0.3]])

print(res.objs)
print(res.cons)
```

Example output:

```text
[[0.058]]
None
```

`evaluate()` returns an `Eval` object. Use `res.objs` for objectives and `res.cons` for constraints. Here there are no constraints, so `res.cons` is `None`.

## 3. Define a `ModelProblem`

`ModelProblem` places simulation output `sim` in the middle of the evaluation chain. The recommended pattern is to define `simFunc(X)` first, then derive objectives or constraints from `context.sim`, `context.obs`, and `context.mask`.

```python
import numpy as np

from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)

obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


def objFunc(X, context):
    err = context.sim - context.obs
    return np.mean(err**2, axis=(1, 2)).reshape(-1, 1)


problem = ModelProblem(
    nInput=2,
    nObj=1,
    lb=0.0,
    ub=3.0,
    simFunc=simFunc,
    objFunc=objFunc,
    obs=obs,
    seriesLabels=["Q"],
    name="ToyModel",
)

res = problem.evaluate([[1.0, 2.2]])

print(res.objs)
print(res.sim)
```

Example output:

```text
[[0.02]]
[[[1. ]
  [2.2]]]
```

The important contract here is the full chain:

```text
X -> simFunc(X) -> context.sim -> objFunc/conFunc -> Eval
```

## 4. Generate Samples

DOE samplers read bounds from `problem`.

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.2 * X[:, 1] ** 2
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D", xLabels=["x1", "x2"])

X = LHS("classic").sample(problem, nSamples=10, seed=123)
Y = problem.evaluate(X).objs

print(X.shape)
print(Y.shape)
print(X[:3])
print(Y[:3])
```

Example output:

```text
(10, 2)
(10, 1)
[[ 0.9598  0.6464]
 [-0.2153 -0.9892]
 [ 0.3648  0.9036]]
[[1.0048]
 [0.2421]
 [0.2964]]
```

Rows stay aligned: `Y[i]` is the output for `X[i, :]`.

## 5. Run Analysis

Analysis explains how inputs affect outputs. Here `x1` should matter more because its coefficient is larger.

```python
import numpy as np

from UQPyL.analysis import RBDFAST
from UQPyL.doe import LHS
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.2 * X[:, 1] ** 2
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D", xLabels=["x1", "x2"])

X = LHS("classic").sample(problem, nSamples=256, seed=123)
Y = problem.evaluate(X).objs
result = RBDFAST(verboseFlag=False).analyze(problem, X, Y=Y)

print(result.metricNames)
print(result.getMetric("S1").values)
print(result.getMetric("S1").colLabels)
```

Example output:

```text
['S1']
[[0.951  0.0475]]
['x1', 'x2']
```

The first value belongs to `x1`, and it is much larger than the second value.

## 6. Run Optimization

Optimization searches for a good input vector. This problem is minimized near `[0, 0]`.

```python
import numpy as np

from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.2 * X[:, 1] ** 2
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D", xLabels=["x1", "x2"])

result = GA(nPop=8, maxFEs=40, maxIters=5, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False).run(problem, seed=123)

print(result.bestDecs)
print(result.bestObjs)
print(result.FEs, result.iters)
```

Example output:

```text
[[-0.0233 -0.0817]]
[[0.0019]]
40 5
```

`bestDecs` is the best decision row found. `bestObjs` is the objective value at that row.

You can also pass an initial population directly to `run()`:

```python
initialPop = np.array([
    [0.8, 0.8],
    [0.2, 0.1],
    [-0.5, 0.3],
])

result = GA(
    nPop=8,
    maxFEs=40,
    maxIters=5,
    tolerate=None,
    verboseFlag=False,
    logFlag=False,
    saveFlag=False,
).run(problem, initialPop=initialPop, seed=123)
```

`initialPop` can be a decision matrix or a `Population` object. If it contains fewer members than `nPop`, UQPyL fills the rest automatically.

## 7. Run Inference

Inference samples chains for a scalar objective or log-probability.

```python
import numpy as np

from UQPyL.inference import MH
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.2 * X[:, 1] ** 2
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D", xLabels=["x1", "x2"])

result = MH(nChains=3, warmUp=5, maxIters=30, verboseFlag=False, logFlag=False, saveFlag=False).run(problem, gamma=0.2, seed=123)

print(result.decs.shape)
print(result.acceptanceRate)
print(result.bestDecs)
print(result.bestObjs)
```

Example output:

```text
(3, 30, 2)
[0.8966 0.7586 0.9655]
[[-0.0296  0.1336]]
[[0.0044]]
```

`decs.shape` is `(n_chains, draws, n_input)`. `acceptanceRate` reports one value per chain.

## 6. Train a Surrogate

A surrogate learns a cheap prediction model from evaluated `X` and `Y`.

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.problem import Problem
from UQPyL.surrogate.rbf import RBF

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.2 * X[:, 1] ** 2
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D", xLabels=["x1", "x2"])

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
[[0.0004]
 [0.3024]]
```

The prediction at `[0.5, 0.5]` is close to the true value `0.5^2 + 0.2*0.5^2 = 0.3`.

## 7. Calibrate a Simulation Model

Calibration uses `ModelProblem`, because it compares simulations with observations.

This toy model has two parameters and two observed time steps:

```text
obs = [[1.0], [2.0]]
sim(t1) = x1
sim(t2) = x2
```

```python
import numpy as np

from UQPyL.calibration import GLUE
from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)

obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, lb=0.0, ub=3.0, simFunc=simFunc, obs=obs, seriesLabels=["Q"], name="ToyModel")
X = np.array([[1.0, 2.0], [1.0, 2.4], [0.0, 0.0]])

result = GLUE(metric="rmse", verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X, threshold=0.3)

print(result.bestDecs)
print(result.bestSim)
print(result.behavioralDecs)
```

Example output:

```text
[[1. 2.]]
[[1. 2.]]
[[1.  2. ]
 [1.  2.4]]
```

`bestDecs` is the best parameter row. `behavioralDecs` are parameter rows accepted by the GLUE threshold.

## Runtime Controls

Most runnable methods accept the same runtime controls:

| Option | Meaning |
|---|---|
| `verboseFlag` | Print terminal progress and final summary. |
| `verboseFreq` | Progress output interval. |
| `logFlag` | Write a text log when supported. |
| `saveFlag` | Save structured runtime data, usually sqlite, under `Result/`. |
| `saveFreq` | Saved snapshot interval when the method supports snapshots. |

Use `verboseFlag=False`, `logFlag=False`, and `saveFlag=False` while learning so examples stay quiet and do not create files.

## Next Steps

| Goal | Read |
|---|---|
| Understand the modeling protocol | [Problem](problem.md) |
| Generate input samples | [Design of Experiment](doe.md) |
| Analyze input effects | [Analysis](analysis.md) |
| Optimize parameters | [Optimization](optimization.md) |
| Infer parameter distributions | [Inference](inference.md) |
| Calibrate simulation models | [Calibration](calibration.md) |
| Train surrogate models | [Surrogate Modeling](surrogate.md) |
| Copy complete workflows | [Examples](examples.md) |
