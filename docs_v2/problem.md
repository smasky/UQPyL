# Problem

The `problem` module is the modeling core of UQPyL. Sampling, analysis, optimization, inference, calibration, and surrogate workflows all start from a `Problem`-like object.

Use this page when you need to turn your mathematical model, simulation model, or objective function into something UQPyL can evaluate.

## What You Need to Define

Most users need to answer four questions:

| Question | Where it goes |
|---|---|
| What are the input variables? | `nInput`, `lb`, `ub`, optional `xLabels` |
| What does one model evaluation compute? | `objFunc`, `conFunc`, `simFunc`, or `evaluate` |
| Is the objective minimized or maximized? | `optType` |
| Is this a direct objective problem or a simulation-with-observations problem? | `Problem` or `ModelProblem` |

Use `Problem` for ordinary objectives and constraints. Use `ModelProblem` when the main output is a simulation time series or multi-series simulation, especially for calibration.

## Start With a Simple `Problem`

This example defines a two-variable objective:

```text
f(x) = x1^2 + x2^2
```

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, ub=1.0, lb=-1.0, objFunc=objFunc, optType="min", name="Sphere2D")

res = problem.evaluate([[0.2, 0.3]])
print(res.objs)
```

Example output:

```text
[[0.13]]
```

The important contract is:

```text
input X -> objFunc(X) -> objective values
```

## Define the Input Space

`nInput` is the number of input variables. `lb` and `ub` are lower and upper bounds.

### Scalar Bounds

Use scalar bounds when every input variable has the same range.

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=3, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

print(problem.lb)
print(problem.ub)
```

Example output:

```text
[[0. 0. 0.]]
[[1. 1. 1.]]
```

Here all three variables are in `[0.0, 1.0]`.

### Per-Variable Bounds

Use a list or NumPy array when each input variable has its own range.

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(
    nInput=3,
    nObj=1,
    lb=[0.0, -5.0, 50.0],
    ub=[1.0, 10.0, 100.0],
    objFunc=objFunc,
    xLabels=["width", "slope", "storage"],
)

print(problem.lb)
print(problem.ub)
print(problem.xLabels)
```

Example output:

```text
[[ 0. -5. 50.]]
[[  1.  10. 100.]]
['width', 'slope', 'storage']
```

Read this as:

| Variable | Lower bound | Upper bound |
|---|---:|---:|
| `width` | `0.0` | `1.0` |
| `slope` | `-5.0` | `10.0` |
| `storage` | `50.0` | `100.0` |

## Understand Batched Evaluation

UQPyL often evaluates many candidate points at once. For that reason, functions should usually accept a table-like `X`.

| Shape | Meaning |
|---|---|
| `(n_samples, n_input)` | A batch of input rows. |
| One row of `X` | One candidate input vector. |
| One column of `X` | One input variable. |

For example:

```python
import numpy as np


single_x = np.array([0.2, 0.3])
batch_x = np.array([
    [0.2, 0.3],
    [0.5, 0.1],
    [0.0, 1.0],
])

print(np.atleast_2d(single_x).shape)
print(np.atleast_2d(batch_x).shape)
```

Example output:

```text
(1, 2)
(3, 2)
```

`np.atleast_2d(X)` is useful because it turns both a single row and a batch into the same table shape.

After `X = np.atleast_2d(X)`:

| Expression | Meaning |
|---|---|
| `X[:, 0]` | First variable for all samples. |
| `X[:, 1]` | Second variable for all samples. |
| `X.shape[0]` | Number of samples. |
| `reshape(-1, 1)` | Make a one-column output. |
| `keepdims=True` | Keep a NumPy reduction as a column. |

## Write Objective Functions

### One Objective

For one objective, return shape `(n_samples, 1)`.

```python
import numpy as np


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + X[:, 1] ** 2
    return y.reshape(-1, 1)


print(objFunc([0.2, 0.3]))
print(objFunc([[0.2, 0.3], [0.5, 0.1]]))
```

Example output:

```text
[[0.13]]
[[0.13]
 [0.26]]
```

### Multiple Objectives

For multiple objectives, return one column per objective.

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    f1 = np.sum(X**2, axis=1)
    f2 = np.sum((X - 0.5) ** 2, axis=1)
    return np.vstack([f1, f2]).T


problem = Problem(nInput=2, nObj=2, ub=1.0, lb=0.0, objFunc=objFunc, optType=["min", "min"])

print(problem.evaluate([[0.2, 0.3], [0.5, 0.1]]).objs)
```

Example output:

```text
[[0.13 0.13]
 [0.26 0.16]]
```

The first output row belongs to the first input row. The second output row belongs to the second input row.

## Write Constraints

Use `conFunc` when the problem has constraints. Constraint values are feasible when:

```text
cons <= 0
```

This example means `x1 + x2 <= 1.0`.

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


def conFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] + X[:, 1] - 1.0).reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, nCon=1, ub=1.0, lb=0.0, objFunc=objFunc, conFunc=conFunc)

res = problem.evaluate([[0.2, 0.3], [0.8, 0.4]])
print(res.objs)
print(res.cons)
```

Example output:

```text
[[0.13]
 [0.8 ]]
[[-0.5]
 [ 0.2]]
```

The first row is feasible because `-0.5 <= 0`. The second row violates the constraint because `0.2 > 0`.

For two constraints, return two columns:

```python
def conFunc(X):
    X = np.atleast_2d(X)
    c1 = X[:, 0] + X[:, 1] - 1.0
    c2 = 0.2 - X[:, 0]
    return np.vstack([c1, c2]).T
```

Here `c1 <= 0` means `x1 + x2 <= 1.0`, and `c2 <= 0` means `x1 >= 0.2`.

## Check Return Shapes

These are the shapes UQPyL expects:

| Function | Input shape | Return shape |
|---|---|---|
| `objFunc(X)` with one objective | `(n_samples, n_input)` | `(n_samples, 1)` |
| `objFunc(X)` with two objectives | `(n_samples, n_input)` | `(n_samples, 2)` |
| `conFunc(X)` with one constraint | `(n_samples, n_input)` | `(n_samples, 1)` |
| `simFunc(X)` for simulation models | `(n_samples, n_input)` | `(n_samples, n_time, n_series)` |

Common mistakes:

| Mistake | Why it fails or confuses results | Fix |
|---|---|---|
| Returning `0.13` | This is a scalar, not one row per sample. | Return `[[0.13]]`. |
| Returning shape `(n_samples,)` | This is a flat vector, not a column matrix. | Use `reshape(-1, 1)` or `keepdims=True`. |
| Using `X[0]` as the first variable | `X[0]` is the first row. | Use `X[:, 0]`. |
| Forgetting `np.atleast_2d(X)` | Single inputs and batch inputs may behave differently. | Start with `X = np.atleast_2d(X)`. |

## Use `evaluate()`

`problem.evaluate(X)` returns an `Eval` object.

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


def conFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] + X[:, 1] - 1.0).reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, nCon=1, ub=1.0, lb=0.0, objFunc=objFunc, conFunc=conFunc)
res = problem.evaluate([[0.2, 0.3], [0.8, 0.4]])

print(res.objs)
print(res.cons)
```

| Field | Meaning |
|---|---|
| `objs` | Objective values, or `None` when not requested. |
| `cons` | Constraint values, or `None` when unavailable or not requested. |
| `sim` | Simulation output for `ModelProblem`, otherwise usually `None`. |

Use `target` to request only one output block.

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


def conFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] + X[:, 1] - 1.0).reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, nCon=1, ub=1.0, lb=0.0, objFunc=objFunc, conFunc=conFunc)
obj_res = problem.evaluate([[0.2, 0.3]], target="objs")
con_res = problem.evaluate([[0.2, 0.3]], target="cons")

print(obj_res.objs)
print(obj_res.cons)
print(con_res.objs)
print(con_res.cons)
```

Example output:

```text
[[0.13]]
None
None
[[-0.5]]
```

## When to Use Combined `evaluate`

Use `evaluate` when objectives and constraints share expensive intermediate calculations.

```python
import numpy as np

from UQPyL.problem import Eval, Problem


def evaluate(X):
    X = np.atleast_2d(X)
    total = np.sum(X, axis=1, keepdims=True)
    return Eval(objs=total**2, cons=total - 1.0)


problem = Problem(nInput=2, nObj=1, nCon=1, ub=1.0, lb=0.0, evaluate=evaluate)

res = problem.evaluate([[0.2, 0.3], [0.8, 0.4]])
print(res.objs)
print(res.cons)
```

Example output:

```text
[[0.25]
 [1.44]]
[[-0.5]
 [ 0.2]]
```

`Problem` accepts exactly one callable configuration:

| Configuration | Use when |
|---|---|
| `objFunc` | The problem has objectives only. |
| `objFunc` + `conFunc` | The problem has objectives and constraints. |
| `evaluate` | Objectives and constraints should be computed together. |

Do not combine `evaluate` with `objFunc` or `conFunc`.

## Use Single-Sample Functions

If a batched function feels awkward, write a one-row function and wrap it with `singleFunc`.

```python
import numpy as np

from UQPyL.problem import Problem, singleFunc


@singleFunc
def objFunc(x):
    return float(np.sum(x**2))


problem = Problem(nInput=2, nObj=1, ub=1.0, lb=-1.0, objFunc=objFunc)

print(problem.evaluate([[0.2, 0.3], [0.5, 0.1]]).objs)
```

Example output:

```text
[[0.13]
 [0.26]]
```

For combined objective and constraint evaluation, use `singleEval`.

```python
import numpy as np

from UQPyL.problem import Eval, Problem, singleEval


@singleEval
def evaluate(x):
    return Eval(objs=float(np.sum(x**2)), cons=np.array([x[0] + x[1] - 1.0]))


problem = Problem(nInput=2, nObj=1, nCon=1, ub=1.0, lb=0.0, evaluate=evaluate)

print(problem.evaluate([[0.2, 0.3], [0.8, 0.4]]).objs)
print(problem.evaluate([[0.2, 0.3], [0.8, 0.4]]).cons)
```

## Use Variable Types

By default, all variables are continuous. Use `varType` for integer or discrete variables.

| Value | Type | Meaning |
|---|---|---|
| `0` | Continuous | Any value inside bounds. |
| `1` | Integer | Rounded integer value inside bounds. |
| `2` | Discrete | Mapped to a value from `varSet`. |

Discrete variables require `varSet`.

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X, axis=1, keepdims=True)


problem = Problem(
    nInput=3,
    nObj=1,
    ub=[1.0, 10.0, 1.0],
    lb=[0.0, 0.0, 0.0],
    varType=[0, 1, 2],
    varSet={2: [0.1, 0.5, 0.9]},
    objFunc=objFunc,
)

X = np.array([[0.2, 3.7, 0.8]])
print(problem.apply_var_type(X))
```

Example output:

```text
[[0.2 4.  0.9]]
```

## Use `ModelProblem` for Simulations

Use `ModelProblem` when the main callable is a simulation model. This is common for calibration, time-series models, hydrology models, and other systems where parameters produce simulated outputs.

The simulation function also receives batched input. Its usual return shape is:

```text
(n_samples, n_time, n_series)
```

This example has two parameters, two time steps, and one simulated series.

```python
import numpy as np

from UQPyL.problem import ModelProblem


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, simLabels=["Q"], name="ToyModel")

res = problem.evaluate([[1.0, 2.0], [1.5, 2.5]], target="sim")
print(res.sim.shape)
print(res.sim)
```

Example output:

```text
(2, 2, 1)
[[[1. ]
  [2. ]]

 [[1.5]
  [2.5]]]
```

The three simulation indexes mean:

| Index | Meaning |
|---|---|
| `sim[i, :, :]` | All outputs for sample `i`. |
| `sim[:, t, :]` | All samples at time step `t`. |
| `sim[:, :, j]` | All samples and times for output series `j`. |

### Objective From Simulation Error

If `obs` is provided, an objective can compare simulations with observations through `context`.

```python
import numpy as np

from UQPyL.problem import ModelProblem


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


problem = ModelProblem(nInput=2, nObj=1, ub=3.0, lb=0.0, simFunc=simFunc, objFunc=objFunc, obs=obs, simLabels=["Q"])

res = problem.evaluate([[1.0, 2.2], [0.0, 0.0]])
print(res.objs)
```

Example output:

```text
[[0.02]
 [2.5 ]]
```

### ModelProblem Checklist

| Check | Expected |
|---|---|
| `obs` | 2D array, usually `(n_time, n_series)`. |
| `mask` | Same shape as `obs`; `True` means ignored. |
| `simFunc(X).shape[0]` | Same as `np.atleast_2d(X).shape[0]`. |
| `simFunc(X).shape[1:]` | Should match `obs.shape` when observations are used. |
| `objFunc(X, context)` | Returns `(n_samples, n_obj)`. |

## Core Objects

| Concept | Role |
|---|---|
| `Space` | Defines input dimension, bounds, labels, and variable types. |
| `Problem` | Defines objectives and optional constraints for static problems. |
| `ModelProblem` | Defines simulation models with observations, masks, and simulation context. |
| `Eval` | Standard return object from `evaluate()`. |
| `singleFunc` | Adapts a single-sample objective function to batched input. |
| `singleEval` | Adapts a single-sample evaluation function to batched input. |

## Common Mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Returning objective values with shape `(n_samples,)`. | Some downstream methods expect a 2D objective table and may fail or misread the output. | Return `(n_samples, n_obj)`, for example `y.reshape(-1, 1)` or `keepdims=True`. |
| Writing `objFunc` for one row but passing batched `X`. | The function works for one sample and breaks when DOE, optimization, or analysis sends many rows. | Use `np.atleast_2d(X)` and compute one output per row, or wrap a one-row function with `@singleFunc`. |
| Using dictionary-style result access such as `res["objs"]`. | `Problem.evaluate()` returns an `Eval` object, not a dictionary. | Use `res.objs`, `res.cons`, and `res.sims`. |
| Giving scalar `lb` and `ub` when variables need different ranges. | All variables receive the same bounds. | Use vectors such as `lb=[0.0, -5.0]` and `ub=[1.0, 10.0]`. |
| Defining constraints with the wrong sign. | Feasible and infeasible points are reversed. | Use `cons <= 0` for feasible points. |
| Setting `nCon=1` but returning no constraint values. | Constraint-aware methods cannot read feasibility. | Provide `conFunc` or a combined `evaluate()` that returns `Eval(cons=...)`. |
| Returning simulation output as `(n_time, n_series)` for a batched `ModelProblem`. | Calibration expects one simulation per input row. | Return `(n_samples, n_time, n_series)` from `simFunc(X)`. |
| Applying `mask=True` to values you want to use. | Masked values are ignored in calibration metrics. | Use `True` only for missing or ignored observations. |
| Mixing up `optType` direction. | Optimization and inference orient the objective incorrectly. | Use `"min"` for loss/error and `"max"` for benefit/score. |

## Benchmark Problems

UQPyL includes benchmark problems under `UQPyL.problem`.

| Type | Examples |
|---|---|
| Single-objective | `Sphere`, `Ackley`, `Rosenbrock`, `Rastrigin`, `Griewank`, `Trid` |
| Constrained single-objective | `RosenbrockWithCon` |
| Multi-objective | `ZDT1`, `ZDT2`, `ZDT3`, `ZDT4`, `ZDT6`, `DTLZ1`-`DTLZ7` |

```python
from UQPyL.problem import Sphere, ZDT1


sphere = Sphere(nInput=10)
zdt1 = ZDT1(nInput=5)

print(sphere.evaluate([[0.0] * 10]).objs)
print(zdt1.evaluate([[0.5] * 5]).objs)
```

## Next Steps

| Goal | Read |
|---|---|
| Look up constructor parameters | [Problem API](api/problem.md) |
| Generate samples from a problem | [Design of Experiment](doe.md) |
| Run complete examples | [Examples](examples.md) |
