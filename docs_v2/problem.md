# Problem

The `problem` module is the modeling core of UQPyL. Functional modules such as `doe`, `analysis`, `optimization`, `inference`, `calibration`, and `surrogate` all rely on a `Problem`-like object to describe the input space and evaluation rule.

## Core Concepts

| Concept | Role |
|---|---|
| `Space` | Defines input dimension, bounds, labels, and variable types. |
| `Problem` | Defines objectives and optional constraints for static problems. |
| `ModelProblem` | Defines simulation models with observations, masks, and simulation context. |
| `Eval` | Standard return object from `evaluate()`. |
| `singleFunc` | Adapts a single-sample objective function to batched input. |
| `singleEval` | Adapts a single-sample evaluation function to batched input. |

## Input Space

A `Problem` needs an input space. The common form is to provide `nInput`, `ub`, and `lb` directly.

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(
    nInput=2,
    nObj=1,
    ub=[1.0, 2.0],
    lb=[0.0, -1.0],
    objFunc=objFunc,
    xLabels=["x1", "x2"],
    objLabels=["f"],
    name="Quadratic",
)
```

Bounds can be scalar values or arrays/lists with one value per input variable.

| Field | Meaning |
|---|---|
| `nInput` | Number of input variables. |
| `ub` | Upper bounds. |
| `lb` | Lower bounds. |
| `xLabels` | Optional input variable labels. |

## Variable Types

By default, all variables are continuous. Use `varType` to declare mixed spaces.

| Value | Type |
|---|---|
| `0` | Continuous |
| `1` | Integer |
| `2` | Discrete |

Discrete variables require `varSet`.

```python
problem = Problem(
    nInput=3,
    nObj=1,
    ub=[1.0, 10.0, 1.0],
    lb=[0.0, 0.0, 0.0],
    varType=[0, 1, 2],
    varSet={2: [0.1, 0.5, 0.9]},
    objFunc=objFunc,
)
```

`Space` provides transformation helpers:

| Method | Purpose |
|---|---|
| `validate(X)` | Convert input to 2D and check column count. |
| `unit_to_space(X)` | Map unit-space samples to problem bounds. |
| `cast_int_vars(X)` | Round integer variables. |
| `map_discrete_vars(X)` | Map discrete variables to values from `varSet`. |
| `apply_var_type(X)` | Apply integer and discrete transformations. |

## Defining a Problem

`Problem` accepts exactly one of these callable configurations:

| Configuration | Use when |
|---|---|
| `objFunc` | The problem has objectives only. |
| `objFunc` + `conFunc` | The problem has objectives and constraints. |
| `evaluate` | Objectives and constraints should be computed together. |

Do not combine `evaluate` with `objFunc` or `conFunc`.

### Objective Function

Objective functions receive a 2D NumPy array with shape `(n_samples, n_input)` and return a 2D array with shape `(n_samples, n_obj)`.

```python
def objFunc(X):
    X = np.atleast_2d(X)
    f1 = np.sum(X**2, axis=1)
    f2 = np.sum((X - 0.5) ** 2, axis=1)
    return np.vstack([f1, f2]).T


problem = Problem(
    nInput=2,
    nObj=2,
    ub=1.0,
    lb=0.0,
    objFunc=objFunc,
    optType=["min", "min"],
)
```

### Constraints

Constraint functions return a 2D array with shape `(n_samples, n_con)`.

In UQPyL, constraints are feasible when:

```text
cons <= 0
```

Example:

```python
def conFunc(X):
    X = np.atleast_2d(X)
    return X[:, [0]] + X[:, [1]] - 1.0


problem = Problem(
    nInput=2,
    nObj=1,
    nCon=1,
    ub=1.0,
    lb=0.0,
    objFunc=objFunc,
    conFunc=conFunc,
    conLabels=["x1_plus_x2_minus_1"],
)
```

## Unified Evaluation

`problem.evaluate(X)` always returns an `Eval` object.

```python
res = problem.evaluate([[0.2, 0.3]])

print(res.objs)
print(res.cons)
```

| Field | Meaning |
|---|---|
| `objs` | Objective values, or `None` when not requested. |
| `cons` | Constraint values, or `None` when unavailable or not requested. |
| `sim` | Simulation output for `ModelProblem`, otherwise usually `None`. |

Use `target` to request only part of the evaluation:

```python
obj_res = problem.evaluate(X, target="objs")
con_res = problem.evaluate(X, target="cons")
```

## Combined `evaluate`

Use `evaluate` when objective and constraint values share expensive intermediate computation.

```python
from UQPyL.problem import Eval, Problem


def evaluate(X):
    X = np.atleast_2d(X)
    total = np.sum(X, axis=1, keepdims=True)
    return Eval(
        objs=total**2,
        cons=total - 1.0,
    )


problem = Problem(
    nInput=2,
    nObj=1,
    nCon=1,
    ub=1.0,
    lb=0.0,
    evaluate=evaluate,
)
```

## Single-Sample Functions

Some models are easier to write for one input vector at a time. Use `singleFunc` or `singleEval` to adapt them to UQPyL's batched protocol.

```python
from UQPyL.problem import Problem, singleFunc


@singleFunc
def objFunc(x):
    return float(np.sum(x**2))


problem = Problem(
    nInput=2,
    nObj=1,
    ub=1.0,
    lb=-1.0,
    objFunc=objFunc,
)
```

For combined objective and constraint evaluation:

```python
from UQPyL.problem import Eval, Problem, singleEval


@singleEval
def evaluate(x):
    return Eval(
        objs=float(np.sum(x**2)),
        cons=np.array([x[0] - 0.5]),
    )
```

## ModelProblem

`ModelProblem` is used when the primary computation is a simulation model. It extends `Problem` with:

| Field | Meaning |
|---|---|
| `simFunc` | Batched simulation function. |
| `obs` | Optional observed data. |
| `mask` | Optional boolean mask for ignored observations. |
| `simLabels` | Optional labels for simulation series. |
| `ModelEvalContext` | Object passed to `objFunc`, `conFunc`, or `evaluate`. |

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
    return np.mean(err**2, axis=(1, 2), keepdims=False).reshape(-1, 1)


problem = ModelProblem(
    nInput=2,
    nObj=1,
    ub=3.0,
    lb=0.0,
    simFunc=simFunc,
    objFunc=objFunc,
    obs=obs,
    simLabels=["Q"],
    name="ToyModel",
)
```

Evaluate simulation output only:

```python
res = problem.evaluate([[1.0, 2.0]], target="sim")
print(res.sim)
```

`ModelProblem` also provides:

| Method | Purpose |
|---|---|
| `buildContext(X)` | Run simulation and build `ModelEvalContext`. |
| `flattenSim(sim)` | Flatten simulation output to `(n_samples, n_obs)`. |
| `flattenObs()` | Flatten observations to `(n_obs,)`. |
| `flattenMask()` | Flatten mask to `(n_obs,)`. |

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
```
