# Problem API

## `UQPyL.problem`

The `problem` module defines the shared modeling protocol used by sampling, analysis, optimization, inference, calibration, and surrogate modeling.

### Import

```python
from UQPyL.problem import Problem, ModelProblem, Eval
```

### Public Objects

| Object | Role |
|---|---|
| `Problem` | Define a static objective and optional constraint problem. |
| `ModelProblem` | Define a simulation problem with optional observations and masks. |
| `ModelEvalContext` | Runtime context passed to `ModelProblem` objective, constraint, or evaluation callables. |
| `Eval` | Standard return object from `evaluate()`. |
| `Space` | Define a bounded input space with continuous, integer, or discrete variables. |
| `SpaceBase` | Base input-space interface. |
| `ProblemBase` | Base problem interface used by built-in benchmark problems. |
| `ProblemABC` | Alias of `ProblemBase`. |
| `singleFunc` | Decorator for adapting one-sample objective functions to batched input. |
| `singleEval` | Decorator for adapting one-sample `Eval` functions to batched input. |

## `Problem`

Use `Problem` when objectives and constraints are computed directly from input variables.

```python
Problem(
    nInput=None,
    nObj=None,
    ub=None,
    lb=None,
    objFunc=None,
    conFunc=None,
    evaluate=None,
    conWgt=None,
    nCon=0,
    varType=None,
    varSet=None,
    optType="min",
    xLabels=None,
    name=None,
    space=None,
    objLabels=None,
    conLabels=None,
)
```

| Parameter | Meaning |
|---|---|
| `nInput` | Number of input variables. Required unless `space` is provided. |
| `nObj` | Number of objectives. Required. |
| `ub` | Upper bounds. Scalar, list, or NumPy array. |
| `lb` | Lower bounds. Scalar, list, or NumPy array. |
| `objFunc` | Objective callable. Receives batched `X`, returns shape `(n_samples, n_obj)`. |
| `conFunc` | Constraint callable. Receives batched `X`, returns shape `(n_samples, n_con)`. |
| `evaluate` | Combined evaluation callable. Must return `Eval`. |
| `conWgt` | Optional constraint weights. |
| `nCon` | Number of constraints. Defaults to `0`. |
| `varType` | Variable type list. `0` continuous, `1` integer, `2` discrete. |
| `varSet` | Discrete value mapping for variables with `varType=2`. |
| `optType` | Optimization direction. `"min"`, `"max"`, or one value per objective. |
| `xLabels` | Optional input labels. |
| `name` | Optional problem name. |
| `space` | Optional `SpaceBase` object. When provided, it defines the input space. |
| `objLabels` | Optional objective labels. |
| `conLabels` | Optional constraint labels. |

`Problem` accepts exactly one callable configuration:

| Configuration | Use when |
|---|---|
| `objFunc` | The problem has objectives only. |
| `objFunc` + `conFunc` | The problem has objectives and constraints. |
| `evaluate` | Objectives and constraints should be computed together. |

Do not combine `evaluate` with `objFunc` or `conFunc`.

### Methods

| Method | Returns | Meaning |
|---|---|---|
| `evaluate(X, target=None)` | `Eval` | Evaluate objectives and constraints. |
| `objFunc(X)` | `np.ndarray` | Evaluate objectives only. |
| `conFunc(X)` | `np.ndarray` or `None` | Evaluate constraints only. |
| `validate(X)` | `np.ndarray` | Convert input to 2D and check dimension. |
| `unit_to_space(X, IFlag=True, DFlag=True)` | `np.ndarray` | Map unit-space values to problem bounds. |
| `apply_var_type(X, IFlag=True, DFlag=True)` | `np.ndarray` | Apply integer and discrete variable transforms. |
| `cast_int_vars(X)` | `np.ndarray` | Round integer variables. |
| `map_discrete_vars(X)` | `np.ndarray` | Map discrete variables to values in `varSet`. |
| `getOptimum()` | implementation-specific | Return a known optimum when implemented. |

`target` can be:

| Value | Meaning |
|---|---|
| `None` | Return all available outputs. |
| `"objs"` | Return objectives only. |
| `"cons"` | Return constraints only. |

Example:

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(
    nInput=2,
    nObj=1,
    ub=1.0,
    lb=-1.0,
    objFunc=objFunc,
)

res = problem.evaluate([[0.2, 0.3]])
print(res.objs)
```

## `ModelProblem`

Use `ModelProblem` when the primary callable is a simulation model.

```python
ModelProblem(
    nInput=None,
    nObj=1,
    ub=None,
    lb=None,
    simFunc=None,
    objFunc=None,
    conFunc=None,
    evaluate=None,
    obs=None,
    mask=None,
    conWgt=None,
    nCon=0,
    varType=None,
    varSet=None,
    optType="min",
    xLabels=None,
    name=None,
    space=None,
    objLabels=None,
    conLabels=None,
    simLabels=None,
)
```

| Parameter | Meaning |
|---|---|
| `simFunc` | Required simulation callable. Receives batched `X`; returns numeric NumPy array whose first dimension is `n_samples`. |
| `objFunc` | Optional objective callable. Receives `(X, context)`. |
| `conFunc` | Optional constraint callable. Receives `(X, context)`. |
| `evaluate` | Optional combined callable. Receives `(X, context)` and returns `Eval`. |
| `obs` | Optional 2D observation array with shape `(n_time, n_series)`. |
| `mask` | Optional boolean mask with the same shape as `obs`. |
| `simLabels` | Optional labels for simulation series. |

Other parameters are the same as `Problem`.

### Methods

| Method | Returns | Meaning |
|---|---|---|
| `evaluate(X, target=None)` | `Eval` | Evaluate simulation, objectives, and constraints. |
| `simFunc(X)` | `np.ndarray` | Run the simulation callable and validate output. |
| `buildContext(X)` | `ModelEvalContext` | Run simulation and build the context object. |
| `objFunc(X, context=None)` | `np.ndarray` | Evaluate objectives. Builds context when omitted. |
| `conFunc(X, context=None)` | `np.ndarray` or `None` | Evaluate constraints. Builds context when omitted. |
| `flattenSim(sim)` | `np.ndarray` | Flatten simulation output to `(n_samples, n_obs)`. |
| `flattenObs()` | `np.ndarray` | Flatten observations to `(n_obs,)`. |
| `flattenMask()` | `np.ndarray` | Flatten mask to `(n_obs,)`. |

`ModelProblem.evaluate()` supports one extra target:

| Value | Meaning |
|---|---|
| `"sim"` | Return simulation output only in `Eval.sim`. |

Example:

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


problem = ModelProblem(
    nInput=2,
    nObj=1,
    ub=3.0,
    lb=0.0,
    simFunc=simFunc,
    objFunc=objFunc,
    obs=obs,
)

res = problem.evaluate([[1.0, 2.2]])
print(res.objs)
print(res.sim)
```

## `Eval`

`Eval` is the standard return object from `evaluate()`.

```python
Eval(objs=None, cons=None, sim=None)
```

| Field | Type | Meaning |
|---|---|---|
| `objs` | `np.ndarray` or `None` | Objective values. |
| `cons` | `np.ndarray` or `None` | Constraint values. Feasible constraints satisfy `cons <= 0`. |
| `sim` | `np.ndarray` or `None` | Simulation output for `ModelProblem`. |

| Property | Meaning |
|---|---|
| `hasObjs` | `True` when `objs` is not `None`. |
| `hasCons` | `True` when `cons` is not `None`. |
| `hasSim` | `True` when `sim` is not `None`. |

## `Space`

`Space` defines input dimension, bounds, labels, and variable type transformations.

```python
Space(
    nInput,
    ub,
    lb,
    varType=None,
    varSet=None,
    xLabels=None,
)
```

| Parameter | Meaning |
|---|---|
| `nInput` | Number of input variables. |
| `ub` | Upper bounds. Scalar, list, or NumPy array. |
| `lb` | Lower bounds. Scalar, list, or NumPy array. |
| `varType` | Optional variable type list. |
| `varSet` | Required mapping for discrete variables. |
| `xLabels` | Optional input labels. |

Variable types:

| Value | Meaning |
|---|---|
| `0` | Continuous |
| `1` | Integer |
| `2` | Discrete |

Methods:

| Method | Returns | Meaning |
|---|---|---|
| `validate(X)` | `np.ndarray` | Convert input to 2D and check dimension. |
| `transform(X)` | `np.ndarray` | Validate input and apply variable type transforms. |
| `unit_to_space(X, IFlag=True, DFlag=True)` | `np.ndarray` | Map unit-space values to bounds. |
| `apply_var_type(X, IFlag=True, DFlag=True)` | `np.ndarray` | Apply integer and discrete transforms. |
| `cast_int_vars(X)` | `np.ndarray` | Round integer variables. |
| `map_discrete_vars(X)` | `np.ndarray` | Map discrete variables to `varSet` values. |

Attributes:

| Attribute | Meaning |
|---|---|
| `nInput` | Number of input variables. |
| `ub` | Upper bounds as shape `(1, n_input)`. |
| `lb` | Lower bounds as shape `(1, n_input)`. |
| `varType` | Variable type array. |
| `idxF` | Continuous variable indices. |
| `idxI` | Integer variable indices. |
| `idxD` | Discrete variable indices. |
| `varSet` | Discrete variable value mapping. |
| `xLabels` | Input labels. |
| `encoding` | `"real"` for continuous spaces, `"mix"` for mixed spaces. |

## Decorators

### `singleFunc`

Use `singleFunc` when an objective function is easier to write for one sample at a time.

```python
from UQPyL.problem import Problem, singleFunc


@singleFunc
def objFunc(x):
    return x[0] ** 2 + x[1] ** 2


problem = Problem(nInput=2, nObj=1, ub=1.0, lb=-1.0, objFunc=objFunc)
```

The wrapped function receives one 1D sample and the wrapper returns a 2D objective array.

### `singleEval`

Use `singleEval` when a one-sample function returns `Eval`.

```python
import numpy as np

from UQPyL.problem import Eval, Problem, singleEval


@singleEval
def evaluate(x):
    return Eval(
        objs=np.array([x[0] ** 2 + x[1] ** 2]),
        cons=np.array([x[0] + x[1] - 1.0]),
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

## Benchmark Problems

Built-in benchmark problems can be imported from `UQPyL.problem`.

### Single-Objective Problems

| Class |
|---|
| `Sphere` |
| `Schwefel_2_22` |
| `Schwefel_1_22` |
| `Schwefel_2_21` |
| `Rosenbrock` |
| `Step` |
| `Quartic` |
| `Schwefel_2_26` |
| `Rastrigin` |
| `Ackley` |
| `Griewank` |
| `Trid` |
| `Bent_Cigar` |
| `Discus` |
| `Weierstrass` |
| `RosenbrockWithCon` |

### Multi-Objective Problems

| Class |
|---|
| `ZDT1` |
| `ZDT2` |
| `ZDT3` |
| `ZDT4` |
| `ZDT6` |
| `DTLZ1` |
| `DTLZ2` |
| `DTLZ3` |
| `DTLZ4` |
| `DTLZ5` |
| `DTLZ6` |
| `DTLZ7` |

Example:

```python
from UQPyL.problem import Sphere, ZDT1


sphere = Sphere(nInput=10)
zdt1 = ZDT1(nInput=30)

print(sphere.evaluate([[0.0] * 10]).objs)
print(zdt1.evaluate([[0.5] * 30]).objs)
```

