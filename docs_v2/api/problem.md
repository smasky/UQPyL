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
| `ProblemBase` | Shared protocol base for all problem objects. |
| `Problem` | Define a direct problem that follows `X -> objs/cons`. |
| `ModelProblem` | Define a simulation-backed problem that follows `X -> sim -> objs/cons`. |
| `SimContext` | Runtime context passed to `ModelProblem` objective and constraint callables. |
| `Eval` | Standard return object from `evaluate()`. |
| `Space` | Define a bounded input space with continuous, integer, or discrete variables. |
| `SpaceBase` | Base input-space interface. |
| `ProblemABC` | Alias of `ProblemBase`. |
| `singleFunc` | Decorator for adapting one-sample objective functions to batched input. |
| `EvaluatorBase` | Shared base interface for direct evaluators. |
| `Evaluator` | Direct evaluator type used by `Problem`. |
| `ModelEvaluatorBase` | Shared base interface for model evaluators. |
| `ModelEvaluator` | Simulation-postprocessing evaluator type used by `ModelProblem`. |
| `SimulatorBase` | Base simulator interface for advanced customization. |

Notes:

- `Evaluator.evaluate()` uses `evaluate(X, target=None)`
- `ModelEvaluator.evaluate()` uses `evaluate(X, simContext, target=None)`

## Architecture

The current `problem` module is organized in three layers:

| Layer | Role |
|---|---|
| `ProblemBase` | Shared protocol and shared structure such as `Space`, dimensions, labels, and common validation. |
| `Problem` | Direct mode. Maps `X` directly to objectives and constraints. |
| `ModelProblem` | Simulation mode. Maps `X` to `sim` first, then to objectives and constraints. |

`Problem` and `ModelProblem` are parallel problem modes. They are not parent-child variants of each other.

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
    evaluator=None,
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
| `evaluator` | Optional advanced evaluator. Prefer `objFunc` / `conFunc` unless you need advanced customization. The recommended route is to subclass `Evaluator` and override `evaluate(X, target=None)`, then pass the instance here. |

`Problem` accepts the following callable configurations:

| Configuration | Use when |
|---|---|
| `objFunc` | The problem has objectives only. |
| `objFunc` + `conFunc` | The problem has objectives and constraints. |

Rule:

- `evaluator` is an advanced route and cannot be used together with `objFunc` or `conFunc`

### Recommended Extension Route

Use this priority order:

1. `objFunc`
2. `objFunc + conFunc`
3. subclass `Evaluator` and override `evaluate(X, target=None)`

### Object `evaluator` Example

If `objFunc` / `conFunc` is not enough, subclass `Evaluator`, override `evaluate(X, target=None)`, and pass the instance into `Problem`:

```python
import numpy as np

from UQPyL.problem import Eval, Evaluator, Problem


class QuadraticEvaluator(Evaluator):
    def evaluate(self, X, target=None):
        X = np.atleast_2d(X)
        objs = np.sum(X**2, axis=1, keepdims=True)
        cons = (X[:, 0] + X[:, 1] - 1.0).reshape(-1, 1)
        return Eval(objs=objs, cons=cons, target=target)


problem = Problem(
    nInput=2,
    nObj=1,
    nCon=1,
    lb=0.0,
    ub=1.0,
    evaluator=QuadraticEvaluator(),
)
```

This is the recommended direct-evaluator route.

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

`evaluate` checks the target, converts a one-dimensional sample to a two-dimensional batch, executes the evaluation, and validates the returned `Eval`. Overridden problem methods receive the same normalized input; arrays with three or more dimensions are rejected. `Problem` requires `objs` for `target=None` or `"objs"`. Both problem types require `cons` when `nCon > 0` and the target is `None` or `"cons"`; an unconstrained problem may return an empty constraint result. A simulation-only `ModelProblem` may return only `sims` for `target=None`; an explicit `"objs"` request requires objectives.

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

The recommended `ModelProblem` flow is:

```text
X -> simFunc(X) -> context.sim -> objFunc/conFunc -> Eval
```

```python
ModelProblem(
    nInput=None,
    nObj=1,
    ub=None,
    lb=None,
    simFunc=None,
    objFunc=None,
    conFunc=None,
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
    seriesLabels=None,
)
```

| Parameter | Meaning |
|---|---|
| `simFunc` | Required simulation callable. Receives batched `X`; returns numeric NumPy array whose first dimension is `n_samples`. |
| `objFunc` | Optional objective callable. Receives `(X, context)`. Prefer deriving objectives from `context.sim`. |
| `conFunc` | Optional constraint callable. Receives `(X, context)`. Prefer deriving constraints from `context.sim`. |
| `obs` | Optional 2D observation array with shape `(n_time, n_series)`. |
| `mask` | Optional boolean mask with the same shape as `obs`. |
| `seriesLabels` | Optional labels for simulation series. |

Usage note:

- A `ModelProblem` with only `simFunc` is valid. This is the simulation-only route and is mainly intended for calibration or simulation-centric workflows.
- If you want a `ModelProblem` to be consumed by general objective-driven modules such as optimization, inference, or internal evaluation-driven analysis, define at least `objFunc`.
- Add `conFunc` when the simulation-backed problem also needs constraints.

Other parameters are the same as `Problem`.

Extension layers for `ModelProblem` are:

- Simple route: `simFunc`, `objFunc`, `conFunc`
- Advanced route: `evaluator`, and `simulator` when needed

### Methods

| Method | Returns | Meaning |
|---|---|---|
| `evaluate(X, target=None)` | `Eval` | Evaluate simulation, objectives, and constraints. |
| `simFunc(X)` | `np.ndarray` | Run the simulation callable and validate output. |
| `simulate(X)` | `SimContext` | Run the configured simulator and return the simulation context. |
| `objFunc(X, context)` | `np.ndarray` | Evaluate objectives with an explicit simulation context. |
| `conFunc(X, context)` | `np.ndarray` or `None` | Evaluate constraints with an explicit simulation context. |
| `flattenSim(sim)` | `np.ndarray` | Flatten simulation output to `(n_samples, n_obs)`. |
| `flattenObs()` | `np.ndarray` | Flatten observations to `(n_obs,)`. |
| `flattenMask()` | `np.ndarray` | Flatten mask to `(n_obs,)`. |

`ModelProblem.evaluate()` supports one extra target:

| Value | Meaning |
|---|---|
| `"sim"` | Return simulation output only in `Eval.sim`. |

### Recommended Extension Route

Use this priority order:

1. `simFunc + objFunc`
2. `simFunc + objFunc + conFunc`
3. subclass `ModelEvaluator` and override `evaluate(X, simContext, target=None)`
4. when needed, also pass a custom `simulator`

### Object `evaluator` Example

If `simFunc + objFunc (+ conFunc)` is not enough, subclass `ModelEvaluator`, override `evaluate(X, simContext, target=None)`, and pass the instance into `ModelProblem`:

```python
import numpy as np

from UQPyL.problem import Eval, ModelEvaluator, ModelProblem


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


class MSEEvaluator(ModelEvaluator):
    def evaluate(self, X, simContext, target=None):
        sim = simContext.sim
        err = sim - simContext.obs
        objs = np.mean(err**2, axis=(1, 2)).reshape(-1, 1)
        return Eval(objs=objs, sim=sim, target=target)


problem = ModelProblem(
    nInput=2,
    nObj=1,
    lb=0.0,
    ub=3.0,
    simFunc=simFunc,
    obs=obs,
    evaluator=MSEEvaluator(),
)
```

This is the recommended model-evaluator route.

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


## Unit and real coordinate conversion

`problem.unit_to_space(U)` decodes unit coordinates, `problem.space_to_unit(X)` encodes real values, and `problem.canonicalize_unit(U)` assigns equivalent integer/discrete encodings a unique representative. Each returns a copy; `Space` implements the rules behind the Problem interface.

Continuous variables use finite-bound linear scaling; fixed continuous variables encode as `0.5`. Integers from `ceil(lb)` through `floor(ub)` and discrete choices in `varSet` use equal-width bins with midpoint representatives; `U=1` selects the last value. Integer sampling therefore uses equal bins rather than linear scaling followed by rounding.

Discrete choices must currently be distinct finite numeric values. Their real values come from `varSet`, independently of the legacy encoding bounds in that column. Encoding validates real bounds, legal integers and choice membership. Decoding an encoded real value reproduces that value up to floating-point rounding; encoding a decoded unit point returns its canonical representative.

`evaluate(X)` accepts real values without guessing the coordinate system. The legacy `apply_var_type()` is no longer used by the optimization evaluation boundary.

### Constraint weights

`Problem(conWgt=[10, 1], nCon=2, ...)` assigns one finite nonnegative weight per constraint. The length must match `nCon`. `None` leaves violations unweighted; a zero weight ignores that constraint, including in feasibility checks.

```python
CV = np.sum(np.maximum(0, cons * conWgt), axis=1)
```


`varType` must be a one-dimensional vector of length `nInput` containing only integer-valued 0, 1, or 2 entries. Invalid values are rejected before conversion; fractional values are never truncated. The old `_transform_*` wrappers have been removed; use `unit_to_space`, `apply_var_type`, `cast_int_vars`, and `map_discrete_vars`. Independent objective/constraint callbacks run only for the requested `target`; a full evaluation still runs both.
