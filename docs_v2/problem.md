# Problem Module

The `problem` module abstracts real-world tasks into unified problem objects that UQPyL can consume. Each problem is described through three standardized components: **Space** (input space), **Evaluation** (evaluation process), and **Eval** (result container). Once defined, a problem object can be used consistently across sampling, optimization, calibration, and analysis workflows.

## Unit and real coordinate conversion

`problem.unit_to_space(U)` decodes unit coordinates, `problem.space_to_unit(X)` encodes real values, and `problem.canonicalize_unit(U)` assigns equivalent integer/discrete encodings a unique representative. Each returns a copy; `Space` implements the rules behind the Problem interface.

Continuous variables use finite-bound linear scaling; fixed continuous variables encode as `0.5`. Integers from `ceil(lb)` through `floor(ub)` and discrete choices in `varSet` use equal-width bins with midpoint representatives; `U=1` selects the last value. Integer sampling therefore uses equal bins rather than linear scaling followed by rounding.

Discrete choices must currently be distinct finite numeric values. Their real values come from `varSet`, independently of the legacy encoding bounds in that column. Encoding validates real bounds, legal integers and choice membership. Decoding an encoded real value reproduces that value up to floating-point rounding; encoding a decoded unit point returns its canonical representative.

`evaluate(X)` accepts real values without guessing the coordinate system. The legacy `apply_var_type()` is no longer used by the optimization evaluation boundary.


`Problem = Space + Evaluation + Eval`

## Space: Input Space

`Space` describes the dimensionality, ranges, and types of the input variables. If no explicit `space=` is provided, `Problem` and `ModelProblem` construct one automatically from their constructor arguments.

### Basic Definition

Three parameters define the basic shape of the input space:

- `nInput`: number of input variables.
- `lb` / `ub`: lower and upper bounds per dimension. A scalar is broadcast to all dimensions; a list or array specifies per-dimension bounds.
- `xLabels`: labels for input variables. Defaults to `['x_1', 'x_2', ...]`.

### Variable Types

`varType` is a list of length `nInput` declaring the type of each variable:

| Code | Type | Description |
|:----:|------|------|
| `0` | Continuous | Default. Varies continuously within `[lb, ub]`. |
| `1` | Integer | Automatically rounded (`np.round`). |
| `2` | Discrete | Mapped from evenly divided intervals of `[lb, ub]` to values in `varSet`. |

`varSet` is a `dict` mapping dimension indices to lists of allowed discrete values. The interval `[lb_i, ub_i]` is evenly partitioned and mapped to the corresponding `varSet[i]` values.

```python
problem = Problem(
    nInput=3, nObj=1,
    lb=[0.0, 0.0, 0.0],
    ub=[1.0, 5.0, 1.0],
    varType=[0, 1, 2],                              # continuous / integer / discrete
    varSet={2: [0.1, 0.3, 0.5, 0.7, 0.9]},          # allowed values for dimension 3
    objFunc=objFunc,
)
```

### Transformation Methods

`Space` provides methods to transform sample points into the valid space. These can be called directly on `Problem` / `ModelProblem` instances (delegating to `self.space`):

| Method | Purpose |
|------|------|
| `validate(X)` | Checks dimensions, ensures 2D `(nSamples, nInput)` array |
| `cast_int_vars(X)` | Rounds integer-typed variables |
| `map_discrete_vars(X)` | Maps discrete-typed variables to `varSet` values |
| `apply_var_type(X)` | Applies both rounding and discrete mapping |
| `unit_to_space(X)` | Scales from the `[0, 1]` unit hypercube to `[lb, ub]`, then applies type transforms |

A common pattern is for a sampling algorithm to generate points in the unit hypercube and then map them into the actual space with `unit_to_space`:

```python
X_unit = np.random.rand(100, problem.nInput)   # (100, nInput), range [0, 1]
X_real = problem.unit_to_space(X_unit)          # scale + round integers + map discrete
```

---

## Evaluation: Evaluation Process

`Evaluation` answers a simple question: given a batch of inputs, how are the outputs produced? Users provide the evaluation logic through one or more callables, while the framework coordinates them internally through the single `evaluate()` entry point.

From the standpoint of evaluation, real-world problems usually fall into two categories. In the first, objectives and constraints are computed directly from inputs, with no intermediate simulation stage. In the second, a simulation model runs first, and objectives, constraints, or error metrics are derived from the simulation output. UQPyL represents these two modes with `Problem` and `ModelProblem`, respectively.

### Direct Evaluation: `Problem`

Use `Problem` when objectives and constraints can be computed directly from inputs, such as in mathematical test functions or black-box scoring problems.

#### objFunc

`objFunc` is the required callable that maps inputs to objective values:

| Item | Contract |
|------|------|
| Input | `X`, an `np.ndarray` of shape `(nSamples, nInput)` |
| Output | `objs`, an `np.ndarray` of shape `(nSamples, nObj)` |
| Batching | Always handle 2D input. Add `X = np.atleast_2d(X)` as the first line. |

The framework validates output shape against `nObj` inside `evaluate()`.

```python
def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)   # sum across variables; keepdims preserves the column shape
```

#### conFunc

`conFunc` is an optional callable describing constraint functions. If provided, `nCon` must also be declared:

| Item | Contract |
|------|------|
| Input | `X`, shape `(nSamples, nInput)` |
| Output | `cons`, shape `(nSamples, nCon)` |
| Feasibility | `cons <= 0` is considered feasible. If the actual constraint is `g(x) >= 0`, negate it inside the function. |

```python
def conFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] + X[:, 1] - 1.0).reshape(-1, 1)   # reshape to column vector (nSamples, nCon)
```

#### Example

```python
import numpy as np
from UQPyL.problem import Problem

def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)

def conFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] + X[:, 1] - 1.0).reshape(-1, 1)

problem = Problem(
    nInput=2,              # input dimensionality
    nObj=1,                # objective dimensionality
    nCon=1,                # constraint dimensionality
    lb=0.0, ub=1.0,        # input bounds
    objFunc=objFunc,       # objective function (required)
    conFunc=conFunc,       # constraint function (optional; nCon required if provided)
    optType='min',         # optimization direction: 'min' / 'max' / ['min', 'max', ...]
    varType=[0, 0],        # variable types; defaults to all continuous
    xLabels=['x1', 'x2'],  # input labels
    name='MyProblem',
)

res = problem.evaluate([[0.2, 0.3], [0.5, 0.6]])
```

#### optType: Optimization Direction

`optType` declares the optimization direction for each objective. The framework internally converts everything to minimization. Two forms are supported:

- String: `'min'` or `'max'` — applies to all objectives when they share the same direction.
- List: `['min', 'max', 'min']` — per-objective specification; length must equal `nObj`.

After instantiation, `problem.opt` exposes the numeric form (`1` for minimization, `-1` for maximization).

#### Custom Evaluator

When `objFunc` and `conFunc` are not sufficient, for example because preprocessing, caching, or external process calls are involved, subclass `Evaluator` and implement `evaluate(self, X, target)` to encapsulate the full evaluation logic in one place.

`Evaluator.evaluate()` contract:

| Item | Contract |
|------|------|
| Input | `X` shape `(nSamples, nInput)`, `target` is `None` / `"objs"` / `"cons"` |
| Output | Must return an `Eval` instance. Pass `target` through to trigger automatic cleanup. |

```python
from UQPyL.problem import Eval, Evaluator, Problem

class MyEvaluator(Evaluator):
    def evaluate(self, X, target=None):
        X = np.atleast_2d(X)
        objs = np.sum(X**2, axis=1, keepdims=True)
        cons = (X[:, 0] + X[:, 1] - 1.0).reshape(-1, 1)
        return Eval(objs=objs, cons=cons, target=target)

problem = Problem(
    nInput=2, nObj=1, nCon=1,
    lb=0.0, ub=1.0,
    evaluator=MyEvaluator(),
)
```

> **Note**: `evaluator` and `objFunc` / `conFunc` are mutually exclusive; they cannot be provided together.

---

### Simulation-Based Evaluation: `ModelProblem`

Use `ModelProblem` when a simulation model must run first and objectives or constraints are derived from the simulation output, as in model calibration, time-series simulation, or process-model workflows.

`ModelProblem` splits evaluation into two steps: `simFunc` produces `sims`, then `objFunc` / `conFunc` compute objectives and constraints from those simulation results. The bridge between the two steps is `simContext`.

#### simFunc

`simFunc` is the required callable for `ModelProblem` that runs the simulation model:

| Item | Contract |
|------|------|
| Input | `X`, shape `(nSamples, nInput)` |
| Output | `sims`, a 3D numeric array of shape `(nSamples, nTime, nSeries)` |
| NaN | Simulation output must not contain NaN values, except at positions marked by `mask` when a mask matching `obs.shape` is provided. |

```python
def simFunc(X):
    X = np.atleast_2d(X)
    nSamples = X.shape[0]
    sims = np.zeros((nSamples, 3, 1))            # (nSamples, nTime=3, nSeries=1)
    sims[:, 0, 0] = X[:, 0]                      # t=0: first parameter
    sims[:, 1, 0] = 0.5 * X[:, 0] + 0.5 * X[:, 1]  # t=1: mean of two parameters
    sims[:, 2, 0] = X[:, 1]                      # t=2: second parameter
    return sims
```

#### objFunc (ModelProblem)

`ModelProblem`'s `objFunc` has a different signature from `Problem`'s — it additionally receives `simContext`:

| Item | Contract |
|------|------|
| Input | `X` shape `(nSamples, nInput)`, `simContext` is a `SimContext` instance |
| Output | `objs`, shape `(nSamples, nObj)` |

`simContext` is constructed automatically after `simFunc` returns and is then passed into `objFunc` / `conFunc`. Users do not create it manually.

#### conFunc (ModelProblem)

Similar to `Problem`'s `conFunc`, but also receives `simContext`:

| Item | Contract |
|------|------|
| Input | `X` shape `(nSamples, nInput)`, `simContext` is a `SimContext` instance |
| Output | `cons`, shape `(nSamples, nCon)` |
| Feasibility | `cons <= 0` is considered feasible |

#### SimContext

`SimContext` is a frozen dataclass, automatically constructed after `simFunc` returns and then passed into `objFunc` and `conFunc`:

```python
@dataclass(frozen=True)
class SimContext:
    sims: np.ndarray           # simulation output, shape (nSamples, nTime, nSeries)
    obs: np.ndarray | None     # observation data, shape (nTime, nSeries)
    mask: np.ndarray | None    # missing data mask, same shape as obs; True = missing
```

A typical use case is to compute mean squared error between simulations and observations while excluding masked positions:

```python
def objFunc(X, simContext):
    err = simContext.sims - simContext.obs          # element-wise error
    if simContext.mask is not None:
        err = err[:, ~simContext.mask]              # exclude missing positions
    return np.mean(err**2, axis=(1, 2)).reshape(-1, 1)  # average over time and series dimensions
```

#### obs and mask

- **`obs`**: observation matrix, must be 2D `(nTime, nSeries)`. In a hydrological model, for example, `nTime` is the number of time steps and `nSeries` is the number of observation stations.
- **`mask`**: missing data mask, a boolean 2D array with exactly the same shape as `obs`. `True` indicates missing data at that position. When a `mask` is provided, NaN values are permitted in the simulation output at masked positions.

#### Example

```python
import numpy as np
from UQPyL.problem import ModelProblem

obs = np.array([[1.0], [0.8], [2.0]])            # observations (nTime=3, nSeries=1)
mask = np.array([[False], [True], [False]])       # second time step is missing

def simFunc(X):
    X = np.atleast_2d(X)
    nSamples = X.shape[0]
    sims = np.zeros((nSamples, 3, 1))            # (nSamples, nTime=3, nSeries=1)
    sims[:, 0, 0] = X[:, 0]                      # t=0
    sims[:, 1, 0] = 0.5 * X[:, 0] + 0.5 * X[:, 1]  # t=1
    sims[:, 2, 0] = X[:, 1]                      # t=2
    return sims

def objFunc(X, simContext):
    err = simContext.sims - simContext.obs        # element-wise error
    if simContext.mask is not None:
        err = err[:, ~simContext.mask]            # exclude missing positions
    return np.mean(err**2, axis=(1, 2)).reshape(-1, 1)  # average over time and series

problem = ModelProblem(
    nInput=2, nObj=1,
    lb=0.0, ub=3.0,
    simFunc=simFunc,             # simulation function (required)
    objFunc=objFunc,             # objective function
    obs=obs,                     # observation data
    mask=mask,                   # missing data mask
    seriesLabels=['Q'],          # series label
    name='MyModel',
)

res = problem.evaluate([[0.5, 1.5]])
```

#### simulate(): Simulation-Only Access

Use `simulate()` when only the simulation output is needed and objectives or constraints do not need to be computed:

```python
simContext = problem.simulate(X)                   # returns SimContext, .obs / .mask accessible
sims = problem.evaluate(X, target="sims").sims     # returns Eval, only .sims populated
```

`simulate()` returns the full `SimContext`, including observation data and mask. By contrast, `evaluate(X, target="sims")` returns an `Eval` object with only the `sims` field populated.

#### Custom ModelEvaluator

Subclass `ModelEvaluator` for more complex post-processing logic. `ModelEvaluator.evaluate()` contract:

| Item | Contract |
|------|------|
| Input | `X`, `simContext`, `target` |
| Output | Must return an `Eval` instance |

```python
from UQPyL.problem import Eval, ModelEvaluator, ModelProblem

class MyModelEvaluator(ModelEvaluator):
    def evaluate(self, X, simContext, target=None):
        err = simContext.sims - simContext.obs          # element-wise error
        if simContext.mask is not None:
            err = err[:, ~simContext.mask]              # exclude missing positions
        objs = np.mean(err**2, axis=(1, 2)).reshape(-1, 1)  # average over time and series
        return Eval(objs=objs, sims=simContext.sims, target=target)

problem = ModelProblem(
    nInput=2, nObj=1,
    lb=0.0, ub=3.0,
    simFunc=simFunc,
    obs=obs,
    evaluator=MyModelEvaluator(),
)
```

---

## Eval and evaluate()

`Eval` is the unified container for evaluation results. `evaluate()` is the single entry point through which all downstream modules access a problem.

### evaluate(): Unified Entry Point

Users define `objFunc`, `conFunc`, `simFunc`, and related callables to describe the evaluation logic. Downstream modules do not call these functions directly; they access the problem through `evaluate()`.

The internal workflow of `evaluate()`:

```text
validate input dimensions -> call user-defined functions -> validate output shapes -> wrap in Eval
```

For `Problem`, `evaluate()` directly calls `objFunc` / `conFunc`. For `ModelProblem`, `evaluate()` first calls `simFunc` to obtain simulation output, constructs `simContext`, then passes it to `objFunc` / `conFunc`.

### Eval Fields

```python
@dataclass
class Eval:
    objs: np.ndarray | None = None   # objective values (nSamples, nObj)
    cons: np.ndarray | None = None   # constraint values (nSamples, nCon)
    sims: np.ndarray | None = None   # simulation output (nSamples, nTime, nSeries)
```

**Shape Rules**:

| Field | Shape | Requirement |
|------|------|------|
| `objs` | `(nSamples, nObj)` | 2D numeric array |
| `cons` | `(nSamples, nCon)` | 2D numeric array |
| `sims` | `(nSamples, nTime, nSeries)` | 3D numeric array, no NaN (except at masked positions) |

**Null rule**: output blocks that are not provided must be `None`. Empty arrays are not a substitute for missing outputs.

Convenience properties on `Eval` instances:

```python
res = problem.evaluate(X)
print(res.objs)        # np.ndarray or None
print(res.hasObjs)     # bool, equivalent to res.objs is not None
print(res.hasCons)     # bool
print(res.hasSims)     # bool
```

### target: Selective Output

The `target` parameter in `evaluate(X, target=None)` controls which output blocks are returned. Unrequested blocks are set to `None`, which allows downstream modules to skip unnecessary computation.

`target` values supported by both object types:

| target | Meaning | Problem | ModelProblem |
|--------|------|:-------:|:------------:|
| `None` | Return all available output blocks | ✅ | ✅ |
| `"objs"` | Return objectives only; others are `None` | ✅ | ✅ |
| `"cons"` | Return constraints only; others are `None` | ✅ | ✅ |
| `"sims"` | Return simulation output only; others are `None` | ❌ | ✅ |

Behavior by problem type:

**Problem**:

| target | objs | cons | Additional validation |
|--------|:----:|:----:|------|
| `None` | Always returned | Returned when nCon > 0 | — |
| `"objs"` | ✅ | Forced to None | Error if cons is not None |
| `"cons"` | Forced to None | ✅ | Error if objs is not None |

**ModelProblem**:

| target | objs | cons | sims | Additional validation |
|--------|:----:|:----:|:----:|------|
| `None` | Returned if objFunc is present | Returned if conFunc is present | Always returned | — |
| `"objs"` | ✅ | Forced to None | Forced to None | Error if objs is None |
| `"cons"` | Forced to None | ✅ | Forced to None | Error if nCon>0 and cons is None |
| `"sims"` | Forced to None | Forced to None | ✅ | Error if objs or cons is not None |

> **Key constraint**: Simulation output is validated before evaluation. The returned `Eval` includes `sims` only for `target=None` or `target="sims"`; `"objs"` and `"cons"` return only the requested block.

Usage examples:

```python
# Assume `problem` is a Problem instance with nObj=1, nCon=1

# target=None: return all outputs
res = problem.evaluate(X)
print(res.objs)   # (nSamples, 1)
print(res.cons)   # (nSamples, 1)

# target="objs": compute objectives only, cons forced to None
res = problem.evaluate(X, target="objs")
print(res.objs)   # (nSamples, 1)
print(res.cons)   # None

# target="cons": compute constraints only, objs forced to None
res = problem.evaluate(X, target="cons")
print(res.objs)   # None
print(res.cons)   # (nSamples, 1)
```

For `ModelProblem`:

```python
# target="sims": return simulation output only, skipping objFunc / conFunc
res = problem.evaluate(X, target="sims")
print(res.sims)   # (nSamples, nTime, nSeries)
print(res.objs)   # None
print(res.cons)   # None
```

---

## Built-in Test Problems

`UQPyL.problem` provides a collection of classic test functions that can be instantiated directly.

### Single-Objective (SOP)

```python
from UQPyL.problem import Sphere, Rosenbrock, Ackley, Griewank, Rastrigin

problem = Sphere(nInput=10)
problem = Rosenbrock(nInput=5)
```

Full list: `Sphere`, `Schwefel_2_22`, `Schwefel_1_22`, `Schwefel_2_21`, `Rosenbrock`, `Step`, `Quartic`, `Schwefel_2_26`, `Rastrigin`, `Ackley`, `Griewank`, `Trid`, `Bent_Cigar`, `Discus`, `Weierstrass`, `RosenbrockWithCon`

### Multi-Objective (MOP)

```python
from UQPyL.problem import ZDT1, ZDT2, DTLZ1, DTLZ2

problem = ZDT1(nInput=30)
problem = DTLZ2(nInput=12, nObj=3)
```

ZDT family: `ZDT1`, `ZDT2`, `ZDT3`, `ZDT4`, `ZDT6`
DTLZ family: `DTLZ1`—`DTLZ7`

---

## singleFunc Decorator

`singleFunc` wraps a single-sample function into a batch function, eliminating the need for manual `np.atleast_2d` and dimension handling:

```python
from UQPyL.problem import singleFunc, Problem

@singleFunc
def objFunc(x):                        # x: (nInput,) — single sample
    return x[0]**2 + x[1]**2           # returns scalar

problem = Problem(nInput=2, nObj=1, lb=-1, ub=1, objFunc=objFunc)
res = problem.evaluate([[0.2, 0.3], [0.5, 0.6]])   # batch call works correctly
```

Equivalent manual implementation:

```python
def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)
```

---

## Parameter Reference

### Problem Constructor Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `nInput` | `int` | — | Number of input variables |
| `nObj` | `int` | — | Number of objectives (required) |
| `lb` | `int/float/list/array` | — | Input lower bound |
| `ub` | `int/float/list/array` | — | Input upper bound |
| `objFunc` | `callable` | `None` | Objective function |
| `conFunc` | `callable` | `None` | Constraint function |
| `nCon` | `int` | `0` | Number of constraints |
| `optType` | `str/list` | `'min'` | Optimization direction |
| `conWgt` | `list` | `None` | Constraint weights, shape `(1, nCon)` |
| `varType` | `list` | All continuous | Variable types: `0`=continuous, `1`=integer, `2`=discrete |
| `varSet` | `dict` | `None` | Discrete value sets, e.g. `{2: [0.1, 0.3, 0.5]}` |
| `xLabels` | `list` | Auto-generated when omitted | Input variable labels |
| `objLabels` | `list` | Auto-generated when omitted | Objective labels |
| `conLabels` | `list` | Auto-generated when omitted | Constraint labels |
| `name` | `str` | Class name | Problem name |
| `space` | `SpaceBase` | Auto-generated when omitted | Custom input space |
| `evaluator` | `EvaluatorBase` | Auto-generated when omitted | Custom evaluator |

### ModelProblem Additional Parameters

| Parameter | Type | Default | Description |
|------|------|--------|------|
| `simFunc` | `callable` | — | Simulation function (required) |
| `obs` | `np.ndarray` | `None` | Observation matrix `(nTime, nSeries)` |
| `mask` | `np.ndarray` | `None` | Missing data mask, same shape as obs |
| `seriesLabels` | `list` | Auto-generated when omitted | Series labels |
| `evaluator` | `ModelEvaluatorBase` | Auto-generated when omitted | Custom simulation evaluator |

### ProblemBase Instance Properties

| Property | Type | Description |
|------|------|------|
| `nInput` | `int` | Input dimensionality |
| `nObj` | `int` | Objective dimensionality |
| `nCon` | `int` | Constraint dimensionality |
| `lb` | `np.ndarray` | Lower bounds `(1, nInput)` |
| `ub` | `np.ndarray` | Upper bounds `(1, nInput)` |
| `optType` | `str` | Optimization direction string |
| `opt` | `int/array` | `1`=minimize, `-1`=maximize |
| `varType` | `np.ndarray` | Variable type codes |
| `idxF` | `np.ndarray` | Continuous variable indices |
| `idxI` | `np.ndarray` | Integer variable indices |
| `idxD` | `np.ndarray` | Discrete variable indices |
| `varSet` | `dict` | Discrete value sets |
| `xLabels` | `list` | Input variable labels |
| `objLabels` | `list` | Objective labels |
| `conLabels` | `list` | Constraint labels (`None` when nCon=0) |
| `conWgt` | `np.ndarray` | Constraint weights |

### ModelProblem Additional Properties

| Property | Type | Description |
|------|------|------|
| `obs` | `np.ndarray` | Observation matrix |
| `mask` | `np.ndarray` | Missing data mask |
| `obsShape` | `tuple` | `obs.shape` |
| `nObs` | `int` | Total flattened observation length |
| `seriesLabels` | `list` | Series labels |

### Constraint weights

`Problem(conWgt=[10, 1], nCon=2, ...)` assigns one finite nonnegative weight per constraint. The length must match `nCon`. `None` leaves violations unweighted; a zero weight ignores that constraint, including in feasibility checks.

```python
CV = np.sum(np.maximum(0, cons * conWgt), axis=1)
```
