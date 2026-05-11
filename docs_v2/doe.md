# Design of Experiment

The `doe` module generates input samples from a `Problem` or `ModelProblem` input space.

Use DOE when you need model evaluation points for sensitivity analysis, surrogate training, calibration candidates, plotting, or optimizer initialization.

## What Sampling Does

A sampler first creates points in unit space, then maps them to the bounds defined by the problem.

```text
unit samples in [0, 1] -> problem.unit_to_space(...) -> samples in [lb, ub]
```

For example, if `x1 in [0, 10]` and `x2 in [-1, 1]`, the returned sample matrix already uses those real problem bounds.

## Basic Workflow

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=[0.0, -1.0], ub=[10.0, 1.0], objFunc=objFunc, name="BoundedSphere")

X = LHS("classic").sample(problem, nSamples=5, seed=123)
Y = problem.evaluate(X).objs

print(X)
print(Y.shape)
```

Example output:

```text
[[ 7.84669     0.1248378 ]
 [ 4.3518118   0.95595708]
 [ 1.36470373 -0.97847159]
 [ 2.44071975 -0.52625128]
 [ 9.63950912  0.31062976]]
(5, 1)
```

`X` has shape `(n_samples, n_input)`. `Y` has shape `(n_samples, n_obj)`.

## Choose a Sampler

Start from the task, not the class name.

| Task | Recommended sampler | Why |
|---|---|---|
| General model exploration | `LHS("classic")` | Covers each variable range more evenly than plain random sampling. |
| Small quick smoke test | `Random()` | Simple and cheap. Good for checking code paths. |
| Grid over a few variables | `FFD()` | Deterministic full factorial grid. Useful when dimensions and levels are small. |
| Low-discrepancy sequence | `Sobol()` | Useful when you want structured space-filling samples. |
| Sobol sensitivity analysis | `SaltelliDesign()` | Produces the exact design metadata `Sobol` analysis expects. |
| FAST sensitivity analysis | `FASTDesign()` | Produces the exact design metadata `FAST` analysis expects. |
| Morris screening | `MorrisDesign()` | Produces Morris trajectories and metadata. |

Practical default: use `LHS("classic")` unless a downstream method explicitly requires a special design.

## How Many Samples

There is no universal sample count. The right number depends on dimension, model cost, and the downstream method.

| Goal | Starting point |
|---|---|
| Smoke test | `nSamples = 5` to `20` |
| Plotting or rough exploration | `nSamples = 20` to `100` |
| Surrogate training | Start around `10 * nInput`, then validate on held-out data. |
| RBDFAST / RSA / DeltaTest | Start around `100` to `500`, depending on `nInput`. |
| Sobol analysis | Use `SaltelliDesign`; choose base `N` such as `128`, `256`, or `512`. |
| FAST analysis | Use `FASTDesign`; `N` must satisfy `N > 4 * M^2`. |
| Morris screening | Use `numTrajectory`, often `10` to `50` for a first pass. |

For expensive simulations, begin small and increase only after the workflow is correct.

## Reproducible Sampling

Pass `seed` when you need repeatable samples.

```python
import numpy as np

from UQPyL.doe import Random
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

X1 = Random().sample(problem, nSamples=4, seed=42)
X2 = Random().sample(problem, nSamples=4, seed=42)

print(X1)
print((X1 == X2).all())
```

Example output:

```text
[[0.77395605 0.43887844]
 [0.85859792 0.69736803]
 [0.09417735 0.97562235]
 [0.7611397  0.78606431]]
True
```

Use the same seed for reproducibility. Change the seed when you want a different random design.

## `sample()` vs `sampleWithMeta()`

Use `sample()` when you only need the sample matrix.

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

X = LHS("classic").sample(problem, nSamples=6, seed=123)
print(X.shape)
```

Use `sampleWithMeta()` when a downstream method needs design metadata.

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

X, meta = LHS("classic").sampleWithMeta(problem, nSamples=6, seed=123)

print(X.shape)
print(meta)
```

Example output:

```text
(6, 2)
{'designType': 'lhs', 'criterion': 'classic', 'iterations': 5, 'seed': 123}
```

Metadata records how the design was generated. Some analysis methods require it.

## What `meta` Is For

`meta` is a small dictionary that describes the sampling design. It is not model output. It tells downstream methods how to interpret the rows of `X`.

For ordinary sampling, `meta` is mostly record keeping:

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

X, meta = LHS("classic").sampleWithMeta(problem, nSamples=6, seed=123)

print(meta["designType"])
print(meta["criterion"])
print(meta["seed"])
```

Example output:

```text
lhs
classic
123
```

For method-specific analysis designs, `meta` is part of the required input. For example, `Sobol` needs to know that `X` came from `SaltelliDesign`, whether second-order samples were generated, and what base sample size was used.

```python
import numpy as np

from UQPyL.analysis import Sobol
from UQPyL.doe import SaltelliDesign
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] + X[:, 1]).reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

X, meta = SaltelliDesign(secondOrder=False).sampleWithMeta(problem, N=64, seed=123)
Y = problem.evaluate(X).objs

result = Sobol(verboseFlag=False).analyze(problem, X, Y=Y, meta=meta)

print(meta)
print(result.metricNames)
```

Example output:

```text
{'designType': 'saltelli', 'N': 64, 'secondOrder': False, 'skipValue': 0, 'scramble': True, 'blockSize': 4, 'seed': 123}
['S1', 'S1_norm', 'ST', 'ST_norm']
```

Rule of thumb:

| Situation | Use `meta`? |
|---|---|
| You only need `X` for evaluation, optimization initialization, or surrogate training | Optional |
| You need reproducibility records for a design | Useful |
| You will run `Sobol`, `FAST`, or `Morris` analysis | Required |
| You are not sure whether the downstream method needs metadata | Use `sampleWithMeta()` and keep it |

## Full Factorial Design

`FFD` builds a grid. It uses `levels`, not `nSamples`.

```python
import numpy as np

from UQPyL.doe import FFD
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=[0.0, -1.0], ub=[1.0, 1.0], objFunc=objFunc)

X, meta = FFD().sampleWithMeta(problem, levels=[3, 4])

print(X.shape)
print(X[:5])
print(meta["levels"])
```

Example output:

```text
(12, 2)
[[ 0.         -1.        ]
 [ 0.         -0.33333333]
 [ 0.          0.33333333]
 [ 0.          1.        ]
 [ 0.5        -1.        ]]
[3, 4]
```

For two variables with `[3, 4]` levels, the design contains `3 * 4 = 12` rows. Full factorial designs grow quickly as dimensions increase.

## Designs for Sensitivity Analysis

Some sensitivity methods need special sampling patterns. Use the matching DOE class and keep the returned `meta`.

### Sobol

```python
import numpy as np

from UQPyL.analysis import Sobol
from UQPyL.doe import SaltelliDesign
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] + 0.1 * X[:, 1]
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

X, meta = SaltelliDesign(secondOrder=False).sampleWithMeta(problem, N=64, seed=123)
Y = problem.evaluate(X).objs
result = Sobol(verboseFlag=False).analyze(problem, X, Y=Y, meta=meta)

print(X.shape)
print(meta["designType"])
print(result.metricNames)
```

Example output:

```text
(256, 2)
saltelli
['S1', 'S1_norm', 'ST', 'ST_norm']
```

For `D` inputs and base size `N`, `SaltelliDesign(secondOrder=False)` returns `(D + 2) * N` rows.

### FAST

```python
import numpy as np

from UQPyL.doe import FASTDesign
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X, axis=1, keepdims=True)


problem = Problem(nInput=3, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

X, meta = FASTDesign(M=4).sampleWithMeta(problem, N=256, seed=123)

print(X.shape)
print(meta["designType"])
print(meta["M"])
```

Example output:

```text
(768, 3)
fast
4
```

For FAST, `N` must satisfy `N > 4 * M^2`. With `M=4`, use `N > 64`.

### Morris

```python
import numpy as np

from UQPyL.doe import MorrisDesign
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X, axis=1, keepdims=True)


problem = Problem(nInput=3, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

X, meta = MorrisDesign(numLevels=4).sampleWithMeta(problem, numTrajectory=5, seed=123)

print(X.shape)
print(meta["designType"])
print(meta["trajectorySize"])
```

Example output:

```text
(20, 3)
morris
4
```

For `D` inputs and `numTrajectory` trajectories, Morris returns `numTrajectory * (D + 1)` rows.

## Method Matching

| Downstream method | Use this design | Keep `meta`? |
|---|---|---|
| `RBDFAST` | `LHS`, `Random`, or another ordinary sample matrix | Optional |
| `RSA` | `LHS`, `Random`, or another ordinary sample matrix | Optional |
| `DeltaTest` | `LHS`, `Random`, or another ordinary sample matrix | Optional |
| `Sobol` | `SaltelliDesign` | Required |
| `FAST` | `FASTDesign` | Required |
| `Morris` | `MorrisDesign` | Required |

If an analysis method expects `meta`, generate samples with `sampleWithMeta()` and pass both `X` and `meta`.

## Common Mistakes

| Mistake | Why it matters | Fix |
|---|---|---|
| Using `sample()` for Sobol/FAST/Morris analysis | The analysis method needs design metadata. | Use `sampleWithMeta()` and pass `meta`. |
| Treating `nSamples` as final row count for Saltelli | Saltelli expands base `N` into more rows. | Check `X.shape` after sampling. |
| Using `FFD` in high dimensions | Grid size grows as the product of levels. | Use `LHS` or `Sobol` for high-dimensional exploration. |
| Forgetting `seed` | Results change between runs. | Pass `seed` for reproducible examples and tests. |
| Sampling before checking bounds | Bad bounds produce bad samples. | Print `problem.lb`, `problem.ub`, and a few sample rows. |

## Next Steps

| Goal | Read |
|---|---|
| Look up sampler constructors and metadata fields | [DOE API](api/doe.md) |
| Define input bounds and variable types | [Problem](problem.md) |
| Use samples for sensitivity analysis | [Analysis](analysis.md) |
| Use samples for surrogate training | [Surrogate Modeling](surrogate.md) |
