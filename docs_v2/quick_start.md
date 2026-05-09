# Quick Start

This page shows the shortest path through the current UQPyL architecture.

```text
Problem -> Method -> Result
```

## Define a Problem

In UQPyL, a `Problem` is the shared contract between your model and the functional modules. It defines the input space, evaluation rule, and direction. Once a problem is defined, samplers, analysis methods, optimizers, inference methods, and calibration workflows can consume it through a consistent interface.

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
    optType="min",
    name="Sphere2D",
)

res = problem.evaluate([[0.2, 0.3]])
print(res.objs)
```

`evaluate()` returns an `Eval` object. Objective values are available from `res.objs`.

## Runtime Controls

Runnable methods may print progress, write logs, or save structured results.

| Option | Role |
|---|---|
| `verboseFlag` | Controls terminal progress and summary output. |
| `logFlag` | Writes a text log when enabled. |
| `saveFlag` | Saves structured runtime data, usually sqlite, when enabled. |

Example:

```python
from UQPyL.optimization.soea import GA


algorithm = GA(
    nPop=6,
    maxFEs=200,
    verboseFlag=True,
    verboseFreq=10,
    logFlag=True,
    saveFlag=True,
    saveFreq=50,
)
```

With these options enabled, a runnable method may print settings, progress, and a final summary:

```text
Algorithm: GA
Problem: Sphere2D
nInput: 2
nObj: 1
maxFEs: 200
maxIters: 1000
GA | iter=0 eval=6 best=2.3565e-01 cv=0 time=0.0s
Optimization finished
  algorithm        : GA
  status           : finished
  iterations       : 3
  evaluations      : 18
  best value       : 3.8711e-02
  best X           : [1.0035e-01, -1.6923e-01]
  constraint viol. : 0
  elapsed          : 0.0s
```

When `saveFlag=True`, structured runtime data is written under the problem's `workDir` if set, otherwise under the current working directory. For example:

```text
Result/ga_Sphere2D_20260509_0946_3370.sqlite3
```

`logFlag=True` enables module-specific text logging when that method implements it.

These controls are shared across the main functional modules, but the exact output format can differ by module. See each module page for its result object, saved artifacts, reader class, and snapshot behavior.

In this quick start, these options are set to `False` so examples stay quiet and do not create result files.

## Generate Samples

```python
from UQPyL.doe import LHS


sampler = LHS("classic")
X = sampler.sample(problem, nSamples=10, seed=123)
Y = problem.evaluate(X).objs

print(X.shape)
print(Y.shape)
```

Sampling methods read the bounds and variable types from `problem`.

## Run Analysis

```python
from UQPyL.analysis import RBDFAST
from UQPyL.doe import LHS


sampler = LHS("classic")
X = sampler.sample(problem, nSamples=256, seed=123)
Y = problem.evaluate(X, target="objs").objs

analysis = RBDFAST(verboseFlag=False)
result = analysis.analyze(
    problem,
    X,
    Y=Y,
    target="objs",
)

print(result.metricNames)
print(result.getMetric("S1").values)
```

Analysis methods return `AnaResult`.

## Run Optimization

```python
from UQPyL.optimization.soea import GA


algorithm = GA(maxFEs=200, verboseFlag=False, logFlag=False, saveFlag=False)
result = algorithm.run(
    problem,
    seed=123,
)

print(result.bestDecs)
print(result.bestObjs)
```

Optimization methods return `OptResult`.

## Run Inference

```python
from UQPyL.inference import MH


inference = MH(
    nChains=3,
    warmUp=5,
    maxIters=30,
    verboseFlag=False,
    logFlag=False,
    saveFlag=False,
)
result = inference.run(problem, gamma=0.2, seed=123)

print(result.decs.shape)
print(result.acceptanceRate)
```

Inference methods currently expect scalar-objective problems and return `InfResult`.

## Calibrate a Simulation Model

```python
import numpy as np

from UQPyL.calibration import GLUE
from UQPyL.problem import ModelProblem


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


modelProblem = ModelProblem(
    nInput=2,
    ub=3.0,
    lb=0.0,
    simFunc=simFunc,
    obs=obs,
    simLabels=["Q"],
    name="ToyModel",
)

X = np.array([
    [1.0, 2.0],
    [1.0, 2.4],
    [0.0, 0.0],
])

calibrator = GLUE(verboseFlag=False, logFlag=False, saveFlag=False)
result = calibrator.run(modelProblem, X, threshold=0.3)

print(result.bestDecs)
print(result.behavioralDecs)
```

Calibration methods consume `ModelProblem` and return `CalResult`.

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
