# Optimization

The `optimization` module searches for good decision variables for a `Problem`.

UQPyL optimization algorithms are easiest to understand in three groups:

| Group | Use when | Import path | Algorithms |
|---|---|---|---|
| Single-objective | You have one objective, such as one loss, cost, error, or score. | `UQPyL.optimization.soea` | `GA`, `PSO`, `DE`, `SCE_UA`, `ML_SCE_UA`, `CSA`, `ABC` |
| Multi-objective | You have two or more objectives and want a Pareto set. | `UQPyL.optimization.moea` | `NSGAII`, `NSGAIII`, `MOEAD`, `RVEA` |
| Expensive optimization | Each real model evaluation is costly, so a surrogate-assisted search is useful. | `UQPyL.optimization.expensive` | `ASMO`, `EGO`, `MOASMO` |

The core workflow is the same:

```text
define Problem -> choose algorithm -> algorithm.run(problem, seed=...) -> OptResult
```

## Choose the Right Group

Start from the problem, not from the algorithm name.

| Your situation | Start with | Why |
|---|---|---|
| One objective and evaluations are cheap enough to run hundreds or thousands of times. | `GA` or `DE` | Simple population-based search for ordinary single-objective problems. |
| One objective and you want a hydrology-style shuffled complex optimizer. | `SCE_UA` | Common for calibration-like continuous parameter search. |
| Two or more objectives and you want a basic Pareto front. | `NSGAII` | Good default multi-objective optimizer. |
| Many objectives or reference-vector style search is needed. | `NSGAIII` or `RVEA` | Better fit for higher-dimensional objective spaces. |
| One expensive objective. | `ASMO` or `EGO` | Uses a surrogate to propose new real evaluations. |
| Multiple expensive objectives. | `MOASMO` | Surrogate-assisted multi-objective optimization. |

Practical default: use `GA` for the first single-objective run, `NSGAII` for the first multi-objective run, and `ASMO` when objective evaluations are expensive.

## What an Optimizer Needs

An optimizer needs a `Problem` with:

| Problem field | Meaning for optimization |
|---|---|
| `nInput` | Number of decision variables. |
| `lb`, `ub` | Lower and upper bounds searched by the algorithm. |
| `objFunc` | Objective function. It should accept batched `X` and return `(n_samples, n_obj)`. |
| `optType` | `"min"`, `"max"`, or a list such as `["min", "max"]`. |
| `conFunc`, `nCon` | Optional constraints. Constraint values `<= 0` are feasible. |
| `varType`, `varSet` | Optional variable type settings for integer or discrete variables. |

Internally, optimizers minimize. If `optType="max"`, UQPyL converts the objective for search and reports `bestObjs` in the original objective direction.

## Single-Objective Optimization

Single-objective algorithms return one best solution.

This example minimizes:

```text
f(x) = x1^2 + x2^2
```

```python
import numpy as np

from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="Sphere2D")
algorithm = GA(nPop=8, maxFEs=40, maxIters=5, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)

print(result.bestDecs)
print(result.bestObjs)
print(result.FEs, result.iters, result.bestFeasible)
```

Example output:

```text
[[0.003  0.1049]]
[[0.011]]
40 5 True
```

Read this as:

| Output | Meaning |
|---|---|
| `bestDecs` | Best decision vector found. |
| `bestObjs` | Objective value at `bestDecs`. |
| `FEs` | Number of real problem evaluations used. |
| `iters` | Number of optimization iterations completed. |
| `bestFeasible` | Whether the best solution satisfies constraints. |

For this toy problem the true optimum is `[0, 0]`, so a small objective value near zero is expected.

## Single-Objective Algorithm Choices

| Algorithm | Good first use |
|---|---|
| `GA` | General single-objective search with crossover and mutation. |
| `DE` | Continuous-variable search with differential mutation. |
| `PSO` | Particle-swarm search for continuous variables. |
| `SCE_UA` | Shuffled complex evolution, often used for calibration-like continuous problems. |
| `ML_SCE_UA` | Multi-level variant of `SCE_UA`. |
| `CSA` | Crow search algorithm. |
| `ABC` | Artificial bee colony algorithm. |

For constructor parameters, see [Optimization API](api/optimization.md).

## Multi-Objective Optimization

Multi-objective algorithms return a set of non-dominated solutions, not one single best solution.

Use them when objectives conflict. For example, you may want low cost and high reliability at the same time.

```python
import numpy as np

from UQPyL.optimization.moea import NSGAII
from UQPyL.problem import ZDT1

np.set_printoptions(precision=4, suppress=True)


problem = ZDT1(nInput=5)
algorithm = NSGAII(nPop=12, maxFEs=48, maxIters=3, verboseFlag=False, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)

print(result.bestDecs.shape)
print(result.bestObjs.shape)
print(result.bestObjs[:3])
print(round(result.bestMetric, 4), result.FEs, result.iters)
```

Example output:

```text
(8, 5)
(8, 2)
[[0.5078 1.8356]
 [0.5826 1.7025]
 [0.0076 2.3533]]
0.4891 48 4
```

Read this as:

| Output | Meaning |
|---|---|
| `bestDecs.shape == (8, 5)` | The current Pareto set has 8 decision vectors, each with 5 variables. |
| `bestObjs.shape == (8, 2)` | The corresponding Pareto objective matrix has 8 rows and 2 objectives. |
| `bestMetric` | Current multi-objective progress metric, currently hypervolume. Larger is better for progress tracking. |

Do not pick the first row blindly. In a Pareto set, each row is a trade-off. Choose a final solution using domain preference, plotting, or a decision rule.

## Multi-Objective Algorithm Choices

| Algorithm | Good first use |
|---|---|
| `NSGAII` | Default two-objective or small multi-objective run. |
| `NSGAIII` | Many-objective or reference-direction based search. |
| `MOEAD` | Decomposition-based multi-objective search. |
| `RVEA` | Reference-vector guided search. |

Multi-objective `optType` can be a list:

```text
problem = Problem(nInput=3, nObj=2, lb=0.0, ub=1.0, objFunc=objFunc, optType=["min", "max"])
```

The algorithm handles objective directions internally.

## Expensive Optimization

Expensive optimization is for problems where one real evaluation is costly.

Examples:

| Expensive evaluation | Why expensive optimization helps |
|---|---|
| A hydrological simulation that takes minutes. | Avoids thousands of direct model calls. |
| A CFD or structural simulation. | Uses surrogate predictions to propose promising points. |
| A lab experiment or external executable. | Reduces the number of real evaluations. |

Expensive algorithms usually follow this pattern:

```text
initial real samples -> fit surrogate -> optimize surrogate -> evaluate selected real point(s) -> repeat
```

Use this group only when the objective is expensive enough to justify surrogate overhead. For cheap mathematical functions, ordinary `GA`, `DE`, or `NSGAII` is usually simpler.

## Single-Objective Expensive Optimization

`ASMO` and `EGO` are single-objective expensive optimizers.

```python
import numpy as np

from UQPyL.optimization.expensive import ASMO
from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="Sphere2D")
algorithm = ASMO(nInit=6, optimizer=GA(nPop=6, maxFEs=18, maxIters=3, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False), maxFEs=12, maxIters=2, verboseFlag=False, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)

print(result.bestDecs)
print(result.bestObjs)
print(result.FEs, result.iters)
```

Example output:

```text
[[0.0981 0.1052]]
[[0.0207]]
9 4
```

`nInit=6` means the algorithm starts from six real evaluations. Later iterations fit a surrogate and add new real evaluations.

## Use Initial Data in Expensive Optimization

If you already have evaluated samples, pass them as `xInit` and `yInit`.

```python
import numpy as np

from UQPyL.optimization.expensive import ASMO
from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="Sphere2D")
xInit = np.array([[0.8, 0.8], [0.2, 0.1], [-0.5, 0.3]])
yInit = objFunc(xInit)

algorithm = ASMO(nInit=6, optimizer=GA(nPop=6, maxFEs=18, maxIters=3, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False), maxFEs=8, maxIters=1, verboseFlag=False, logFlag=False, saveFlag=False)
result = algorithm.run(problem, xInit=xInit, yInit=yInit, seed=123)

print(yInit)
print(result.FEs)
print(result.bestObjs)
```

Example output:

```text
[[1.28]
 [0.05]
 [0.34]]
5
[[0.0207]]
```

If `xInit` has fewer rows than `nInit`, the algorithm adds extra initial samples internally. If `yInit` is omitted, UQPyL evaluates `xInit` with the real problem.

## Multi-Objective Expensive Optimization

Use `MOASMO` when the real problem has multiple expensive objectives.

```python
import numpy as np

from UQPyL.optimization.expensive import MOASMO
from UQPyL.optimization.moea import NSGAIII
from UQPyL.problem import ZDT1

np.set_printoptions(precision=4, suppress=True)


problem = ZDT1(nInput=6, ub=1.0, lb=0.0)
inner = NSGAIII(nPop=12, maxFEs=40, maxIters=2, verboseFlag=False, logFlag=False, saveFlag=False)
algorithm = MOASMO(optimizer=inner, pct=0.5, nInit=6, maxFEs=20, maxIters=2, verboseFlag=False, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)

print(result.bestDecs.shape)
print(result.bestObjs.shape)
print(result.bestObjs[:3])
```

Example output from a small run:

```text
(4, 6)
(4, 2)
[[0.     4.6318]
 [0.006  2.0188]
 [0.7385 1.1758]]
```

The exact Pareto set can vary with surrogate behavior and sample budget. Use larger budgets for real interpretation.

## Work With Constraints

A constrained `Problem` provides `conFunc` and `nCon`.

Constraint convention:

```text
constraint value <= 0 means feasible
constraint value > 0 means violation
```

```python
import numpy as np

from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


def conFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] + X[:, 1] - 0.5).reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, nCon=1, lb=-1.0, ub=1.0, objFunc=objFunc, conFunc=conFunc, optType="min", name="ConstrainedSphere2D")
algorithm = GA(nPop=8, maxFEs=40, maxIters=5, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)

print(result.bestDecs)
print(result.bestObjs)
print(result.bestCons)
print(result.bestFeasible)
```

Example output:

```text
[[0.044  0.0452]]
[[0.004]]
[[-0.4108]]
True
```

The best constraint value is negative, so the best solution is feasible.

## Control Runtime Budget

The most important budget parameters are:

| Parameter | Meaning |
|---|---|
| `maxFEs` | Maximum number of real problem evaluations. Usually the most important budget. |
| `maxIters` | Maximum number of algorithm iterations. |
| `nPop` | Population size for many evolutionary algorithms. |
| `nInit` | Initial real sample count for expensive algorithms. |
| `maxTolerates` | Stop after too many non-improving iterations. |
| `tolerate` | Minimum improvement size used by tolerance stopping. |

For normal algorithms, `maxFEs` counts direct evaluations of the `Problem`.

For expensive algorithms, `maxFEs` still counts real problem evaluations, not cheap surrogate predictions.

## Use Verbose Output

Set `verboseFlag=True` to print progress and the final summary.

```python
import numpy as np

from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="Sphere2D")
algorithm = GA(nPop=8, maxFEs=24, maxIters=3, tolerate=None, verboseFlag=True, verboseFreq=1, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)
```

Example output:

```text
Algorithm: GA
Problem: Sphere2D
nInput: 2
nObj: 1
maxFEs: 24
maxIters: 3
GA | iter=0 eval=8 best=1.8884e-02 cv=0 time=0.0s
Optimization finished
  algorithm        : GA
  status           : finished
  iterations       : 3
  evaluations      : 24
  best value       : 1.1468e-02
  best X           : [3.0409e-03, 1.0704e-01]
  constraint viol. : 0
  elapsed          : 0.0s
```

Use `verboseFlag=False` in batch scripts when you only need `OptResult`.

## Read `OptResult`

Every optimization run returns an `OptResult`.

| Field | Meaning |
|---|---|
| `bestDecs` | Best decision matrix. For multi-objective runs, this is the current Pareto decision set. |
| `bestObjs` | Best objective matrix. For multi-objective runs, this is the Pareto objective set. |
| `bestCons` | Constraint values for `bestDecs`, when constraints exist. |
| `bestFeasible` | Whether the reported best solution or set is feasible. |
| `bestMetric` | Multi-objective progress metric, currently hypervolume. `None` for single-objective runs. |
| `FEs` | Final number of real problem evaluations. |
| `iters` | Final number of iterations. |
| `appearFEs` | Evaluation count when the best result appeared. |
| `appearIters` | Iteration count when the best result appeared. |
| `history` | Optimization history snapshots. |
| `summary()` | Compact runtime summary dictionary. |

For saved sqlite runs and reader methods, see [Optimization API](api/optimization.md).

### Read an In-Memory Result

Use this pattern immediately after `algorithm.run(...)`.

```python
import numpy as np

from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="Sphere2D")
algorithm = GA(nPop=8, maxFEs=40, maxIters=5, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)

bestX = result.bestDecs[0]
bestY = float(result.bestObjs[0, 0])
recentHistory = result.history.bestObjHistory[-3:]

print(bestX)
print(bestY)
print(recentHistory)
print(result.summary()["fes"], result.summary()["iters"])
```

Example output:

```text
[0.003  0.1049]
0.011019868612514926
[0.01146775171564494, 0.01146775171564494, 0.011019868612514926]
40 5
```

For single-objective runs, `result.bestDecs[0]` is the best decision vector and `result.bestObjs[0, 0]` is its objective value. For multi-objective runs, `result.bestDecs` and `result.bestObjs` contain multiple Pareto rows.

### Read a Saved SQLite Result

Set `saveFlag=True` to save a sqlite file under `Result/`.

```python
from pathlib import Path

import numpy as np

from UQPyL.optimization.runtime import OptReader
from UQPyL.optimization.soea import GA
from UQPyL.problem import Sphere

np.set_printoptions(precision=4, suppress=True)


problem = Sphere(nInput=2, ub=1.0, lb=-1.0)

resultDir = Path("Result")
before = set(resultDir.glob("*.sqlite3")) if resultDir.exists() else set()

algorithm = GA(nPop=6, maxFEs=18, maxIters=3, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=True, saveFreq=2)
algorithm.run(problem, seed=123)

after = set(resultDir.glob("*.sqlite3"))
dbPath = sorted(after - before)[0]

reader = OptReader(dbPath)
summary = reader.get_run_summary()
params = reader.get_run_params()
snapshots = reader.list_snapshots()
best = reader.load_last_best()
reader.close()

print(dbPath.as_posix())
print(summary["method"], summary["problem_name"])
print(summary["final_fes"], summary["final_iters"])
print(params["seed"])
print(len(snapshots))
print(best.decs)
print(best.objs)
```

Example output:

```text
Result/ga_Sphere_20260510_1709_9d67.sqlite3
GA Sphere
18 3
123
3
[[ 0.1004 -0.1692]]
[[0.0387]]
```

If you already know the sqlite path, you only need the reader part:

```text
from UQPyL.optimization.runtime import OptReader

dbPath = "Result/ga_Sphere_20260510_1709_9d67.sqlite3"

reader = OptReader(dbPath)
summary = reader.get_run_summary()
best = reader.load_last_best()
reader.close()
```

Use `summary` for run metadata, and use `best.decs`, `best.objs`, and `best.cons` to inspect the last saved best population.

Saved runs include a serialized problem payload. If you define `objFunc` interactively inside a notebook cell or temporary script, Python may not be able to pickle it. For persistent sqlite runs, prefer importable problem classes or objective functions defined in importable modules.

## Common Mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Using a single-objective algorithm on a multi-objective problem | The result does not represent a Pareto search. | Use `NSGAII`, `NSGAIII`, `MOEAD`, `RVEA`, or `MOASMO`. |
| Treating the first Pareto row as the best solution | You may choose an arbitrary trade-off. | Plot or rank the Pareto set using domain preference. |
| Setting `maxFEs` too small | The algorithm stops before meaningful search. | Use tiny budgets only for smoke tests; increase for interpretation. |
| Forgetting `optType` | Objective direction can be wrong. | Set `"min"`, `"max"`, or a list for multiple objectives. |
| Writing `conFunc` with reversed sign | Feasible and infeasible points are confused. | Return values `<= 0` for feasible constraints. |
| Using expensive optimization for cheap functions | Surrogate overhead may dominate. | Use ordinary `GA`, `DE`, or `NSGAII` for cheap objectives. |
| Passing `xInit` without matching `yInit` rows | Existing evaluations cannot be reused correctly. | Make `xInit.shape[0] == yInit.shape[0]`, or omit `yInit` and let UQPyL evaluate. |

## Next Steps

| Goal | Read |
|---|---|
| Define objectives and constraints | [Problem](problem.md) |
| Look up algorithm constructors and result fields | [Optimization API](api/optimization.md) |
| Generate initial samples | [Design of Experiment](doe.md) |
| Train surrogate models | [Surrogate Modeling](surrogate.md) |
| See complete workflows | [Examples](examples.md) |
