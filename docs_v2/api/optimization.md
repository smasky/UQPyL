# Optimization API

## `UQPyL.optimization`

The `optimization` module searches for optimal decision variables for a `Problem`.

### Import

```python
from UQPyL.optimization.soea import GA, PSO, DE
from UQPyL.optimization.moea import NSGAII, MOEAD
from UQPyL.optimization.expensive import EGO
from UQPyL.optimization.runtime import OptReader
```

### Public Objects

Top-level objects:

| Object | Role |
|---|---|
| `AlgorithmABC` | Base class for optimization algorithms. |
| `Population` | Decision, objective, and constraint population container. |
| `OptResult` | Standard result object returned by optimization runs. |
| `OptHistory` | Runtime history stored inside `OptResult`. |
| `OptReader` | Reader for sqlite results saved with `saveFlag=True`. |
| `Result` | Alias of the mutable optimization state. |
| `soea` | Single-objective algorithm subpackage. |
| `moea` | Multi-objective algorithm subpackage. |
| `expensive` | Expensive-model optimization subpackage. |

Algorithm groups:

| Group | Import path | Algorithms |
|---|---|---|
| Single-objective | `UQPyL.optimization.soea` | `GA`, `PSO`, `DE`, `SCE_UA`, `ML_SCE_UA`, `CSA`, `ABC` |
| Multi-objective | `UQPyL.optimization.moea` | `NSGAII`, `NSGAIII`, `MOEAD`, `RVEA` |
| Expensive optimization | `UQPyL.optimization.expensive` | `ASMO`, `EGO`, `MOASMO` |

## Optimization Workflow

All standard algorithms run through:

```python
result = algorithm.run(problem, seed=None)
```

Expensive optimization algorithms also accept optional initial data:

```python
result = algorithm.run(problem, xInit=None, yInit=None, seed=None)
```

Shared constructor controls:

| Parameter | Meaning |
|---|---|
| `maxFEs` | Maximum number of function evaluations. |
| `maxIters` | Maximum number of iterations. |
| `maxTolerates` | Maximum tolerated non-improving iterations. |
| `tolerate` | Improvement tolerance. |
| `verboseFlag` | Print runtime progress and summaries. |
| `verboseFreq` | Progress output frequency. |
| `logFlag` | Write text logs when enabled. |
| `saveFlag` | Persist sqlite snapshots and final result when enabled. |
| `saveFreq` | Snapshot save frequency. |

Example:

```python
from UQPyL.optimization.soea import GA
from UQPyL.problem import Sphere


problem = Sphere(nInput=5)
algorithm = GA(maxFEs=200, verboseFlag=False, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)
print(result.bestDecs)
print(result.bestObjs)
```

## `OptResult`

`OptResult` is returned by `run()`.

| Field | Type | Meaning |
|---|---|---|
| `bestDecs` | `np.ndarray` or `None` | Best decision variables. For multi-objective runs this is the current Pareto set. |
| `bestObjs` | `np.ndarray` or `None` | Best objective values. |
| `bestCons` | `np.ndarray` or `None` | Constraint values for the best solutions. |
| `bestMetric` | `float` or `None` | Best metric used for multi-objective progress, currently hypervolume. |
| `bestFeasible` | `bool` | Whether the best solution set is feasible. |
| `appearFEs` | `int` or `None` | Function evaluation count when the best result appeared. |
| `appearIters` | `int` or `None` | Iteration count when the best result appeared. |
| `FEs` | `int` | Final number of function evaluations. |
| `iters` | `int` | Final number of iterations. |
| `runtime` | `float` | Runtime in seconds. |
| `history` | `OptHistory` | Population and progress history. |
| `extra` | `dict` | Extra method-specific payload. |

Methods:

| API | Returns | Meaning |
|---|---|---|
| `summary()` | `dict` | Compact runtime summary. |
| `toDict()` | `dict` | Full result dictionary. |

## `OptHistory`

`OptHistory` stores optimization progress.

| Field | Meaning |
|---|---|
| `populations` | Population snapshots. |
| `bests` | Best-solution snapshots. |
| `metrics` | Metric values, such as hypervolume for multi-objective runs. |
| `iterToFEs` | Iteration to function-evaluation mapping. |
| `bestObjHistory` | Single-objective best-value history. |
| `numBestHistory` | Number of current best solutions for multi-objective runs. |
| `bestMetricHistory` | Multi-objective metric history. |
| `improvedHistory` | Whether each update improved the stored best result. |

Method:

| API | Returns | Meaning |
|---|---|---|
| `toDict()` | `dict` | Dictionary representation of the history. |

## Single-Objective Algorithms

Import:

```python
from UQPyL.optimization.soea import GA, PSO, DE, SCE_UA, ML_SCE_UA, CSA, ABC
```

### `GA`

```python
GA(
    nPop=50,
    proC=1,
    disC=20,
    proM=1,
    disM=20,
    maxFEs=50000,
    maxIters=1000,
    maxTolerates=None,
    tolerate=1e-6,
    verboseFlag=True,
    verboseFreq=10,
    logFlag=False,
    saveFlag=True,
    saveFreq=100,
)
```

| Parameter | Meaning |
|---|---|
| `nPop` | Population size. |
| `proC` | Crossover probability. |
| `disC` | Crossover distribution index. |
| `proM` | Mutation probability. |
| `disM` | Mutation distribution index. |

### `PSO`

```python
PSO(w=0.1, c1=0.5, c2=0.5, nPop=50, ...)
```

| Parameter | Meaning |
|---|---|
| `w` | Inertia weight. |
| `c1` | Cognitive coefficient. |
| `c2` | Social coefficient. |
| `nPop` | Population size. |

### `DE`

```python
DE(cr=0.9, f=0.5, nPop=50, ...)
```

| Parameter | Meaning |
|---|---|
| `cr` | Crossover probability. |
| `f` | Differential weight. |
| `nPop` | Population size. |

### `SCE_UA`

```python
SCE_UA(ngs=3, npg=7, nps=4, nspl=7, alpha=1.0, beta=0.5, ...)
```

| Parameter | Meaning |
|---|---|
| `ngs` | Number of complexes. |
| `npg` | Number of points in each complex. |
| `nps` | Number of points in each sub-complex. |
| `nspl` | Number of evolution steps before shuffling. |
| `alpha` | Reflection coefficient. |
| `beta` | Contraction coefficient. |

### `ML_SCE_UA`

```python
ML_SCE_UA(ngs=3, npg=7, nps=4, nspl=7, alpha=1.0, beta=0.5, sita=0.2, ...)
```

`ML_SCE_UA` uses the same parameters as `SCE_UA`, plus:

| Parameter | Meaning |
|---|---|
| `sita` | Multi-level SCE-UA control parameter. |

### `CSA`

```python
CSA(alpha=0.1, beta=0.15, M=3, nPop=25, ...)
```

| Parameter | Meaning |
|---|---|
| `alpha` | CSA control parameter. |
| `beta` | CSA control parameter. |
| `M` | CSA memory or search-control parameter. |
| `nPop` | Population size. |

### `ABC`

```python
ABC(employedRate=0.3, limit=50, nPop=50, ...)
```

| Parameter | Meaning |
|---|---|
| `employedRate` | Fraction of employed bees. |
| `limit` | Trial limit before abandoning a source. |
| `nPop` | Population size. |

## Multi-Objective Algorithms

Import:

```python
from UQPyL.optimization.moea import NSGAII, NSGAIII, MOEAD, RVEA
```

### `NSGAII`

```python
NSGAII(proC=1.0, disC=20.0, proM=1.0, disM=20.0, nPop=50, ...)
```

| Parameter | Meaning |
|---|---|
| `proC` | Crossover probability. |
| `disC` | Crossover distribution index. |
| `proM` | Mutation probability. |
| `disM` | Mutation distribution index. |
| `nPop` | Population size. |

### `NSGAIII`

```python
NSGAIII(proC=1.0, disC=20.0, proM=1.0, disM=20.0, nPop=50, ...)
```

`NSGAIII` uses the same public constructor parameters as `NSGAII`.

### `MOEAD`

```python
MOEAD(aggregation="TCH", nPop=50, ...)
```

| Parameter | Meaning |
|---|---|
| `aggregation` | Decomposition aggregation method. One of `"PBI"`, `"TCH"`, `"TCH_N"`, or `"TCH_M"`. |
| `nPop` | Population size. |

### `RVEA`

```python
RVEA(alpha=2.0, fr=0.1, nPop=50, ...)
```

| Parameter | Meaning |
|---|---|
| `alpha` | RVEA penalty parameter. |
| `fr` | Reference-vector adaptation frequency. |
| `nPop` | Population size. |

Example:

```python
from UQPyL.optimization.moea import NSGAII
from UQPyL.problem import ZDT1


problem = ZDT1(nInput=5)
algorithm = NSGAII(maxFEs=200, verboseFlag=False, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)
print(result.bestDecs.shape)
print(result.bestObjs.shape)
```

## Expensive Optimization Algorithms

Import:

```python
from UQPyL.optimization.expensive import ASMO, EGO, MOASMO
```

### `EGO`

```python
EGO(nInit=50, maxFEs=1000, maxIters=1000, maxTolerates=None, ...)
```

| Parameter | Meaning |
|---|---|
| `nInit` | Number of initial samples. |
| `xInit` | Optional initial decision matrix passed to `run()`. |
| `yInit` | Optional initial objective matrix passed to `run()`. |

### `ASMO`

```python
ASMO(
    nInit=50,
    surrogate=None,
    optimizer=None,
    euclidThres=1e-5,
    maxFEs=1000,
    maxIters=1000,
    maxTolerates=None,
    ...
)
```

| Parameter | Meaning |
|---|---|
| `nInit` | Number of initial samples. |
| `surrogate` | Optional surrogate model. |
| `optimizer` | Optional optimizer used on the surrogate problem. |
| `euclidThres` | Minimum Euclidean distance threshold for accepting new samples. |
| `oneStep` | Optional `run()` flag for one-step execution. |

### `MOASMO`

```python
MOASMO(
    surrogates=None,
    optimizer=None,
    pct=0.2,
    nInit=50,
    nPop=50,
    advance_infilling=False,
    maxFEs=1000,
    maxIters=100,
    ...
)
```

| Parameter | Meaning |
|---|---|
| `surrogates` | Optional multi-surrogate object. |
| `optimizer` | Optional optimizer used on surrogate problems. |
| `pct` | Infill selection fraction. |
| `nInit` | Number of initial samples. |
| `nPop` | Population size used by the internal optimizer. |
| `advance_infilling` | Whether to use advanced infill selection. |

## `Population`

`Population` stores decisions and optional evaluation outputs.

```python
Population(decs, objs=None, cons=None, conWgt=None)
```

| Field | Meaning |
|---|---|
| `decs` | Decision matrix. |
| `objs` | Objective matrix, or `None` before evaluation. |
| `cons` | Constraint matrix, or `None` when unavailable. |
| `conWgt` | Optional constraint weights. |
| `nPop` | Population size. |
| `D` | Number of decision variables. |
| `nOutput` | Number of objective columns when evaluated. |
| `frontNo` | Non-dominated front number for multi-objective populations. |
| `crowdDis` | Crowding distance for multi-objective populations. |

Common methods:

| Method | Returns | Meaning |
|---|---|---|
| `copy()` | `Population` | Deep copy. |
| `add(pop)` / `merge(pop)` | `Population` | Append another population in place. |
| `merged(pop)` | `Population` | Return merged copy. |
| `getBest(k=None)` | `Population` | Return best solution or Pareto front. |
| `getParetoFront()` | `Population` | Return current Pareto front. |
| `argsort()` | `np.ndarray` | Sort solution indices. |
| `clip(lb, ub)` | `Population` | Clip decisions in place. |
| `replace(index, pop)` | `Population` | Replace members in place. |
| `assignEval(objs, cons=None)` | `Population` | Attach objective and constraint values. |
| `size()` | `(int, int)` | Return `(nPop, D)`. |

## `OptReader`

Use `OptReader` to read sqlite snapshots saved with `saveFlag=True`.

```python
from UQPyL.optimization.runtime import OptReader


reader = OptReader("Result/ga_Sphere_20260509_1200_0000.sqlite3")
summary = reader.get_run_summary()
best = reader.load_last_best()
reader.close()
```

| Method | Returns | Meaning |
|---|---|---|
| `OptReader.list_runs(result_dir)` | table-like data | List saved optimization runs in a result directory. |
| `get_run()` | `dict` or `None` | Return raw run metadata. |
| `get_run_params()` | `dict` | Return stored algorithm parameters. |
| `get_run_summary()` | `dict` | Return compact run summary. |
| `load_algorithm()` | algorithm object | Reconstruct the saved algorithm class with stored parameters. |
| `load_problem()` | problem object | Load the saved problem payload. |
| `list_snapshots()` | `list[dict]` | List saved snapshots. |
| `load_population(snapshotId)` | `Population` | Load population for a snapshot. |
| `load_best(snapshotId)` | `Population` | Load best or Pareto population for a snapshot. |
| `load_last_population()` | `Population` | Load the latest population snapshot. |
| `load_last_best()` | `Population` | Load the latest best or Pareto snapshot. |
| `close()` | `None` | Close the sqlite connection. |
