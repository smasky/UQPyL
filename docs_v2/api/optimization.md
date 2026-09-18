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

Optimization algorithms also accept optional initial populations:

```python
result = algorithm.run(problem, initialPop=None, seed=None)
```

`initialPop` can be either:

- a decision matrix with shape `(n, nInput)`
- a `Population` object

If the provided population is not evaluated, UQPyL evaluates it with the real `Problem`. If it has fewer members than the algorithm needs for initialization, UQPyL fills the remaining members automatically.

Shared constructor controls:

| Parameter | Meaning |
|---|---|
| `maxFEs` | Evaluation budget; batch-based methods may finish a batch beyond it. |
| `maxIters` | Maximum completed iterations, excluding initialization. Zero performs initialization only. |
| `maxTolerates` | Consecutive stagnant iterations before single-objective stopping; None disables this stop. |
| `tolerate` | Absolute objective improvement threshold; None disables stagnation stopping. |
| `verboseFlag` | Print runtime progress and summaries. |
| `verboseFreq` | Progress output frequency. |
| `logFlag` | Write text logs when enabled. |
| `saveFlag` | Persist sqlite snapshots and final result when enabled. |
| `saveFreq` | SQLite snapshot save frequency. |
| `historyFreq` | Full in-memory snapshot interval (default 10); None retains only the final snapshot. |

Initialization is iteration 0 and does not increase the stagnation count. Each completed
iteration compares the new historical best with the previous historical best. For feasible
solutions, improvement must strictly exceed `tolerate` to reset the count; smaller improvements
still update the best result. With constraints, any decrease in weighted violation, including
becoming feasible, resets the count even if the objective worsens. Thus `maxTolerates=2` stops
after two consecutive stagnant iterations; zero stops after initialization when this criterion
is enabled. Multi-objective methods do not use this single-objective stopping criterion.

Termination checks do not advance counters. Results, history, printed progress, and SQLite
snapshots use completed iteration numbers. Custom algorithms call `update(pop)` after
initialization and `update(pop, completed=True)` after each completed iteration. Initialization
evaluations still count toward `maxFEs`; existing per-algorithm batch rules are unchanged.

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
| `iterToFEs` | Mapping for every statistics update. |
| `snapshotIterToFEs` | Iteration/evaluation mapping for sparse populations and bests snapshots. |
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
| `initialPop` | Optional initial `Population` or decision matrix passed to `run()`. |

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


## Optimization coordinate convention

Built-in optimizers store and search populations in `[0,1]^d`, using separate `searchLb/searchUb` bounds. The original `problem.lb/ub` are preserved. Public `initialPop` decisions are real values and are encoded once; pre-evaluated objectives must use their original objective directions.

```text
DOE(output="unit") → unit population → search/repair
                                     → problem.unit_to_space(U) → real evaluation
```

Evaluation leaves the unit population unchanged. `OptResult`, exported history, logs, SQLite and NPZ contain real decisions and objectives in their original directions. Internal algorithm scores remain oriented toward minimization. Runtime history/best-decision snapshots are decoded and must not be passed directly to internal search operators.

EGO, ASMO and MOASMO use `problem.canonicalize_unit(U)` consistently for training, prediction and candidate deduplication. Continuous coordinates are preserved; integer/discrete coordinates use bin midpoints. Duplicate physical training solutions retain their first observation. Inner surrogate problems are continuous unit-cube problems with minimization-oriented objectives. Default surrogates add no input scaling; explicitly configured model scalers still apply consistently during training and prediction.

ASMO's `euclidThres` is a distance in unit coordinates. Surrogate optimizers may stop before exhausting the budget when no novel solution is found; small finite domains enumerate remaining representatives. This optimization convention does not prescribe the internal coordinates of inference algorithms.

### Constraint weights

`Problem(conWgt=[10, 1], nCon=2, ...)` assigns one finite nonnegative weight per constraint. The length must match `nCon`. `None` leaves violations unweighted; a zero weight ignores that constraint, including in feasibility checks.

```python
CV = np.sum(np.maximum(0, cons * conWgt), axis=1)
```

Optimizers copy the current Problem weights when accepting an initial population and after true evaluations, replacing any initial Population weights. Slicing, selection, merging and replacement preserve the configuration. Keep weights fixed during a run.

Stored `cons` remain raw constraint values. Printed/logged/stored violation summaries are weighted. Result extra, history snapshots, SQLite and NPZ retain `constraint_weights`; `OptReader` restores weights when loading a Population so subsequent selection uses the same rule.

### Multi-objective archive protocol

- `bestDecs/bestObjs/bestCons`: historical feasible nondominated archive; empty arrays before feasibility.
- `candidateDecs/candidateObjs/candidateCons`: up to 10 historical minimum-violation representatives before feasibility; None afterwards.
- `minViolation`: historical minimum weighted violation, zero after feasibility.
- `bestMetric`: feasible archive HV with a fixed reference and original scale; None before feasibility.
- `appearFEs/appearIters`: most recent archive change, or violation reduction before feasibility.
- `hvRefPoint`: optional constructor argument of NSGAII, NSGAIII, MOEAD, RVEA, and MOASMO, in original objective directions.
- `Population.getParetoFront()`: current feasible front; `getInfeasibleCandidates(k=10)`: separate infeasible diagnostics.
- `OptReader.load_candidates(snapshotId)` / `load_last_candidates()`: load separately stored candidates.

New `toDict()` and NPZ fields use `candidate_decs/candidate_objs/candidate_cons/min_violation`. See [optimization](../optimization.md#constrained-multi-objective-results) for archive and HV semantics.

### Batched High-dimensional HV

`HV(popObjs, refPoint=None, normalize=True, nSamples=1_000_000, rng=None, *, batchSize=4096)` uses Monte Carlo estimation for four or more objectives. The positive integer `batchSize` limits the number of sampled points generated at once. Each batch is compared against at most 256 solution points at a time, avoiding a full samples-by-solutions-by-objectives array. The final incomplete batch is included.

The default total sample count is unchanged. NumPy generators with identical initial states produce identical samples, estimates and post-call RNG states regardless of batch size. Smaller batches change temporary memory usage without reducing sampling accuracy. Fewer than four objectives still use the existing exact computation. batchSize is an HV function option, independent of the historyFreq retention policy below.

### In-memory History and SQLite Frequency

All built-in optimizers accept `historyFreq=10`, configurable before a run through `algorithm.set("historyFreq", value)`. It controls full in-memory population and best-solution/archive snapshots independently of SQLite's `saveFreq`.

| Configuration | Full in-memory snapshots |
|---|---|
| `historyFreq=10` (default) | First update, updates whose iteration number is a multiple of 10, and final update |
| `historyFreq=1` | Every update |
| `historyFreq=None` | Final update only |

A final update already captured is not appended twice. Lightweight statistics (iteration/evaluation counts, best values, HV, archive sizes and improvement flags) still record every update. Final best solutions and the full nondominated archive are preserved. This policy does not cap the live nondominated archive maintained by the optimizer.

`history.populations` and `history.bests` align with **`history.snapshotIterToFEs`**, containing each snapshot's `[iteration, FEs]`. Do not index sparse snapshots using positions in the full `iterToFEs` statistics sequence. Snapshot iterations identify actual state updates; final `result.iters` may additionally include a termination check. `toDict()` exports `snapshot_iter_to_fes`, and `result.extra["history_freq"]` records the policy. Decisions and objective directions retain their real problem representation.

For example, `GA(historyFreq=None, saveFlag=True, saveFreq=20)` retains all lightweight statistics and one final snapshot in memory, while SQLite stores full snapshots every 20 iterations and at completion. Periodic SQLite writes do not copy the accumulated in-memory history. NPZ retains its existing final-result and statistics-curve protocol; it does not automatically include full population history.

Sparse recording reduces memory growth rather than imposing a fixed limit. Use `None` to avoid accumulating full history snapshots in long runs.


Pre-evaluated `initialPop` must match the problem's objective/constraint dimensions and sample count, and must include constraints when `nCon > 0`. Incomplete evaluations raise an error before search; complete evaluations are reused. Completely unevaluated populations are evaluated normally.

`OptReader.load_result()` restores an `OptResult` with history and population data from the saved snapshots, including missing HV entries and actual feasibility; it does not reconstruct unsaved iterations. Plotting filters missing values together with their coordinates. `load_algorithm()` restores stored simple configuration, including budgets, stopping rules and output flags. Component instances (surrogate models and internal optimizers) require manual restoration and produce a warning; this API does not resume search state. Random seeds are available in run metadata for an explicit new run. Runtime counters and elapsed time are recorded independently of verbose output.


Runtime persistence uses a domain marker; readers reject another module's database and unmarked legacy databases. Every run has a UUID-based identifier shared by its database and log, even when SQLite saving is disabled. All readers support `with` and idempotent `close()`. Internal runtime objects use `state` and `params`; returned result objects retain their documented fields.
