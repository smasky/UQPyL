# Calibration

> 2.1.7 development interface: `obs` / `mask` are `(nObs,)`; `simFunc(X)` returns `(nSamples,nObs)`. No automatic flattening or legacy grid layout. Observations align by array position; observation labels are not required. CalResult/SQLite summaries use `nObs` / `n_obs` and `n_output=n_obs`; `nTime/nSeries/seriesLabels` are removed. Old calibration databases are incompatible; rerun with the new interface.

## Unified result interface

All successful methods return CalResult. Prefer these generic fields in shared code; new public fields use snake_case, while bestDecs/bestSim retain their existing names.

| Field | Meaning |
|---|---|
| `bestDecs`, `bestSim` | Best parameters `(1,nInput)` and full flattened simulation `(1,nObs)` |
| `best_score` | Raw best-member score under the configured metric |
| `best_index` | Best-member row in samples/simulations/scores |
| `samples` | Primary parameter collection `(nSamples,nInput)` |
| `simulations` | Row-aligned full flattened simulations `(nSamples,nObs)`, including masked columns |
| `scores` | Row-aligned raw scores `(nSamples,)`, evaluated with the mask |
| `sample_kind` | behavioral / sampling_ensemble / updated_ensemble |
| `weights` | Explicit row-aligned primary weights, or None (not zero weights) |
| `intervals` | Typed, provenance-labelled intervals; `[]` if not estimated |
| `uncertainty` | Optional independent prior-weighted result, or None |

GLUE exposes threshold-passing members and corresponding weights. SUFI2 exposes its final full search ensemble as sampling_ensemble. ES/IES expose updated_ensemble, without claiming exact Bayesian posterior samples.

Each interval has kind, space (parameter/simulation), lower, upper, probability, sample_source and indices. Simulation intervals cover only unmasked flattened observations; indices identify the relevant columns of simulations. Parameter indices identify parameter columns. Probability is the quantile level, not a validated coverage claim. Sources are samples for the primary collection and uncertainty.samples for the independent prior pool.

SUFI2 independent prior samples/weights live inside uncertainty; these weights are never attached to the primary search rows. Sampling envelopes and independent postprocessing intervals are separate list entries. Legacy posteriorDecs/behavioralDecs/eliteDecs and diagnostic/extra fields remain method-specific data. SUFI2 posteriorDecs means its final search ensemble; generic consumers should use samples/sample_kind.

summary() and CalReader.get_run_summary() also return best_score, best_index, sample_kind, n_samples, has_weights and interval_count. New arrays and nested intervals are independent copies of runtime state and legacy fields; building results does not simulate the model.

```python
result = method.run(problem, X, **runOptions)
print(result.bestDecs, result.best_score)
print(result.sample_kind, result.samples.shape)
print(result.scores[result.best_index])
for interval in result.intervals:
    print(interval["space"], interval["kind"], interval["sample_source"])
```

The `calibration` module estimates model parameters by comparing simulations with observations.

Use calibration when you have:

| Item | Meaning |
|---|---|
| Observed data | Measured time series, event values, or other reference outputs. |
| A simulation model | A function that maps parameter rows to simulated outputs. |
| Parameter bounds | Lower and upper limits for the parameters to calibrate. |
| A performance metric | For example `rmse`, `nse`, or `kge`. |

Calibration methods in UQPyL work with `ModelProblem`, not ordinary `Problem`.

In UQPyL, the standard calibration flow is:

```text
X -> simFunc(X) -> sim -> calibration metric / score -> parameter update or selection
```

At the modeling level, that becomes:

```text
obs + simFunc + parameter bounds -> ModelProblem -> calibration.run(...) -> CalResult
```

In other words, calibration is not just ordinary `Problem` plus a different algorithm. It is explicitly built on the simulation-backed `ModelProblem` path.

## Choose a Calibration Method

Start from how you want to use candidate parameter sets.

| Method | Use when | Main output |
|---|---|---|
| `GLUE` | You already have candidate parameters and want to keep behavioral samples under a threshold. | `behavioralDecs`, `behavioralSims` |
| `SUFI2` | You want elite samples and updated uncertainty bounds. | `eliteDecs`, `updatedLb`, `updatedUb`, `pfactor`, `rfactor` |
| `ES` | You want one ensemble-smoother update. | `posteriorDecs`, `posteriorSims` |
| `IES` | You want repeated ensemble-smoother updates. | `posteriorDecs`, `posteriorSims`, iterative history |

Practical default: use `GLUE` when you want a simple first calibration pass with existing samples. Use `SUFI2` when uncertainty bounds matter. Use `ES` or `IES` when you are working with ensemble smoothing.

## Calibration Workflow

The usual workflow is:

```text
obs + simFunc + parameter bounds -> ModelProblem -> calibration.run(...) -> CalResult
```

| Step | Action |
|---|---|
| Prepare observations | Store observations as a 1D array with shape `(n_obs,)`. |
| Define simulation | Write `simFunc(X)` for batched parameter rows. |
| Build `ModelProblem` | Provide `nInput`, `lb`, `ub`, `simFunc`, `obs`, and optional `mask`. |
| Choose method | Use `GLUE`, `SUFI2`, `ES`, or `IES`. |
| Read result | Inspect `bestDecs`, `bestSim`, posterior, elite, or behavioral samples. |

## Build a `ModelProblem`

Treat `ModelProblem` as the standard modeling container for calibration:

- `simFunc(X)` produces raw simulation output
- `obs` provides the observation reference
- `mask` controls which observation entries participate in scoring
- calibration methods then compute metrics, keep samples, or update parameters from that simulation context

For calibration, `ModelProblem` does not need to define `objFunc`. A simulation-only `ModelProblem` with `simFunc + obs` is already a valid calibration container.

`ModelProblem` connects parameter samples to simulation outputs.

In this toy model, the two parameters directly simulate two time steps:

```text
params [p1, p2] -> simulation [p1, p2]
obs = [1.0, 2.0]
```

```python
import numpy as np

from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)


obs = np.array([1.0, 2.0])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2))
    sim[:, 0] = X[:, 0]
    sim[:, 1] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, name="ToyModel")

sim = problem.simFunc([[1.0, 2.0]])

print(sim.shape)
print(problem.flattenObs())
print(problem.flattenMask())
```

Example output:

```text
(1, 2)
[1. 2.]
[False False]
```

Shape rules:

| Object | Required shape | Meaning |
|---|---|---|
| `X` | `(n_samples, n_input)` | Candidate parameter rows. |
| `obs` | `(n_obs,)` | Observed values. |
| `simFunc(X)` | `(n_samples, n_obs)` | Simulated values for every candidate row. |
| flattened simulation | `(n_samples, n_obs)` | Internal scoring layout. |

For non-computer-science users, read `X` as a table:

```text
one row = one parameter set
one column = one parameter
```

`simFunc` must return one simulation for each row of `X`.

## Run GLUE

`GLUE` scores every candidate parameter row and keeps behavioral samples.

For lower-is-better metrics such as `rmse`, a sample is behavioral when:

```text
score <= threshold
```

For higher-is-better metrics such as `nse`, a sample is behavioral when:

```text
score >= threshold
```

```python
import numpy as np

from UQPyL.calibration import GLUE
from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)


obs = np.array([1.0, 2.0])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2))
    sim[:, 0] = X[:, 0]
    sim[:, 1] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, name="ToyModel")

X = np.array([[1.0, 2.0], [1.0, 2.4], [0.0, 0.0]])

result = GLUE(metric="rmse", verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X, threshold=0.3)

print(result.bestDecs)
print(result.bestSim)
print(result.behavioralDecs)
print(result.diagnostics["scores"])
print(result.diagnostics["behavioralMask"])
```

Example output:

```text
[[1. 2.]]
[[1. 2.]]
[[1.  2. ]
 [1.  2.4]]
[0.     0.2828 1.5811]
[ True  True False]
```

Interpretation:

| Output | Meaning |
|---|---|
| `bestDecs` | Best parameter row. |
| `bestSim` | Simulation from the best parameter row. |
| `behavioralDecs` | Candidate rows that pass the threshold. |
| `scores` | Metric value for every candidate row. |
| `behavioralMask` | Boolean mask showing which rows passed. |

The first sample is perfect. The second sample has RMSE below `0.3`, so it is also behavioral. The third sample is rejected.

## Metric Direction

Calibration methods accept metric names or a custom callable.

| Metric | Better direction |
|---|---|
| `mse` | Lower is better |
| `mae` | Lower is better |
| `rmse` | Lower is better |
| `nse` | Higher is better |
| `r2` | Higher is better |
| `pbias` | Lower is better |
| `pearson_r` | Higher is better |
| `kge` | Higher is better |

For example, `nse` uses a higher-is-better threshold:

```python
import numpy as np

from UQPyL.calibration import GLUE
from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)


obs = np.array([1.0, 2.0])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2))
    sim[:, 0] = X[:, 0]
    sim[:, 1] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, name="ToyModel")

X = np.array([[1.0, 2.0], [1.0, 3.0], [0.0, 0.0]])
result = GLUE(metric="nse", verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X, threshold=0.0)

print(result.diagnostics["scores"])
print(result.diagnostics["behavioralMask"])
```

Example output:

```text
[ 1. -1. -9.]
[ True False False]
```

Only the first sample has `nse >= 0.0`.

## Use Masks

Use `mask` to ignore observation entries during scoring.

`mask` must have the same shape as `obs`.

```python
import numpy as np

from UQPyL.problem import ModelProblem


obs = np.array([1.0, 10.0, 2.0, 20.0])
mask = np.array([False, True, False, True])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 4))
    sim[:, 0] = X[:, 0]
    sim[:, 2] = X[:, 1]
    sim[:, [1, 3]] = 999.0
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, mask=mask, name="MaskedToyModel")

print(problem.obs.shape)
print(problem.mask.shape)
print(problem.flattenMask())
```

Example output:

```text
(4,)
(4,)
[False  True False  True]
```

Masked entries are ignored by calibration metrics. In this example, the second series is ignored even though the simulation writes `999.0` into it.

## Run SUFI2

`SUFI2` selects elite samples and updates uncertainty bounds from those elite samples.

```python
import numpy as np

from UQPyL.calibration import SUFI2
from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)


obs = np.array([1.0, 2.0])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2))
    sim[:, 0] = X[:, 0]
    sim[:, 1] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, name="ToyModel")

X = np.array([[1.0, 2.0], [1.0, 2.4], [0.0, 0.0]])

result = SUFI2(verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X, eliteSize=2)

print(result.bestDecs)
print(result.eliteDecs)
print(result.diagnostics["scores"])
print(result.diagnostics["updatedLb"])
print(result.diagnostics["updatedUb"])
print(result.diagnostics["pfactor"], result.diagnostics["rfactor"])
```

Example output:

```text
[[1. 2.]]
[[1.  2. ]
 [1.  2.4]]
[0.     0.2828 1.5811]
[1. 2.]
[1.  2.4]
0.5 0.38000000000000034
```

Read this as:

| Output | Meaning |
|---|---|
| `eliteDecs` | Best `eliteSize` parameter rows. |
| `updatedLb`, `updatedUb` | New parameter bounds inferred from elite samples. |
| `pfactor` | Fraction of observations bracketed by the prediction uncertainty band. |
| `rfactor` | Average width of the uncertainty band relative to observation variability. |

`SUFI2` can also generate samples internally. Set `nSamples` on the method and omit `X`:

```python
import numpy as np

from UQPyL.calibration import SUFI2
from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)


obs = np.array([1.0, 2.0])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2))
    sim[:, 0] = X[:, 0]
    sim[:, 1] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, name="ToyModel")

result = SUFI2(maxIters=3, nSamples=12, verboseFlag=False, logFlag=False, saveFlag=False).run(problem, eliteSize=4, seed=123)

print(result.bestDecs)
print(result.posteriorDecs.shape)
print(len(result.history.metricsHistory))
print(result.history.metricsHistory[-1].keys())
```

Example output:

```text
[[0.9825 1.8991]]
(12, 2)
3
dict_keys(['iter', 'pfactor', 'rfactor', 'updatedLb', 'updatedUb'])
```

## Run ES

`ES` performs one ensemble-smoother update.

```python
import numpy as np

from UQPyL.calibration import ES
from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)


obs = np.array([1.0, 2.0])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2))
    sim[:, 0] = X[:, 0]
    sim[:, 1] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, name="ToyModel")

X = np.array([[0.0, 0.0], [2.0, 3.0], [1.5, 0.5]])

result = ES(verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X)

print(result.posteriorDecs)
print(result.bestDecs)
print(result.posteriorSims.shape)
print(result.diagnostics["priorMean"])
print(result.diagnostics["posteriorMean"])
print(result.diagnostics["scores"])
```

Example output:

```text
[[1. 2.]
 [1. 2.]
 [1. 2.]]
[[1. 2.]]
(3, 2)
[1.1667 1.1667]
[1. 2.]
[0. 0. 0.]
```

In this linear toy model, the ensemble update moves all members exactly to the observation-matching parameters.

## Run IES

`IES` iterates a prior-anchored randomized maximum-likelihood update.

```python
import numpy as np

from UQPyL.calibration import ES, IES
from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)


obs = np.array([1.0, 2.0])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2))
    sim[:, 0] = X[:, 0]
    sim[:, 1] = X[:, 1] ** 2
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, name="NonlinearToyModel")

X = np.array([[0.0, 0.5], [2.0, 1.0], [1.5, 2.0]])

esResult = ES(verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X)
iesResult = IES(maxIters=4, lam=1e-6, seed=42, verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X)

print(iesResult.bestDecs)
print(iesResult.posteriorDecs.shape)
print(iesResult.posteriorSims.shape)
print(len(iesResult.history.metricsHistory))
print(np.mean(esResult.diagnostics["scores"]), np.mean(iesResult.diagnostics["scores"]))
```

Example output:

```text
[[1.     1.2353]]
(3, 2)
(3, 2)
4
0.33520286859016274 0.3352026900493963
```

Use `history.metricsHistory` to inspect iteration-level summaries.

## Use Verbose Output

Set `verboseFlag=True` to print a compact final summary.

```python
import numpy as np

from UQPyL.calibration import GLUE
from UQPyL.problem import ModelProblem


obs = np.array([1.0, 2.0])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2))
    sim[:, 0] = X[:, 0]
    sim[:, 1] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, name="ToyModel")
X = np.array([[1.0, 2.0], [1.0, 2.4], [0.0, 0.0]])

result = GLUE(metric="rmse", verboseFlag=True, logFlag=False, saveFlag=False).run(problem, X, threshold=0.3)
```

Example output:

```text
GLUE finished
  problem   : ToyModel
  metric    : rmse
  bestScore : 0
  bestX     : [1.0000e+00, 2.0000e+00]
  iters     : 0
  runtime   : 0.000s
```

## Read `CalResult`

Every calibration method returns a `CalResult`.

```python
import numpy as np

from UQPyL.calibration import GLUE
from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)


obs = np.array([1.0, 2.0])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2))
    sim[:, 0] = X[:, 0]
    sim[:, 1] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, name="ToyModel")
X = np.array([[1.0, 2.0], [1.0, 2.4], [0.0, 0.0]])

result = GLUE(metric="rmse", verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X, threshold=0.3)

print(result.method)
print(result.bestDecs)
print(result.bestSim)
print(result.diagnostics["scores"])
print(result.summary()["best_score"])
```

Example output:

```text
GLUE
[[1. 2.]]
[[1. 2.]]
[0.     0.2828 1.5811]
0.0
```

Important fields:

| Field | Meaning |
|---|---|
| `bestDecs` | Best parameter row under the configured metric. |
| `bestSim` | Simulation output for `bestDecs`, flattened to the observation vector layout. |
| `behavioralDecs`, `behavioralSims` | GLUE samples that pass the threshold. |
| `eliteDecs`, `eliteSims` | SUFI2 elite samples. |
| `posteriorDecs`, `posteriorSims` | ES or IES posterior ensemble. |
| `diagnostics` | Method-specific scores, masks, bounds, and summary values. |
| `history.metricsHistory` | Iteration summaries for iterative methods. |
| `summary()` | Compact dictionary for reporting. |

## Read a Saved SQLite Result

Set `saveFlag=True` to save a sqlite file under `Result/`.

```python
from pathlib import Path

import numpy as np

from docs_v2.assets.calibration_demo_model import sqliteSimFunc
from UQPyL.calibration import CalReader, GLUE
from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)


obs = np.array([1.0, 2.0])


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=sqliteSimFunc, obs=obs, name="ToyModel")
X = np.array([[1.0, 2.0], [1.0, 2.4], [0.0, 0.0]])

resultDir = Path("Result")
before = set(resultDir.glob("*.sqlite3")) if resultDir.exists() else set()

GLUE(metric="rmse", verboseFlag=False, logFlag=False, saveFlag=True).run(problem, X, threshold=0.3)

after = set(resultDir.glob("*.sqlite3"))
dbPath = sorted(after - before)[0]

with CalReader(dbPath) as reader:
    summary = reader.get_run_summary()
    params = reader.get_run_params()
    loaded = reader.load_result()

print(dbPath.as_posix())
print(summary["method"], summary["problem_name"])
print(summary["metric"], summary["best_score"])
print(params["metric"])
print(loaded.bestDecs)
print(loaded.behavioralDecs)
print(loaded.diagnostics["scores"])
```

Example output:

```text
Result/glue_ToyModel_YYYYMMDD_HHMM_xxxx.sqlite3
GLUE ToyModel
rmse 0.0
'rmse'
[[1. 2.]]
[[1.  2. ]
 [1.  2.4]]
[0.     0.2828 1.5811]
```

If you already know the sqlite path, use only the reader part:

```text
from UQPyL.calibration import CalReader

dbPath = "Result/glue_ToyModel_YYYYMMDD_HHMM_xxxx.sqlite3"

with CalReader(dbPath) as reader:
    summary = reader.get_run_summary()
    result = reader.load_result()
```

Saved calibration runs include a serialized `ModelProblem`. If you define `simFunc` interactively inside a notebook cell or temporary script, Python may not be able to pickle it. For persistent sqlite runs, prefer importable simulation functions or model classes.

Calibration persistence currently saves the final `CalResult` and related artifacts. It does not save per-iteration snapshots.

## Common Mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Using `Problem` instead of `ModelProblem` | Calibration methods reject the problem. | Build a `ModelProblem` with `simFunc` and `obs`. |
| Returning the wrong `simFunc` shape | Scoring fails or simulations do not align with observations. | Return `(n_samples, n_obs)`. |
| Passing a grid or column-vector observation array | `ModelProblem` requires `(nObs,)`. | Explicitly flatten in simulation column order, e.g. `obsGrid.reshape(-1)`. |
| Forgetting metric direction | GLUE may keep the wrong samples. | Use `<= threshold` for lower-is-better metrics and `>= threshold` for higher-is-better metrics. |
| Setting a GLUE threshold too strict | No behavioral samples are found. | Inspect `diagnostics["scores"]` and adjust the threshold. |
| Using a mask with the wrong shape | The model problem cannot validate it. | Make `mask.shape == obs.shape`. |
| Restoring a time/station grid | `bestSim` has shape `(1,nObs)`. | Retain the original grid dimensions and ordering yourself. |
| Saving an interactive `simFunc` | Pickling can fail. | Define persistent functions in importable modules. |

## Next Steps

| Goal | Read |
|---|---|
| Build simulation problems | [Problem](problem.md) |
| Generate candidate parameter sets | [Design of Experiment](doe.md) |
| Look up constructors and result fields | [Calibration API](api/calibration.md) |
| Compare with inference workflows | [Inference](inference.md) |
| See complete workflows | [Examples](examples.md) |

### ES / IES best-member metric direction

Final posterior members are ranked in the configured metric direction: RMSE/MSE/MAE are minimized, while NSE/KGE/R² are maximized. Internally, `normalizedScore()` negates higher-is-better metrics and takes absolute PBIAS for minimization; it does not rescale scores to 0–1. Diagnostics and reported/saved scores retain original metric values. Metric choice does not alter the ensemble update equations.


### PBIAS comparison semantics

The explicit `metric="pbias"` label minimizes `abs(PBIAS)` in GLUE, SUFI2, ES, and IES. GLUE requires a finite nonnegative percentage-point tolerance: a threshold of 5 accepts `-5% <= PBIAS <= 5%`, including endpoints.

The signed formula remains `100 * sum(sim - obs) / sum(obs)`, and diagnostics, reports, and saved scores retain that sign. Absolute value is applied after computing the full metric, not to each residual. Zero aggregate bias does not imply accurate individual observations. Callable metrics retain their default minimization semantics; the special rule is enabled by the string label.


### ES / IES covariance rank handling

Both methods solve the covariance system at full numerical rank and otherwise apply a symmetric eigendecomposition-based pseudoinverse. Eigenvalues at or below `n_valid_obs * eps * max(abs(eigenvalues))` are discarded. A zero-rank ensemble has zero gain and remains unchanged. The pseudoinverse cannot recover information absent from the ensemble.

R still defaults to zero and IES lam to zero; no noise or ridge is inserted automatically. R must have the correct shape and be finite, symmetric, and positive semidefinite. Roundoff-sized asymmetry is symmetrized and roundoff-sized negative eigenvalues are clipped to zero. IES lam must be a finite nonnegative scalar.

ES records solver/rank/dimension/cutoff in `diagnostics['covarianceSolve']`; IES records one entry per iteration in `diagnostics['covarianceSolves']`. General R uses observation-space eigendecomposition; default zero noise uses thin SVD when observations outnumber members.

### ES / IES update equations and uncertainty

ES uses a deterministic symmetric square-root update: Kalman mean plus transformed centered members. For a linear model without clipping, sample mean and covariance match Gaussian conditioning based on the initial sample moments. Nonlinear models remain ensemble-linearized approximations.

IES uses stochastic, prior-anchored Gauss–Newton EnRML. `seed=None` is optional; an integer reproduces the run. Perturbed observations are drawn once and remain fixed across iterations. `lam=0.0` selects the full GN step. Positive lam is dimensionless, fixed prior-metric damping, giving Hessian `(1+lam)*C_prior^-1 + H.T@R^-1@H` (implemented in gain form without requiring an invertible prior). **lam no longer adds an observation-space ridge `lam*I` to Cyy+R.** Fixed-step mode has no adaptive damping or acceptance/rejection; optional step backtracking is described below. Initial members and prior covariance remain fixed, preventing repeated assimilation of the same observations as independent data.

A finite stochastic IES ensemble need not exactly match population posterior moments; nonlinear posterior accuracy is not guaranteed. Zero/singular R uses a pseudoinverse extension: unresolved regression directions retain their previous slope, preventing collapse after hard observations from resetting the ensemble to its prior. Inconsistent hard observations need not be satisfied. Parameter regression uses initial per-column scales. `diagnostics["regressionRanks"]` records identifiable ranks; `updateMethod` identifies the update, with `damping` and `seed` also recorded for IES. Box clipping changes unconstrained moment identities.

`maxIters` is a fixed iteration budget, not a convergence guarantee. Scores do not enter the update; compare metrics with the same seed. ES still evaluates two batches; IES evaluates one initial batch plus one per iteration.

### Sampling guards, weighted intervals and optional backtracking

SUFI2 defaults to `explorationFraction=0.1` and `minRangeFraction=0.05`. The first internal batch covers the original domain. Subsequent batches reserve `ceil(nSamples*explorationFraction)` members for the original legal domain and sample the rest near the elite envelope. Local non-discrete widths are at least the specified fraction of original widths; originally fixed parameters stay fixed. Integer/discrete legality is retained, and exploration can reintroduce previously excluded choices. This reduces permanent exclusion risk but does not guarantee global optimality or implement full published SUFI-2. Set both fractions to zero for pure elite-envelope contraction.

`updatedLb/updatedUb` remain actual elite envelopes. History adds `samplingLb/samplingUb` (local bounds used in that iteration) and `explorationCount`. Supplied X is not resampled. `maxIters=0` warns and performs one screening iteration; negative/noninteger counts and invalid sample/elite sizes fail before simulation.

GLUE adds `run(..., logLikelihood=None, interval=0.95)`. The callback `logLikelihood(obs, sim, mask=mask)` receives flattened observations, **behavioral** simulations and flattened mask, and returns one log weight per behavioral member. The callback defines its noise model and respects the mask. Negative infinity means zero weight; NaN, positive infinity and zero total mass are rejected. Normalization subtracts the maximum log weight first. With no callback, weights are uniform and `weighting="uniform"`; scores are not automatically probabilities. Statistical interpretation depends on the supplied candidate distribution and prior/proposal weighting, which are not automatically corrected.

Diagnostics add `behavioralWeights`, `effectiveSampleSize=1/sum(w**2)`, `interval`, `ppuLower/ppuUpper`. Bounds are inverse weighted empirical-CDF quantiles of each unmasked simulation output (step quantiles without linear interpolation). They do not add future observation noise or guarantee nominal coverage. Best-member selection still uses the configured score.

IES adds optional `adaptive=True` (default False), `tolerance=1e-6`, and `maxBacktracks=8`. Actual trial simulations are checked against the fixed-prior, fixed-perturbation RML objective. Worsening full steps are halved, with at most 1+maxBacktracks trials. Exhausting trials warns, retains the last accepted ensemble, and stops. For zero/singular R, hard-data residual takes priority over the soft-data/prior objective; this is a degenerate extension. Clipped candidates outside the initial prior affine support are not assigned zero penalty. `lineSearch` records steps, acceptance and [hard residual, prior+soft residual] objectives. `stopReason` distinguishes `step_tolerance`, `line_search_stalled`, and `iteration_budget`; none certifies global convergence or posterior accuracy. Backtracking adds simulation batches; disabled mode preserves the fixed-step equations and call budget.

ES/IES add `boundHandling="rescale"`, with `"clip"` still the default. Rescaling shortens a member's entire proposed direction, using 99% of its maximum feasible step when it would leave the box. This reduces direct boundary pile-up; outward steps from an existing boundary can still stall. `boundEffects` reports the adjusted fraction, before/after means and ranges, and `unconstrained_moments_preserved`. Neither mode is exact truncated-Gaussian sampling; both can alter statistical moments.

### Accuracy updates: interval provenance and local derivatives

SUFI2 now computes `ppuLower/ppuUpper` and P/R factors from the **full current sampling ensemble**. Elite-output quantiles remain separately available as `elitePpuLower/elitePpuUpper`. `intervalKind="sampling_envelope"` identifies an output envelope under the current search/exploration distribution, not a parameter credible interval. Subsequent internal batches retain one incumbent best member before allocating local/global samples, within the same nSamples budget. History adds `retainedCount` and raw `bestScore`. With nSamples=1, retaining the incumbent leaves no exploration slot.

SUFI2 `run` adds `logLikelihood=None, uncertaintyX=None, uncertaintySamples=2048, interval=0.95` for **separate prior importance-weighting postprocessing**, without changing calibration selection. Without likelihood, uncertaintyStatus is not_estimated. Supplied uncertaintyX must represent prior samples not selected using the current data. If omitted, an independent RNG stream draws uncertaintySamples LHS members from the original legal domain, implying its uniform prior. The callback receives full flattened observations/simulations and mask, returning one log likelihood per prior member, as in GLUE. Nonuniform proposals require correct prior/proposal ratios in user-supplied log weights; contracted elites are not a prior pool.

`extra["uncertainty"]` contains `method="prior_importance_weighting"`, `prior_source`, `samples`, `weights`, `parameter_mean/variance/lower/upper`, `simulation_lower/upper`, `interval`, and `effective_sample_size`. Simulation intervals do not add future observation noise. ESS below 20 produces RuntimeWarning and low_effective_sample_size status; this heuristic threshold is not an accuracy guarantee. The independent sample pool adds one simulation batch. This explicitly named postprocessing does not claim to turn SUFI-2 into a full Bayesian algorithm.

IES adds **`localLinearization=True`** (default False). Memberwise finite differences replace the common regression slope, while retaining the original prior and fixed perturbed observations. Combine with adaptive=True to check actual RML descent. Each iteration adds at most twice the number of nonfixed parameters in simulation batches; bounded perturbations become one-sided near boundaries, and fixed parameters are skipped. Diagnostics record `linearization="member_finite_difference"` and `derivativeBatches`; covarianceSolves contains memberwise solves and regressionRanks entries are None in this mode.

This is a more expensive memberwise randomized MAP approximation for locally smooth simulators, not exact nonlinear posterior sampling. It does not guarantee multimodal mass accuracy or work reliably for nonsmooth models. The default ensemble-regression mode is unchanged. Unrepresentable finite-difference steps fail explicitly rather than becoming silent zero derivatives.

### Error metric range and low-ESS diagnostics

MSE, MAE and RMSE use per-row binary residual scaling to avoid intermediate subtraction, squaring or summation overflow when the final metric is representable. Masks, row shapes and physical units are unchanged. For example, MSE of `[1.4e154,0]` against zero is about `9.8e307`; MAE of `[1e308,1e308]` against `[-1e308,1e308]` is `1e308`. Truly unrepresentable final values still overflow to infinity or underflow to zero with a metric-specific RuntimeWarning.

GLUE records diagnostics["uncertaintyStatus"] as low_effective_sample_size below ESS 20, otherwise estimated (not an accuracy guarantee). When an explicit logLikelihood yields low ESS, it warns: `GLUE uncertainty effective sample size is below 20; weighted intervals may be unreliable.` Weights, quantiles and results remain unchanged; no uniform replacement or artificial interval widening occurs. Uniform screening without a likelihood also records small-ESS status but does not emit the likelihood-weighting warning. A roundoff tolerance prevents 20 equal-weight members from being spuriously flagged; SUFI2 uses the same threshold tolerance.
