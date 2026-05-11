# Calibration

The `calibration` module estimates model parameters by comparing simulations with observations.

Use calibration when you have:

| Item | Meaning |
|---|---|
| Observed data | Measured time series, event values, or other reference outputs. |
| A simulation model | A function that maps parameter rows to simulated outputs. |
| Parameter bounds | Lower and upper limits for the parameters to calibrate. |
| A performance metric | For example `rmse`, `nse`, or `kge`. |

Calibration methods in UQPyL work with `ModelProblem`, not ordinary `Problem`.

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
| Prepare observations | Store observations as a 2D array with shape `(n_time, n_series)`. |
| Define simulation | Write `simFunc(X)` for batched parameter rows. |
| Build `ModelProblem` | Provide `nInput`, `lb`, `ub`, `simFunc`, `obs`, and optional `mask`. |
| Choose method | Use `GLUE`, `SUFI2`, `ES`, or `IES`. |
| Read result | Inspect `bestDecs`, `bestSim`, posterior, elite, or behavioral samples. |

## Build a `ModelProblem`

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


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, simLabels=["Q"], name="ToyModel")

sim = problem.simFunc([[1.0, 2.0]])

print(sim.shape)
print(problem.flattenObs())
print(problem.flattenMask())
```

Example output:

```text
(1, 2, 1)
[1. 2.]
[False False]
```

Shape rules:

| Object | Required shape | Meaning |
|---|---|---|
| `X` | `(n_samples, n_input)` | Candidate parameter rows. |
| `obs` | `(n_time, n_series)` | Observed values. |
| `simFunc(X)` | `(n_samples, n_time, n_series)` | Simulated values for every candidate row. |
| flattened simulation | `(n_samples, n_time * n_series)` | Internal scoring layout. |

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


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, simLabels=["Q"], name="ToyModel")

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


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, simLabels=["Q"], name="ToyModel")

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


obs = np.array([[1.0, 10.0], [2.0, 20.0]])
mask = np.array([[False, True], [False, True]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 2))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    sim[:, :, 1] = 999.0
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, mask=mask, simLabels=["Q", "Ignored"], name="MaskedToyModel")

print(problem.obs.shape)
print(problem.mask.shape)
print(problem.flattenMask())
```

Example output:

```text
(2, 2)
(2, 2)
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


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, simLabels=["Q"], name="ToyModel")

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


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, simLabels=["Q"], name="ToyModel")

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


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, simLabels=["Q"], name="ToyModel")

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

`IES` repeats the ensemble-smoother update for multiple iterations.

```python
import numpy as np

from UQPyL.calibration import ES, IES
from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1] ** 2
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, simLabels=["Q"], name="NonlinearToyModel")

X = np.array([[0.0, 0.5], [2.0, 1.0], [1.5, 2.0]])

esResult = ES(verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X)
iesResult = IES(maxIters=4, lam=1e-6, verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X)

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


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, simLabels=["Q"], name="ToyModel")
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


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, simLabels=["Q"], name="ToyModel")
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


obs = np.array([[1.0], [2.0]])


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=sqliteSimFunc, obs=obs, simLabels=["Q"], name="ToyModel")
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
| Returning the wrong `simFunc` shape | Scoring fails or simulations do not align with observations. | Return `(n_samples, n_time, n_series)`. |
| Passing one-dimensional observations | `ModelProblem` rejects `obs`. | Use a 2D array, for example `obs.reshape(-1, 1)`. |
| Forgetting metric direction | GLUE may keep the wrong samples. | Use `<= threshold` for lower-is-better metrics and `>= threshold` for higher-is-better metrics. |
| Setting a GLUE threshold too strict | No behavioral samples are found. | Inspect `diagnostics["scores"]` and adjust the threshold. |
| Using a mask with the wrong shape | The model problem cannot validate it. | Make `mask.shape == obs.shape`. |
| Expecting `bestSim` to keep 3D shape | Result simulations are flattened for scoring. | Reshape using `obs.shape` when you need time-series layout. |
| Saving an interactive `simFunc` | Pickling can fail. | Define persistent functions in importable modules. |

## Next Steps

| Goal | Read |
|---|---|
| Build simulation problems | [Problem](problem.md) |
| Generate candidate parameter sets | [Design of Experiment](doe.md) |
| Look up constructors and result fields | [Calibration API](api/calibration.md) |
| Compare with inference workflows | [Inference](inference.md) |
| See complete workflows | [Examples](examples.md) |
