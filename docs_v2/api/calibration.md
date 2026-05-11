# Calibration API

## `UQPyL.calibration`

The `calibration` module calibrates simulation models represented by `ModelProblem`.

### Import

```python
from UQPyL.calibration import GLUE, SUFI2, ES, IES
from UQPyL.calibration import CalReader, CalResult
```

### Public Objects

| Object | Role |
|---|---|
| `CalibrationABC` | Base class for calibration methods. |
| `GLUE` | Generalized likelihood uncertainty estimation. |
| `SUFI2` | Sequential uncertainty fitting version 2. |
| `ES` | Ensemble smoother. |
| `IES` | Iterative ensemble smoother. |
| `CalResult` | Standard result object returned by calibration runs. |
| `CalHistory` | Runtime history stored inside `CalResult`. |
| `CalReader` | Reader for sqlite results saved with `saveFlag=True`. |

## Calibration Workflow

Calibration methods only accept `ModelProblem`.

```python
result = method.run(problem, ...)
```

The `ModelProblem` must include:

| Requirement | Meaning |
|---|---|
| `simFunc` | Batched simulation function. |
| `obs` | 2D observation array with shape `(n_time, n_series)`. |
| `mask` | Optional boolean mask with the same shape as `obs`. |

`problem.simFunc(X)` must return simulation output whose first dimension is `n_samples`. Calibration methods flatten simulations and observations into a shared observation vector for scoring.

Shared constructor controls:

| Parameter | Meaning |
|---|---|
| `verboseFlag` | Print final summary when enabled. |
| `verboseFreq` | Runtime summary frequency for methods that record iterative history. |
| `saveFlag` | Persist final result and artifacts to sqlite. |
| `logFlag` | Write a text log when enabled. |
| `metric` | Calibration metric name or callable. |

Supported metric names:

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

Example:

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


problem = ModelProblem(
    nInput=2,
    ub=3.0,
    lb=0.0,
    simFunc=simFunc,
    obs=obs,
    simLabels=["Q"],
)

X = np.array([
    [1.0, 2.0],
    [1.0, 2.4],
    [0.0, 0.0],
])

result = GLUE(metric="rmse", verboseFlag=False).run(problem, X, threshold=0.3)
print(result.bestDecs)
print(result.behavioralDecs)
```

## `CalResult`

`CalResult` is returned by every calibration method.

| Field | Type | Meaning |
|---|---|---|
| `runId` | `str` or `None` | Saved run id when `saveFlag=True`. |
| `method` | `str` | Calibration method name. |
| `problemName` | `str` | Problem name. |
| `nInput` | `int` | Number of input variables. |
| `nTime` | `int` | Number of observation time steps. |
| `nSeries` | `int` | Number of observation series. |
| `nObs` | `int` | Total observation count. |
| `settings` | `dict` | Method settings. |
| `runtime` | `float` | Runtime in seconds. |
| `createdAt` | `str` | Creation timestamp. |
| `obs` | `np.ndarray` | Observation array. |
| `mask` | `np.ndarray` | Boolean mask array. |
| `simLabels` | `list[str]` | Simulation series labels. |
| `bestDecs` | `np.ndarray` or `None` | Best decision variables. |
| `bestSim` | `np.ndarray` or `None` | Simulation output for the best decision. |
| `posteriorDecs` | `np.ndarray` or `None` | Posterior decision ensemble. |
| `posteriorSims` | `np.ndarray` or `None` | Posterior simulation ensemble. |
| `behavioralDecs` | `np.ndarray` or `None` | Behavioral samples retained by GLUE. |
| `behavioralSims` | `np.ndarray` or `None` | Behavioral simulations retained by GLUE. |
| `eliteDecs` | `np.ndarray` or `None` | Elite samples retained by SUFI2. |
| `eliteSims` | `np.ndarray` or `None` | Elite simulations retained by SUFI2. |
| `diagnostics` | `dict` | Method diagnostics. |
| `history` | `CalHistory` | Iterative metric history. |
| `extra` | `dict` | Extra method-specific payload. |

Method:

| API | Returns | Meaning |
|---|---|---|
| `summary()` | `dict` | Compact runtime summary. |

## `CalHistory`

`CalHistory` stores method progress.

| Field | Meaning |
|---|---|
| `metricsHistory` | List of per-iteration metric summaries. |

## `GLUE`

Generalized likelihood uncertainty estimation.

```python
GLUE(
    verboseFlag=False,
    verboseFreq=1,
    saveFlag=False,
    logFlag=False,
    metric="rmse",
)
```

Run signature:

```python
result = method.run(problem, X, threshold)
```

| Parameter | Meaning |
|---|---|
| `X` | Candidate parameter matrix. |
| `threshold` | Behavioral threshold under the configured metric. |

For higher-is-better metrics such as `nse`, behavioral samples satisfy `score >= threshold`. For lower-is-better metrics such as `rmse`, behavioral samples satisfy `score <= threshold`.

Recorded outputs:

| Output | Meaning |
|---|---|
| `bestDecs`, `bestSim` | Best sample under the configured metric. |
| `behavioralDecs`, `behavioralSims` | Samples passing the threshold. |
| `diagnostics["scores"]` | Scores for all candidate samples. |
| `diagnostics["behavioralMask"]` | Boolean mask for retained samples. |

## `SUFI2`

Sequential uncertainty fitting version 2.

```python
SUFI2(
    verboseFlag=False,
    verboseFreq=1,
    saveFlag=False,
    logFlag=False,
    maxIters=1,
    nSamples=None,
    metric="rmse",
)
```

Run signature:

```python
result = method.run(problem, X=None, eliteSize=5, seed=None)
```

| Parameter | Meaning |
|---|---|
| `X` | Optional sample matrix. If omitted, `nSamples` must be set and samples are generated internally by LHS. |
| `eliteSize` | Number of elite samples retained each iteration. |
| `seed` | Optional seed used for internally generated samples. |
| `maxIters` | Number of SUFI2 iterations. |
| `nSamples` | Internal sample count when `X` is omitted. |

Recorded outputs:

| Output | Meaning |
|---|---|
| `bestDecs`, `bestSim` | Best sample from the final iteration. |
| `posteriorDecs`, `posteriorSims` | Final sampled population and simulations. |
| `eliteDecs`, `eliteSims` | Final elite samples and simulations. |
| `diagnostics["updatedLb"]` | Updated lower bounds from elite samples. |
| `diagnostics["updatedUb"]` | Updated upper bounds from elite samples. |
| `diagnostics["pfactor"]` | P-factor from the final elite set. |
| `diagnostics["rfactor"]` | R-factor from the final elite set. |

## `ES`

Single-pass ensemble smoother.

```python
ES(
    verboseFlag=False,
    verboseFreq=1,
    saveFlag=False,
    logFlag=False,
    metric="rmse",
)
```

Run signature:

```python
result = method.run(problem, X, r=None)
```

| Parameter | Meaning |
|---|---|
| `X` | Initial ensemble matrix with shape `(n_samples, n_input)`. |
| `r` | Optional observation error covariance matrix with shape `(n_valid_obs, n_valid_obs)`. If omitted, zeros are used. |

Recorded outputs:

| Output | Meaning |
|---|---|
| `posteriorDecs`, `posteriorSims` | Updated ensemble and simulations. |
| `bestDecs`, `bestSim` | Best posterior member under the configured metric. |
| `diagnostics["priorMean"]` | Prior ensemble mean. |
| `diagnostics["posteriorMean"]` | Posterior ensemble mean. |
| `diagnostics["scores"]` | Posterior scores. |

## `IES`

Iterative ensemble smoother.

```python
IES(
    verboseFlag=False,
    verboseFreq=1,
    saveFlag=False,
    logFlag=False,
    maxIters=5,
    lam=0.0,
    metric="rmse",
)
```

Run signature:

```python
result = method.run(problem, X, r=None)
```

| Parameter | Meaning |
|---|---|
| `X` | Initial ensemble matrix with shape `(n_samples, n_input)`. |
| `r` | Optional observation error covariance matrix with shape `(n_valid_obs, n_valid_obs)`. |
| `maxIters` | Number of smoother iterations. |
| `lam` | Diagonal regularization strength added to `C_yy + R`. |

Recorded outputs are the same shape as `ES`, with per-iteration summaries in `result.history.metricsHistory`.

## `CalReader`

Use `CalReader` to read sqlite results saved with `saveFlag=True`.

```python
from UQPyL.calibration import CalReader


with CalReader("Result/glue_ToyModel_20260509_1200_0000.sqlite3") as reader:
    result = reader.load_result()
    print(result.summary())
```

| Method | Returns | Meaning |
|---|---|---|
| `CalReader.list_runs(result_dir)` | table-like data | List saved calibration runs in a result directory. |
| `get_run()` | `dict` or `None` | Return raw run metadata. |
| `get_run_params()` | `dict` | Return stored method parameters. |
| `get_run_summary()` | `dict` | Return compact run summary. |
| `get_artifacts()` | `dict` | Load saved artifacts. |
| `load_problem()` | problem object | Load the saved `ModelProblem`. |
| `load_result()` | `CalResult` | Load the saved final calibration result. |
| `close()` | `None` | Close the sqlite connection. |
