# Analysis API

## `UQPyL.analysis`

The `analysis` module evaluates how input variables affect objective or constraint outputs.

### Import

```python
from UQPyL.analysis import RBDFAST, Sobol, Morris
from UQPyL.analysis.runtime import AnaReader
```

### Public Objects

| Object | Role |
|---|---|
| `Sobol` | Variance-based global sensitivity analysis using Saltelli samples. |
| `FAST` | Fourier amplitude sensitivity test using FAST design samples. |
| `RBDFAST` | Random balance design FAST for first-order sensitivity. |
| `Morris` | Screening method based on elementary effects. |
| `RSA` | Regional sensitivity analysis. |
| `DeltaTest` | Nearest-neighbor delta test for variable sensitivity. |
| `MARS` | MARS-based sensitivity analysis. May be `None` if optional dependencies are unavailable. |

Runtime result objects are available from `UQPyL.analysis.runtime`.

| Object | Role |
|---|---|
| `AnaResult` | Standard result object returned by `analyze()`. |
| `AnaMetric` | One metric matrix in an `AnaResult`. |
| `AnaReader` | Reader for sqlite results saved with `saveFlag=True`. |

## Analysis Workflow

All analysis methods use the same public workflow.

```python
result = method.analyze(
    problem,
    X,
    Y=None,
    meta=None,
    target="objs",
    index="all",
)
```

| Parameter | Meaning |
|---|---|
| `problem` | A `ProblemBase` instance. |
| `X` | Input sample matrix. |
| `Y` | Optional output matrix corresponding to `X`. If omitted, the method evaluates `problem`. |
| `meta` | Optional sampling metadata from `sampleWithMeta()`. Required by some methods. |
| `target` | Output block to analyze. Usually `"objs"` or `"cons"`. |
| `index` | Output column selection. Use `"all"`, an integer, or a list of integers. |

Constructor runtime flags are shared by all analysis methods:

| Parameter | Meaning |
|---|---|
| `verboseFlag` | Print compact runtime summaries. |
| `logFlag` | Write a text log when enabled. |
| `saveFlag` | Persist result data to sqlite when enabled. |

Example:

```python
from UQPyL.analysis import RBDFAST
from UQPyL.doe import LHS
from UQPyL.problem import Sphere


problem = Sphere(nInput=3)
X = LHS("classic").sample(problem, nSamples=128, seed=123)
Y = problem.evaluate(X, target="objs").objs

method = RBDFAST(verboseFlag=False)
result = method.analyze(problem, X, Y=Y, target="objs")

print(result.metricNames)
print(result.getMetric("S1").values)
```

## `AnaResult`

`AnaResult` is returned by every analysis method.

| Field | Type | Meaning |
|---|---|---|
| `runId` | `str` or `None` | Saved run id when `saveFlag=True`. |
| `method` | `str` | Analysis method name. |
| `problemName` | `str` | Problem name. |
| `nInput` | `int` | Number of input variables. |
| `nOutput` | `int` | Number of objectives. |
| `nCon` | `int` | Number of constraints. |
| `target` | `str` | Analyzed target, such as `"objs"` or `"cons"`. |
| `settings` | `dict` | Method settings. |
| `meta` | `dict` | Sampling metadata recorded with the result. |
| `metrics` | `list[AnaMetric]` | Analysis metrics. |
| `X` | `np.ndarray` or `None` | Recorded input matrix. |
| `Y` | `np.ndarray` or `None` | Recorded output matrix. |
| `runtime` | `float` | Runtime in seconds. |
| `createdAt` | `str` | Creation timestamp. |
| `extra` | `dict` | Extra method-specific payload. |

### Methods and Properties

| API | Returns | Meaning |
|---|---|---|
| `metricNames` | `list[str]` | Names of recorded metrics. |
| `metricMap` | `dict[str, AnaMetric]` | Metric lookup by name. |
| `getMetric(name)` | `AnaMetric` | Return one metric by name. |
| `result[name]` | `AnaMetric` | Alias for `getMetric(name)`. |
| `summary()` | `dict` | Runtime summary. |
| `toDict()` | `dict` | Full serializable result dictionary. |

## `AnaMetric`

`AnaMetric` stores one analysis metric matrix.

| Field | Type | Meaning |
|---|---|---|
| `name` | `str` | Metric name, such as `"S1"`, `"ST"`, or `"mu_star"`. |
| `values` | `np.ndarray` | Metric values. Rows are analyzed outputs; columns are input variables or input pairs. |
| `rowLabels` | `list[str]` | Output labels, such as `obj1`. |
| `colLabels` | `list[str]` | Input labels or input-pair labels. |
| `colDim` | `str` | Column dimension type, such as `"decsDim1"` or `"decsDim2"`. |

Method:

| API | Returns | Meaning |
|---|---|---|
| `toDict()` | `dict` | Dictionary representation of the metric. |

## `Sobol`

Variance-based global sensitivity analysis.

```python
Sobol(verboseFlag=True, logFlag=False, saveFlag=False)
```

Use with `SaltelliDesign.sampleWithMeta()`.

```python
from UQPyL.analysis import Sobol
from UQPyL.doe import SaltelliDesign


X, meta = SaltelliDesign(secondOrder=True).sampleWithMeta(problem, N=512, seed=123)
Y = problem.evaluate(X, target="objs").objs

result = Sobol(verboseFlag=False).analyze(problem, X, Y=Y, meta=meta)
```

Required metadata:

| Key | Required value |
|---|---|
| `designType` | `"saltelli"` |
| `secondOrder` | `True` or `False` |

Metrics:

| Metric | Meaning |
|---|---|
| `S1` | First-order Sobol index. |
| `S1_norm` | First-order index normalized by row sum. |
| `ST` | Total-order Sobol index. |
| `ST_norm` | Total-order index normalized by row sum. |
| `S2` | Second-order Sobol index. Present only when `secondOrder=True`. |

## `FAST`

Fourier amplitude sensitivity test.

```python
FAST(verboseFlag=True, logFlag=False, saveFlag=False)
```

Use with `FASTDesign.sampleWithMeta()`.

```python
from UQPyL.analysis import FAST
from UQPyL.doe import FASTDesign


X, meta = FASTDesign(M=4).sampleWithMeta(problem, N=256, seed=123)
result = FAST(verboseFlag=False).analyze(problem, X, meta=meta)
```

Required metadata:

| Key | Required value |
|---|---|
| `designType` | `"fast"` |
| `M` | FAST interference parameter. |

Metrics:

| Metric | Meaning |
|---|---|
| `S1` | First-order FAST index. |
| `S1_norm` | First-order index normalized by row sum. |
| `ST` | Total-order FAST index. |
| `ST_norm` | Total-order index normalized by row sum. |

## `RBDFAST`

Random balance design FAST.

```python
RBDFAST(M=4, verboseFlag=True, logFlag=False, saveFlag=False)
```

| Parameter | Meaning |
|---|---|
| `M` | Number of harmonics used in the periodogram estimate. |

`RBDFAST` can analyze ordinary sample matrices. It does not require metadata.

```python
from UQPyL.analysis import RBDFAST
from UQPyL.doe import LHS


X = LHS("classic").sample(problem, nSamples=500, seed=123)
Y = problem.evaluate(X, target="objs").objs

result = RBDFAST(M=4, verboseFlag=False).analyze(problem, X, Y=Y)
```

Metrics:

| Metric | Meaning |
|---|---|
| `S1` | First-order RBD-FAST index. |

## `Morris`

Morris screening analysis based on elementary effects.

```python
Morris(verboseFlag=True, logFlag=False, saveFlag=False)
```

Use with `MorrisDesign.sampleWithMeta()`.

```python
from UQPyL.analysis import Morris
from UQPyL.doe import MorrisDesign


X, meta = MorrisDesign(numLevels=4).sampleWithMeta(problem, numTrajectory=20, seed=123)
Y = problem.evaluate(X, target="objs").objs

result = Morris(verboseFlag=False).analyze(problem, X, Y=Y, meta=meta)
```

Required metadata:

| Key | Required value |
|---|---|
| `designType` | `"morris"` |
| `numLevels` | Even integer greater than or equal to 4. |

Metrics:

| Metric | Meaning |
|---|---|
| `mu` | Mean elementary effect. |
| `mu_star` | Mean absolute elementary effect. |
| `sigma` | Standard deviation of elementary effects. |
| `S1_norm` | `mu_star` normalized by row sum. |

## `RSA`

Regional sensitivity analysis.

```python
RSA(nRegion=20, verboseFlag=True, logFlag=False, saveFlag=False)
```

| Parameter | Meaning |
|---|---|
| `nRegion` | Number of output regions used by RSA. |

Metrics:

| Metric | Meaning |
|---|---|
| `S1` | Regional sensitivity statistic. |
| `S1_norm` | `S1` normalized by row sum. |

## `DeltaTest`

Nearest-neighbor delta test for variable sensitivity.

```python
DeltaTest(nNeighbors=2, verboseFlag=True, logFlag=False, saveFlag=False)
```

| Parameter | Meaning |
|---|---|
| `nNeighbors` | Number of nearest neighbors used by the delta estimate. |

### Methods

| Method | Returns | Meaning |
|---|---|---|
| `analyze(problem, X, Y=None, meta=None, target="objs", index="all")` | `AnaResult` | Run the delta test. |
| `findCombEA(problem, X, Y=None, FEs=10000, verboseFlag=True, saveFlag=True)` | optimization result | Search for a variable subset with GA. |
| `findCombVio(problem, X, Y=None)` | `list[str]` | Brute-force variable subset search. |

Metrics:

| Metric | Meaning |
|---|---|
| `S1` | Delta-test variable contribution. |
| `S1_norm` | `S1` normalized by row sum. |

## `MARS`

MARS-based sensitivity analysis.

```python
MARS(verboseFlag=True, logFlag=False, saveFlag=False)
```

`MARS` can be `None` when optional surrogate dependencies are unavailable.

Metrics:

| Metric | Meaning |
|---|---|
| `S1` | Absolute change in MARS GCV when one variable is removed. |
| `S1_norm` | `S1` normalized by row sum. |

## `AnaReader`

Use `AnaReader` to read sqlite results saved by analysis methods with `saveFlag=True`.

```python
from UQPyL.analysis.runtime import AnaReader


with AnaReader("Result/rbdfast_Sphere_20260509_1200_0000.sqlite3") as reader:
    result = reader.load_result()
    print(result.metricNames)
```

| Method | Returns | Meaning |
|---|---|---|
| `AnaReader.list_runs(result_dir)` | table-like data | List saved analysis runs in a result directory. |
| `get_run()` | `dict` or `None` | Return raw run metadata. |
| `get_run_summary()` | `dict` | Return compact run summary. |
| `get_run_params()` | `dict` | Return raw stored run parameters. |
| `get_metrics()` | `list[AnaMetric]` | Load all metrics. |
| `get_metric(name)` | `AnaMetric` | Load one metric by name. |
| `get_artifacts()` | `dict` | Load saved artifacts such as `X`, `Y`, `settings`, `meta`, and `extra`. |
| `load_problem()` | problem object | Load the saved problem payload. |
| `load_result()` | `AnaResult` | Reconstruct the full analysis result. |
| `close()` | `None` | Close the sqlite connection. |

