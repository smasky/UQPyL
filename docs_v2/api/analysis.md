# Analysis API

DeltaTest recovers overflowing intermediate input/bound differences by halving both numerator and denominator before range normalization. Ordinary arithmetic is unchanged; fixed coordinates must still match their declared value, and out-of-bound samples are not clipped. Analysis, exhaustive subset selection and EA subset selection share this path.

After restoring output units, Morris checks `mu`, `mu_star` and `sigma` for floating-point range loss. A `RuntimeWarning` names the statistic, output row and input columns; `extra['morris_statistic_status']` stores `available`, `overflow` or `underflow` in output-by-input order. Truly overflowing values remain inf, while underflowed values remain zero; neither is an ordinary available statistic. Normalized effects are computed before unit restoration and remain available. SQLite results retain these statuses. This restoration check does not replace the existing finite-elementary-effect validation.

## Meaning of the reported indices

The name `S1` does not give every method the same variance-decomposition meaning.
Sobol and FAST estimate first-order (`S1`) and total-effect (`ST`) variance indices;
Sobol also reports pair interactions (`S2`). These decompositions assume independent
inputs and require the corresponding sampling design. RBDFAST estimates first-order
indices only; this implementation clips the bias-corrected estimates to `[0, 1]`.

Morris uses parameter-range-relative input steps (`ΔY / (ΔX / input_range)`),
retaining output units while accounting for input units and ranges.
RSA reports regional two-sample Cramér–von Mises distribution differences.
DeltaTest reports the change in half the mean squared difference to k non-self
neighbors after removing a variable; distances use parameter-range-scaled inputs.
Scores have squared output units and may be negative. Signed `S1_norm` divides by
the sum of absolute scores, preserving their signs and ranking. MARS reports positive increases in refitted GCV after a holdout predictive-quality check.
These screening scores are not Sobol variance fractions.

Normalized display weights are not substitutes for the original indices. In a
pure-interaction model, first-order effects can vanish while total effects remain
large; first-order ranking alone cannot establish that an input is irrelevant.

Analysis decodes X through Problem.unit_to_space when meta.output="unit", before evaluation and computation. Supplied Y must correspond to those real samples and is neither reevaluated nor rescaled by coordinate decoding. Result X uses real coordinates; metadata records output="real" and source_output="unit". Caller arrays and metadata are preserved. Omitted output means real; unknown output tags raise. Positional and keyword metadata follow the same rules. Morris normalizes scaled elementary effects so small nonzero output units are not mistaken for constants; mu, mu_star, and sigma retain their dimensional units.

DeltaTest leave-one-input-out sensitivity analysis requires at least two inputs and a positive integer neighbor count smaller than the sample count. Missing compiled MARS extensions leave the optional component unavailable; other import/initialization errors propagate. Reader `list_runs()` outputs use `run_id`, `created_at`, `finished_at`, `final_fes`/`final_iters` where applicable, `db_path`, and `file_name`; database column names and internal object fields retain their existing protocols.

RSA supports binary and discrete outputs: constant output values inside a quantile group do not invalidate the comparison of input distributions. Each group and its complement must contain at least two samples. Nonconstant outputs with no usable regions emit `RuntimeWarning` and return zero placeholders marked `insufficient_samples`; those zeros do not establish insensitivity. Constant outputs retain zero statistics without a warning.

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

Sobol, FAST, and RBDFAST internally center and scale finite outputs before computing
variance or spectral power. Changing output units, including a negative scale, preserves
the indices within floating-point accuracy. Scaling is independent for each output column
(each FAST trajectory block); original `Y` values remain in the result. Exactly constant
outputs retain the package convention of zero indices, and nonfinite outputs raise
`ValueError`. Variation already lost when input values were rounded cannot be recovered.

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
| `runId` | `str` or `None` | Unique run id, including runs without SQLite persistence. |
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

Sampling metadata:

| Key | Required value |
|---|---|
| `designType` | `"saltelli"` |
| `secondOrder` | Boolean `True` or `False`. Invalid or missing values trigger a warning and a recovery attempt. |
| `N` | Positive integer base sample size. Invalid or missing values trigger a warning and a recovery attempt. |

The block length is `problem.nInput + 2` for first-order designs and `2 * problem.nInput + 2` for second-order designs. Optional `blockSize` should be a positive integer equal to that length, and the sample matrix should contain `N * block_length` rows in the original Saltelli order. Inconsistent values, types, or row counts emit one `RuntimeWarning` per run instead of raising an error for the metadata discrepancy.

For inconsistent metadata, Sobol verifies the copied A/B hybrid coordinates for candidate layouts. A unique supported layout, or one selected by mutually consistent remaining fields, allows calculation with the recovered order and actual base sample count. Complete remaining or extra blocks are used with a warning; no rows are dropped, filled, or reordered. If no complete layout can be determined, the method returns zero metric placeholders marked `not_estimated` without evaluating the model. Supplied Y is retained after output selection; absent Y remains `None`. Placeholder S2 is included only when the declared secondOrder is boolean True. These zeros do not establish insensitivity.

`result.extra["sobol_design"]` records `status` (`validated`, `recovered`, or `not_estimated`), `n_samples`, `effective_n`, `effective_block_size`, `effective_second_order`, `metrics_available`, `recovery_basis`, and `issues`. The original metadata remains in `result.meta`; settings reflect the effective order, or `None` when unavailable. Diagnostics and absent outputs survive SQLite / `AnaReader` round trips.

Consistent metadata retains the original fast calculation path. These checks do not verify arbitrary supplied Y against X, and recovering a complete layout does not establish sampling quality or statistical accuracy. Missing metadata, an incorrect designType, invalid array shapes/nonfinite outputs, and the existing zero A/B variance condition retain their error behavior.

If all outputs are constant, the indices are zero. If hybrid outputs vary but the A/B base samples have zero variance, analysis raises `ValueError` with a request to increase the base sample size; the estimator cannot infer sensitivity from that base sample.

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
| `M` | Positive integer FAST interference parameter. |
| `N` | Positive integer number of rows per input block, with `N > 4*M**2`. |

Samples must contain exactly `N * problem.nInput` rows in the original block order. When supplied, `blockSize` must equal `N`. Incomplete or inconsistent blocks raise `ValueError`; samples are never silently truncated.

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

`RBDFAST` can analyze ordinary sample matrices and does not require metadata. `M` must be a positive integer and the sample count must exceed `2*M`. Inputs and outputs must be finite. Every varying input column must have distinct values; nonconstant columns with repeated values (including many discrete/integer samples) raise `ValueError` because their spectral ordering is not defined by this implementation. Constant sample columns return zero. This restriction prevents arbitrary row order from creating false sensitivity; it does not add an estimator for discrete inputs.

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

Morris uses `ΔY / (ΔX / input_range)`, the elementary effect based on input steps in the unit interval. Positive affine input-unit changes preserve effects when bounds are updated. Only the input step is dimensionless; `mu`, `mu_star` and `sigma` retain output units. The model and saved samples use real input coordinates. These screening effects are not Sobol indices.

Parameter ranges must be finite and positive. Continuous/integer axes use declared `ub-lb`; numeric discrete axes use the maximum minus minimum of their actual choices. Unordered categories are not assigned a numerical sensitivity interpretation. Unit-coordinate sample metadata is decoded before calculation. `result.extra["morris_effects"]` records `effect_mode="unit"`, `effect_units="output"` and `input_ranges`, including in saved results. This development-stage correction replaces the earlier physical-step calculation; there is no `effectMode` constructor parameter or physical-slope option.

At least two complete trajectories are required to estimate the sample standard deviation `sigma`; fewer trajectories raise `ValueError`. Calculations use signed floating arrays so unsigned/boolean differences retain direction; results retain the original input/output arrays and dtypes. Inputs and outputs must be finite.

## `RSA`

Regional sensitivity analysis.

```python
RSA(nRegion=20, verboseFlag=True, logFlag=False, saveFlag=False)
```

| Parameter | Meaning |
|---|---|
| `nRegion` | Integer number of output regions, at least 2; booleans are invalid. |

Metrics:

| Metric | Meaning |
|---|---|
| `S1` | Regional sensitivity statistic. |
| `S1_norm` | `S1` normalized by row sum. |

Inputs and outputs must be finite; NaN/inf raise `ValueError`. Output regions use linear quantiles with safe interpolation across opposite-sign extreme values, retaining the original ordering even when tiny and huge outputs coexist. Saved outputs keep their original values. Legitimate constant outputs retain zero statistics.

For each nonconstant output with no usable region comparisons, RSA emits one `RuntimeWarning` recommending more samples or fewer regions, then returns `S1=S1_norm=0` as placeholders. Each region and its complement need at least two samples. Empty regions alone do not trigger warnings when other regions are usable; RSA averages over the usable regions without changing `nRegion` automatically.

`result.extra["rsa_regions"]` contains `n_regions`, `n_samples`, and an `outputs` list aligned with the selected output rows. Each item records `output_label`, `status` (`estimated`, `constant_output`, or `insufficient_samples`), `valid_region_count`, and `region_sample_counts`. These diagnostics persist through `AnaReader`. The `estimated` status indicates usable comparisons, not a guarantee of statistical accuracy. Invalid `nRegion` values and empty input/output matrices raise `ValueError`.

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
| `findCombEA(problem, X, Y=None, FEs=10000, verboseFlag=True, saveFlag=True, *, seed=None)` | optimization result | Search for a variable subset with GA; optional local RNG seed. |
| `findCombVio(problem, X, Y=None)` | `list[str]` | Brute-force variable subset search. |

Metrics:

| Metric | Meaning |
|---|---|
| `S1` | Delta after removing a variable minus full-input Delta; signed squared-output units. |
| `S1_norm` | `S1 / sum(abs(S1))`; signed relative scores with absolute values summing to one when nonzero. |

All three public methods compute Euclidean distances after scaling continuous and integer coordinates by their declared parameter ranges: `(X - lb) / (ub - lb)`. Numeric discrete choices use the minimum/maximum of their actual `varSet` values; encoding bounds do not describe their physical value range. Fixed axes are mapped to zero, and sample-constant axes have zero leave-variable-out contribution. Bounds must be finite and ordered, and fixed inputs must match their declared value. Provided out-of-bound values are not clipped. Scaling is done once before selecting subsets, and saved result samples retain their original real coordinates.

At the k-th distance, tied neighbors share the remaining neighbor slots uniformly. Strictly closer neighbors keep their full weight, and the query row is excluded by identity. Distance differences within relative roundoff tolerance (`8 * eps * max(1, input_dimension)`, zero absolute tolerance) count as ties. All three entry points use this permutation-invariant rule. The ordinary path queries only k+2 neighbors; tied boundaries are handled one row at a time, and large identical-coordinate groups use aggregated output moments rather than materializing all pairwise distances.

Negative contributions are retained. Signed normalization handles negative totals and exact cancellation without reversing rankings; its row sum need not be one. An all-zero score row normalizes to zero. A nonconstant output with no positive scores emits `RuntimeWarning` and still returns results; constant outputs return zero without that warning. Unit changes must also update bounds or numeric discrete choices. Unordered categories are not assigned a new categorical distance model. This development-stage change can alter scores and subset selection compared with raw-coordinate distances and arbitrary tied-neighbor selection.

`analyze` centers/scales each output internally and normalizes the signed increments before restoring physical squared-output units. Raw `S1` values that underflow to zero emit `RuntimeWarning`; `S1_norm` remains based on the scaled increments. Consequently the normalized result may be nonzero while every stored raw score is zero. Unrepresentable raw overflow raises an explicit `ValueError` rather than returning infinities. `result.extra["delta_scaling"]["outputs"]` records each scale as `scale_mantissa * 2**scale_exponent`, plus `raw_underflow`; result `Y` retains its original values.

Both subset searches center each output column and apply one common amplitude scale before comparing objectives. This retains the original relative multi-output weighting (half the mean squared neighbor difference across rows, neighbors and outputs), without letting a large constant column erase a tiny varying output. Common positive or negative output-unit changes preserve subset comparisons within floating-point precision. Extremely different *varying* output amplitudes can still lose the smallest squared contribution; there is no implicit per-column standardization.

EA `bestObjs` and objective histories now use **scaled squared-output units**, including persisted results. `result.extra["delta_selection"]` records `objective_units="scaled_output_squared"`, `output_aggregation`, `scale_mantissa`, `scale_exponent`, `raw_underflow` and `raw_overflow`. The physical objective is the scaled objective times `(scale_mantissa * 2**scale_exponent)**2` when representable; avoid directly squaring an extreme scale in double precision. Physical objective underflow/overflow emits `RuntimeWarning` but search and returned scaled objectives remain valid. The empty subset is invalid and has an infinite penalty. `findCombVio` continues to return labels. These objectives are within-run comparison values; comparisons across datasets must account for the recorded scales. Use `seed` for reproducible EA searches; GA still does not guarantee an exhaustive optimum.

## `MARS`

MARS-based sensitivity analysis.

```python
MARS(verboseFlag=True, logFlag=False, saveFlag=False,
     maxDegree=2, maxTerms=40, minValidationR2=0.8,
     nValidationRepeats=1, stabilityTolerance=0.05, gcvImprovementTolerance=0.02)
```

`MARS` can be `None` when optional surrogate dependencies are unavailable.

At least 20 representative, exchangeable rows are required. A fixed local seed (0) reserves 20% of rows (at least five) for validation. Training-only scaling, paired hinge terms and up to second-order interactions are used by default. Basis size is capped to keep GCV effective complexity below the training sample count. All GCV scores use the same training subset; the holdout is not used for fitting. A nonconstant output whose full-model holdout R² falls below `minValidationR2` emits `RuntimeWarning` and continues to return importance. Constant outputs return zero scores.

`result.extra["mars_validation"]` records split sizes, effective term limit and per-output validation R² and full/reduced GCV. The threshold is a warning threshold, not a proof or confidence level; higher-order interactions, noise and unrepresentative samples may still require a different configuration or method. Duplicate, dependent or time-ordered rows need an appropriate sampling design; this random holdout does not establish out-of-group generalization. Negative GCV changes are clipped to zero rather than converted to positive importance. Adaptive basis selection can respond to rounding after output-unit changes; raw scores are approximate, not algebraically invariant. These scores are not Sobol variance fractions. This development-stage change replaces the previous all-row absolute-GCV scoring, so numerical scores can change.

Output centering/scaling is fitted on training rows using power-of-two preprocessing. Positive GCV increments are normalized before physical squared units are restored. Raw score underflow emits `RuntimeWarning` and preserves the scaled normalization; raw overflow raises an explicit `ValueError`. Per-output diagnostics additionally record `scaled_base_gcv`, `scaled_removed_gcv`, `scale_mantissa`, `scale_exponent` and `raw_underflow`, so raw-unit underflow does not erase the fitted GCV evidence. Exactly constant outputs are detected from the original values before averaging, including very small/large finite constants. The original result `Y` is preserved.

The default also checks whether removing one input substantially improves GCV. It warns when the improvement exceeds `gcvImprovementTolerance` times training output variance **and** at least half the full-model GCV. This heuristic can expose instability in greedy basis selection even with high holdout R²; it does not certify accuracy or correct the primary scores. Nonconstant-output diagnostics include `gcv_improvement_fraction`, `gcv_improvement_variable` (zero-based input index, or null when no improvement) and `gcv_search_unstable`.

Set `nValidationRepeats=3` to repeat training/validation with fixed local seeds 0, 1 and 2. The returned `S1` and `S1_norm` remain the primary seed-0 estimates; extra fits diagnose sensitivity rather than average it away. `result.extra["mars_stability"]` records each split's normalized weights and validation evidence, plus per-variable `normalized_min`, `normalized_max`, `normalized_mean`, `normalized_std` (ddof=0) and `max_normalized_range`. A range exceeding `stabilityTolerance` (default 0.05, five percentage points) emits `RuntimeWarning` and returns results. With one split, `assessed=False` and `stable=None`; with multiple splits, `stable` refers only to weight variation. These summaries are not confidence intervals, and stable weights can still be biased. Additional splits reuse the supplied/evaluated outputs and add surrogate fitting cost roughly in proportion to the repeat count, without additional problem evaluations. Default interaction order and one-split fitting cost are unchanged.

Metrics:

| Metric | Meaning |
|---|---|
| `S1` | Positive GCV increase after removing one variable: max(0, reduced GCV − full GCV). |
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


All built-in analysis methods normalize a one-dimensional external Y to a column before selecting outputs, validate row correspondence with X, and preserve the selected problem output labels. Use the public `UQPyL.analysis` imports or `UQPyL.analysis.methods`; the redundant outer forwarding modules have been removed. Runtime state is `method.state` and runtime parameters are `method.params`.


Runtime persistence uses a domain marker; readers reject another module's database and unmarked legacy databases. Every run has a UUID-based identifier shared by its database and log, even when SQLite saving is disabled. All readers support `with` and idempotent `close()`. Internal runtime objects use `state` and `params`; returned result objects retain their documented fields.
