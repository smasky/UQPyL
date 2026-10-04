# Calibration API

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

Calibration `rmse` uses per-row binary residual scaling, so a representable final RMSE no longer requires its intermediate squared errors to be representable. Masks, multiple simulation rows, and original inputs are preserved. Subtraction overflow for finite operands is handled at half scale before restoration. A final result outside the representable range that rounds to zero or infinity emits `RuntimeWarning: RMSE exceeds floating-point range; underflow is returned as zero and overflow as infinity.` MSE/MAE are unchanged by this fix.


SUFI2 internal LHS sampling preserves continuous, integer and discrete types. The first iteration uses the complete `varSet`; encoding lb/ub do not filter real discrete choices. Later iterations retain original choices within the numeric min/max envelope of elite real values, preserving their order, including legal intermediate choices absent from the elite samples. Integers are decoded within legal integer bounds; singleton domains remain valid. `diagnostics["updatedVarSet"]` and iteration history record the remaining discrete choices; `updatedLb/updatedUb` remain elite real-value bounds. The original problem is not mutated. Supplied X is validated in the real parameter domain and evaluated without remapping; illegal integer/discrete values are rejected before simulation.


`nse`, `r2`, `pbias`, `pearson_r`, `kge`, and `rfactor` preserve their scores under positive unit scaling. Small but valid values are not treated as zero merely because of their magnitude. Constant observations, constant simulations (correlation/KGE), zero observation sums (PBIAS), and zero observation means (KGE) retain their existing errors. Masks are applied before calculation and simulation rows receive separate scores. MSE/MAE/RMSE retain their dimensional definitions.

ES/IES validate observation covariance shape, finiteness, symmetry and positive semidefiniteness before the first simulation, using the number of unmasked observations. Unmasked observations must be nonempty and finite. IES also validates runtime `lam` and nonnegative integer `maxIters` before simulation; `maxIters=0` evaluates only the initial ensemble. Covariance validation is reused across iterations, without adding simulation calls.

ES/IES support continuous variables with box bounds only. Initial ensembles must be finite, inside the declared bounds, and contain at least two members. Integer/discrete variables and general constraints are rejected before simulation. Each update is projected onto the box before evaluating the simulator; fixed dimensions stay fixed. diagnostics["boundUpdates"] records adjusted_members and adjusted_values for each update. Projection changes otherwise out-of-bounds updates; interior updates keep the original formula and model evaluation counts are unchanged. Nonfinite updates raise instead of being hidden by clipping.

Reader `list_runs()` outputs use `run_id`, `created_at`, `finished_at`, `final_fes`/`final_iters` where applicable, `db_path`, and `file_name`; database column names and internal object fields retain their existing protocols.

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
| `obs` | 1D observation array with shape `(n_obs,)`. |
| `mask` | Optional boolean mask with the same shape as `obs`. |

`objFunc` is optional for calibration. Calibration methods are simulation-centric and can work with a simulation-only `ModelProblem`.

`problem.simFunc(X)` must return `(nSamples, nObs)`, with each column aligned to the same position in `obs` and `mask`. Calibration selects unmasked columns for scoring.

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


obs = np.array([1.0, 2.0])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2))
    sim[:, 0] = X[:, 0]
    sim[:, 1] = X[:, 1]
    return sim


problem = ModelProblem(
    nInput=2,
    ub=3.0,
    lb=0.0,
    simFunc=simFunc,
    obs=obs,
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

Each result is an independent snapshot. Reusing or resetting the method does not
change earlier results. `history`, `diagnostics`, `extra`, and `settings` are deeply
copied at the result boundary, including nested lists, dictionaries, and arrays.
Editing these result fields does not modify the method's runtime state or settings.
This copying does not perform additional model evaluations.

| Field | Type | Meaning |
|---|---|---|
| `runId` | `str` or `None` | Unique run id, including runs without SQLite persistence. |
| `method` | `str` | Calibration method name. |
| `problemName` | `str` | Problem name. |
| `nInput` | `int` | Number of input variables. |
| `nObs` | `int` | Total observation count. |
| `settings` | `dict` | Method settings. |
| `runtime` | `float` | Runtime in seconds. |
| `createdAt` | `str` | Creation timestamp. |
| `obs` | `np.ndarray` | Observation array. |
| `mask` | `np.ndarray` | Boolean mask array. |
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
result = method.run(problem, X=None, eliteSize=5, seed=None,
                    logLikelihood=None, uncertaintyX=None,
                    uncertaintySamples=2048, interval=0.95)
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
| `diagnostics["pfactor"]` | P-factor from the full final sampling ensemble. |
| `diagnostics["rfactor"]` | R-factor from the full final sampling ensemble. |

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
    seed=None,
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
| `lam` | Dimensionless fixed prior-metric GN damping; zero gives a full GN step. |
| `seed` | Optional seed for observation perturbations, fixed throughout each run. |

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


ES evaluates the initial and updated ensembles once each. IES retains full simulations between iterations, requiring one initial batch plus one batch per iteration; SUFI2 selects elite simulations from the already evaluated sample set. Masks select valid observations from those same simulations. SQLite stores one result artifact and a compact summary, without separate duplicate array artifacts. `load_result()` exposes all result fields; `get_run_summary()` reads only the small summary.


Runtime persistence uses a domain marker; readers reject another module's database and unmarked legacy databases. Every run has a UUID-based identifier shared by its database and log, even when SQLite saving is disabled. All readers support `with` and idempotent `close()`. Internal runtime objects use `state` and `params`; returned result objects retain their documented fields.

## Capabilities and covariance solving

`getCapabilities()` distinguishes ES/IES continuous box support from GLUE/SUFI2, whose general constraints are not used.
With default `r=None` (zero observation noise), ES/IES use a thin SVD of the scaled ensemble anomalies when valid observations outnumber ensemble members, without forming zero R or Cyy; diagnostics report `solver="svd"`. Smaller observation spaces retain the dense solver to avoid extra SVD overhead. Explicit R retains the general dense path (ES routes all-zero R through its zero-noise projection), reusing the rank-check eigendecomposition (`eigh` at full rank, `pinv` otherwise).
Fixed R is validated once per run; IES also computes its noise square root once. See the revised lam semantics below.
The default zero-noise gain keeps the observation-dimension cutoff. lam no longer raises observation nullspace rank. General dense-R optimization remains deferred.

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
