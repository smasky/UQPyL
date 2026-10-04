# Inference API

A custom `logProbFunc` must return one real value per row, shaped `(n,)` or `(n,1)`; a scalar is also accepted for one state. `-inf` denotes zero probability. NaN, positive infinity, complex values and invalid shapes stop the run instead of producing a successful invalid chain. Initialization retains only feasible states with finite log probability, using at most `maxInitAttempts` LHS batches; insufficient support stops initialization. Zero-probability proposals are rejected without subtracting two negative infinities. The callback is also used for initialization validation and may be called repeatedly for the same state; it must return a deterministic log density.

With active dimensions, `DREAM_ZS(ps=1)` now mixes in a 10% full-dimensional symmetric Gaussian random walk to escape the affine subspace of a small archive. `snookerRefreshProb` accepts `(0,1]` and applies only when `ps=1`. Refresh standard deviations are 0.1 times each coordinate range; out-of-bounds proposals are rejected. A `RuntimeWarning` announces the policy, and `diagnostics['sampler']['proposal_settings']` records `full_support_refresh_probability`, `effective_snooker_probability` and `refresh_scale`. The default effective snooker probability is therefore 90%; ordinary `ps<1` runs do not use this refresh.

AMH computes unbiased history covariance from incremental centered moments, retaining rejected repeats, the existing scaling and covariance floor. Updates process only appended rows, with an extra mean vector and scatter matrix per chain; reset clears this cache. Full sample history is still retained. Rounding differences from full-history recomputation can change long trajectories, so bitwise equality is not guaranteed.

For MH/AMH/MH_Gibbs, the initial Gaussian standard deviation is `gamma * (ub-lb)`; the uniform proposal uses the same value as its half-width. MH/MH_Gibbs reflect at bounds; AMH rejects raw out-of-box proposals. Uniform variance is half-width squared divided by three, so the two distributions do not share an identical variance. AMH derives uniform half-widths from the square roots of covariance diagonals; Gaussian proposals retain the full covariance. Its adaptive covariance floor is `1e-3 * diag((ub-lb)^2)`, preserving input units and adding no floor on fixed coordinates. With insufficient history, an available covariance is retained as a copy.

Runtime summaries process only newly completed chain draws. Decoded samples use preallocated buffers; acceptance, feasibility, log-probability means, best samples and decision moments are updated incrementally. Best-sample ties retain chain-major, earliest-draw order. Intermediate SQLite snapshots read only each chain's latest draw; full arrays and independent history are copied when an explicit result is requested or the run finishes. Chain storage is append-only during a run. Full traces still require memory proportional to chains × draws × dimensions; incremental floating-point summaries can differ from full reductions by roundoff.

InfResult history, settings, diagnostics, and extra are isolated from mutable run state and other returned results. Reuse/reset cannot alter an earlier result. Public objs, bestObjs, and SQLite sample objectives use the original Problem objective direction, including maximization; logProb retains the probability used for sampling. Internal Chain/InfState objectives and inputs to custom logProbFunc remain minimization-oriented. By default logProb=-original_objective*problem.opt. This development-time export correction does not migrate existing databases.

DEMC defaults to nChains=3 and requires an integer of at least three at construction; booleans are rejected. Reader `list_runs()` outputs use `run_id`, `created_at`, `finished_at`, `final_fes`/`final_iters` where applicable, `db_path`, and `file_name`; database column names and internal object fields retain their existing protocols.

## `UQPyL.inference`

The `inference` module runs MCMC-style parameter inference on scalar-objective `Problem` instances.

### Import

```python
from UQPyL.inference import MH, AMH, MH_Gibbs, DEMC, DREAM_ZS
from UQPyL.inference import InfReader, InfResult
```

### Public Objects

| Object | Role |
|---|---|
| `MH` | Random-walk Metropolis-Hastings sampler. |
| `AMH` | Adaptive Metropolis-Hastings sampler. |
| `MH_Gibbs` | Metropolis-Hastings within Gibbs sampler. |
| `DEMC` | Differential evolution MCMC sampler. |
| `DREAM_ZS` | DREAM(ZS) sampler. |
| `InfResult` | Standard result object returned by inference runs. |
| `InfReader` | Reader for sqlite results saved with `saveFlag=True`. |

## Inference Workflow

All inference methods use:

```python
result = method.run(problem, gamma=0.1, seed=None)
```

`problem` must provide a scalar objective. Inference currently rejects problems where `problem.nOutput != 1`.

Shared constructor controls:

| Parameter | Meaning |
|---|---|
| `nChains` | Number of parallel chains. |
| `warmUp` | Number of warm-up iterations before formal sampling. |
| `maxIters` | Number of formal sampling draws, including the initial draw. |
| `verboseFlag` | Print compact runtime summaries. |
| `verboseFreq` | Iteration interval for terminal and log summaries. |
| `logFlag` | Write a text log when enabled. |
| `saveFlag` | Persist sqlite snapshots and final result when enabled. |
| `saveFreq` | Snapshot save frequency. |
| `logProbFunc` | Optional custom log-probability function. |
| `maxInitAttempts` | Maximum LHS batches used to find feasible initial chains. |

Default log-probability convention:

```text
log_prob = -oriented_objective
```

For minimization problems, this means lower objective values have higher default log probability. Provide `logProbFunc(y, decs=None, cons=None)` to override this convention.

Example:

```python
from UQPyL.inference import MH
from UQPyL.problem import Sphere


problem = Sphere(nInput=2)
method = MH(
    nChains=3,
    warmUp=5,
    maxIters=30,
    verboseFlag=False,
    logFlag=False,
    saveFlag=False,
)

result = method.run(problem, gamma=0.2, seed=123)
print(result.decs.shape)
print(result.acceptanceRate)
```

## `InfResult`

`InfResult` is returned by `run()`.

| Field | Type | Meaning |
|---|---|---|
| `runId` | `str` or `None` | Unique run id, including runs without SQLite persistence. |
| `method` | `str` | Inference method name. |
| `problemName` | `str` | Problem name. |
| `nInput` | `int` | Number of input variables. |
| `nOutput` | `int` | Number of outputs. |
| `nCon` | `int` | Number of constraints. |
| `settings` | `dict` | Method settings. |
| `runtime` | `float` | Runtime in seconds. |
| `createdAt` | `str` | Creation timestamp. |
| `decs` | `np.ndarray` | Decision draws with shape `(n_chains, draws, n_input)`. |
| `objs` | `np.ndarray` | Objective draws with shape `(n_chains, draws, n_output)`. |
| `cons` | `np.ndarray` or `None` | Constraint draws with shape `(n_chains, draws, n_con)`. |
| `logProb` | `np.ndarray` | Log-probability values with shape `(n_chains, draws)`. |
| `accepted` | `np.ndarray` | Boolean acceptance mask with shape `(n_chains, draws)`. |
| `feasibleMask` | `np.ndarray` | Boolean feasibility mask with shape `(n_chains, draws)`. |
| `acceptanceRate` | `np.ndarray` | Acceptance rate per chain. |
| `bestDecs` | `np.ndarray` or `None` | Best decision found. |
| `bestObjs` | `np.ndarray` or `None` | Best objective in original optimization direction. |
| `bestCons` | `np.ndarray` or `None` | Constraint values for `bestDecs`. |
| `bestFeasible` | `bool` | Whether the best sample is feasible. |
| `FEs` | `int` | Number of function evaluations. |
| `iters` | `int` | Final iteration count. |
| `history` | `InfHistory` | Runtime history. |
| `diagnostics` | `dict` | Method diagnostics. |
| `extra` | `dict` | Extra method-specific payload. |

Methods:

| API | Returns | Meaning |
|---|---|---|
| `summary()` | `dict` | Compact runtime summary. |
| `toDict()` | `dict` | Full result dictionary. |

## `InfHistory`

`InfHistory` stores runtime progress.

| Field | Meaning |
|---|---|
| `snapshots` | Runtime snapshot summaries. |
| `iterToFEs` | Iteration to function-evaluation mapping. |
| `meanLogProbHistory` | Mean log-probability history. |
| `acceptanceRateHistory` | Mean acceptance-rate history. |
| `feasibleRateHistory` | Feasible-rate history. |
| `bestObjHistory` | Best objective history. |

## `MH`

Random-walk Metropolis-Hastings sampler.

```python
MH(
    nChains=1,
    warmUp=1000,
    propDist="gauss",
    maxIters=10000,
    verboseFlag=True,
    verboseFreq=10,
    logFlag=False,
    saveFlag=True,
    saveFreq=100,
    logProbFunc=None,
    maxInitAttempts=1000,
)
```

| Parameter | Meaning |
|---|---|
| `propDist` | Proposal distribution. One of `"gauss"` or `"uniform"`. |
| `gamma` | Proposal scale passed to `run()`. Can be scalar, list, or array. |

## `AMH`

Adaptive Metropolis-Hastings sampler.

```python
AMH(
    nChains=1,
    warmUp=1000,
    maxIters=1000,
    propDist="gauss",
    verboseFlag=True,
    verboseFreq=10,
    logFlag=False,
    saveFlag=True,
    saveFreq=100,
    logProbFunc=None,
    maxInitAttempts=1000,
)
```

| Parameter | Meaning |
|---|---|
| `maxIters` | Number of formal sampling draws. |
| `propDist` | Proposal distribution. One of `"gauss"` or `"uniform"`. |
| `gamma` | Initial proposal scale passed to `run()`. |

## `MH_Gibbs`

Metropolis-Hastings within Gibbs sampler.

```python
MH_Gibbs(
    nChains=1,
    warmUp=1000,
    maxIters=1000,
    propDist="gauss",
    verboseFlag=True,
    verboseFreq=10,
    logFlag=False,
    saveFlag=True,
    saveFreq=100,
    logProbFunc=None,
    maxInitAttempts=1000,
)
```

| Parameter | Meaning |
|---|---|
| `propDist` | Proposal distribution. One of `"gauss"` or `"uniform"`. |
| `gamma` | Per-variable proposal scale passed to `run()`. |

## `DEMC`

Differential evolution MCMC sampler.

```python
DEMC(
    nChains=3,
    warmUp=1000,
    maxIters=1000,
    verboseFlag=True,
    verboseFreq=10,
    logFlag=False,
    saveFlag=True,
    saveFreq=100,
    logProbFunc=None,
    maxInitAttempts=1000,
)
```

| Parameter | Meaning |
|---|---|
| `gamma` | Optional differential-evolution scale passed to `run()`. If omitted, the method uses its internal default. |

## `DREAM_ZS`

DREAM(ZS) sampler with snooker updates and adaptive crossover weights.

```python
DREAM_ZS(
    nChains=10,
    warmUp=1000,
    ps=0.1,
    k=1,
    jitter=0.1,
    adpInterval=50,
    archSize=10,
    acTarget=0.25,
    nCR=5,
    maxIters=1000,
    verboseFlag=True,
    verboseFreq=10,
    logFlag=False,
    saveFlag=True,
    saveFreq=100,
    logProbFunc=None,
    maxInitAttempts=1000,
)
```

| Parameter | Meaning |
|---|---|
| `ps` | Probability of snooker update. |
| `k` | Number of differential evolution pairs. |
| `jitter` | Multiplicative proposal jitter scale. |
| `adpInterval` | Warm-up interval for crossover and scale adaptation. |
| `archSize` | Warm-up reservoir capacity multiplier relative to chain count; frozen for formal draws. |
| `acTarget` | Target acceptance rate for gamma scaling. |
| `nCR` | Number of crossover rate candidates. |
| `gamma` | DE full-space scale passed to `run()`; default `2.38 / sqrt(2 * max(1, n_active))`, adjusted for selected dimensions and pair count. Snooker independently uses scalar `U[1.2,2.2]`. |

The archive uses a bounded warm-up reservoir of occupation states, including
repeated rejected states. Formal sampling freezes this archive, crossover weights
and the adaptive scale; donors come only from the archive. With `warmUp=0`, the
initial archive and unadapted settings are used. Snooker operates in normalized
active coordinates with its Hastings correction evaluated in log space.

AMH/DEMC/DREAM reject raw out-of-box proposals without calling the user model;
these draws have `accepted=False`, and `FEs` counts actual evaluations. DEMC updates
chains sequentially conditional on earlier accepted moves. Its default scale uses
active dimensions and selects a unit-scale jump with probability 0.1; explicit
`gamma` does not use this scale mixture. Perturbations are independent centered
Gaussians with standard deviations `1e-6 * (ub-lb)`.

## `Chain`

`Chain` is the fixed-length storage container used internally for one inference chain.

```python
Chain(nInput, nOutput, nCons, length)
```

| Field | Meaning |
|---|---|
| `decs` | Decision draws. |
| `objs` | Objective draws. |
| `cons` | Constraint draws, or `None` for unconstrained problems. |
| `logProb` | Log-probability values. |
| `accepted` | Acceptance mask. |
| `count` | Number of stored draws. |

Method:

| Method | Meaning |
|---|---|
| `add(decs, objs, cons=None, logProb=None, accepted=True)` | Append one draw to the chain. |

## `InfReader`

Use `InfReader` to read sqlite results saved with `saveFlag=True`.

```python
from UQPyL.inference import InfReader


with InfReader("Result/mh_Sphere_20260509_1200_0000.sqlite3") as reader:
    result = reader.load_result()
    print(result.acceptanceRate)
```

| Method | Returns | Meaning |
|---|---|---|
| `InfReader.list_runs(result_dir)` | table-like data | List saved inference runs in a result directory. |
| `get_run()` | `dict` or `None` | Return raw run metadata. |
| `get_run_params()` | `dict` | Return stored method parameters. |
| `get_run_summary()` | `dict` | Return compact run summary. |
| `load_problem()` | problem object | Load the saved problem payload. |
| `list_snapshots()` | `list[dict]` | List saved snapshots. |
| `load_snapshot_members(snapshotId)` | `list[dict]` | Load chain members for a snapshot. |
| `load_last_snapshot_members()` | `list[dict]` | Load chain members from the latest snapshot. |
| `load_result()` | `InfResult` | Load the saved final result artifact. |
| `close()` | `None` | Close the sqlite connection. |


Runtime persistence uses a domain marker; readers reject another module's database and unmarked legacy databases. Every run has a UUID-based identifier shared by its database and log, even when SQLite saving is disabled. All readers support `with` and idempotent `close()`. Internal runtime objects use `state` and `params`; returned result objects retain their documented fields.

## Consistent configuration, partial results, and explicit diagnostics

All five samplers use `maxIters`, including AMH and DEMC; the previous `maxIterTimes` keyword is removed.
Results expose `stopReason` and summaries/readers expose `stop_reason` (`max_iters`, or generic `completed`). Exceptions retain failed/interrupted status.
Result attributes remain camelCase; fixed `toDict()` keys use snake_case, including `log_prob`, `feasible_mask`, `acceptance_rate`, `best_decs`, `best_objs`, and `best_cons`.

`InfReader.load_partial_result()` reads saved chain endpoints without requiring the final artifact, including failed runs.
It returns run metadata, snapshots, and last saved iteration/evaluation counts. `complete=False`, `resumable=False`, and `sample_scope="saved_chain_endpoints"` always apply. Empty runs have no snapshots and null last-saved positions.
Decisions use real coordinates and objectives use the original direction. Missing intermediate draws cannot be reconstructed, used as a complete chain, or resumed exactly. Normal `load_result()` still requires the final artifact.

`result.computeDiagnostics()` computes diagnostics explicitly from formal draws and updates only that result's diagnostics. There is no extra model evaluation, RNG consumption, automatic stopping, or database rewrite. Until requested, diagnostics are marked `not_computed`.

Each metric contains per-variable `values` and `status`:

| Field | Definition |
|---|---|
| `split_rhat` | Classical split R-hat, retained for comparison. |
| `rhat` | Maximum of rank-normalized and folded split R-hat. |
| `ess_bulk` | Effective sample size of rank-normalized split chains. |
| `ess_tail` | Minimum ESS of the 5% and 95% quantile indicators. |

Definitions follow [Stan's diagnostic overview](https://mc-stan.org/docs/reference-manual/analysis.html), with numerical comparisons against [ArviZ 0.22.0](https://github.com/arviz-devs/arviz/blob/v0.22.0/arviz/stats/diagnostics.py). ArviZ is not a runtime dependency. Rank normalization uses average ties and `(rank-3/8)/(S+1/4)`.

Four draws per chain are the computational minimum, not evidence of sufficient sampling. R-hat requires two chains; ESS supports one. Negative autocorrelation may yield ESS greater than the draw count. Odd-length splitting omits the middle draw, while tail quantile thresholds use all original formal draws before splitting.

Insufficient data, nonnumeric/nonfinite values, or any constant original chain produce null values and explicit statuses. Constant split/folded sequences invalidate their R-hat statistics; a completely constant indicator in either tail invalidates tail ESS. This conservative policy differs from reference-library values for some degenerate cases. Classical split R-hat still assumes finite marginal variance.

```python
report = result.computeDiagnostics()
rhat = report["rhat"]["values"]
essBulk = report["ess_bulk"]["values"]
essTail = report["ess_tail"]["values"]
# Check the corresponding status before interpreting each value.
```

Autocovariances use FFTs only on explicit request. No global convergence verdict is inferred; these parameter diagnostics do not prove convergence of all posterior features. Decimal rescaling can perturb folded rank ties through floating-point rounding, so bitwise scale invariance is not promised.

## Shared Parameters and Sampler Policies

All five methods validate common settings on every run, including settings changed
through `set()`: positive integer `nChains` (at least 3 for DEMC/DREAM), nonnegative
integer `warmUp`, and positive integer `maxIters`, `maxInitAttempts`, `verboseFreq`
and `saveFreq`. Booleans are not counts. `logProbFunc` must be callable or None.

Explicit `gamma` accepts a real scalar, an `nInput` list/array, or a matrix of shape
`(1,nInput)` or `(nChains,nInput)`. Values must be finite and nonnegative; zero is
allowed. MH_Gibbs also accepts lists and integer scalars. Invalid gamma is rejected
before model evaluation. DEMC/DREAM retain their automatic rules for `gamma=None`.
Invalid configuration raises ValueError rather than silently changing the sampler.

Every `InfResult.diagnostics["sampler"]` has the following common fields:

| Field | Meaning |
|---|---|
| `boundary_policy` | `reflect` or `reject`. |
| `update_mode` | Independent vector, coordinate-wise, sequential conditional, or independent given an archive. |
| `adaptation_phase` | `none`, `formal_sampling`, or `warmup_only`. |
| `proposal_family` | Random walk, differential evolution, or differential evolution with snooker. |
| `proposal_settings` | Includes `gamma` and `distribution`. Explicit/resolved gamma is expanded to chains × parameters. Automatic DEMC gamma and inapplicable distributions remain None. |

DREAM retains archive policy/size, frozen scale and crossover probabilities. DEMC
records unit-jump probability and noise scale in proposal_settings. These are policy
metadata, not convergence evidence. They persist in SQLite and are isolated in result
and exported copies.
