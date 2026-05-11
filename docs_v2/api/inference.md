# Inference API

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
| `maxIters` / `maxIterTimes` | Number of formal sampling draws, including the initial draw. |
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
| `runId` | `str` or `None` | Saved run id when `saveFlag=True`. |
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
    maxIterTimes=1000,
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
| `maxIterTimes` | Number of formal sampling draws. |
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
    nChains=1,
    warmUp=1000,
    maxIterTimes=1000,
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
| `adpInterval` | Interval for adaptive crossover updates. |
| `archSize` | Archive size multiplier relative to chain count. |
| `acTarget` | Target acceptance rate for gamma scaling. |
| `nCR` | Number of crossover rate candidates. |
| `gamma` | Optional scale passed to `run()`. If omitted, the method uses `2.38 / sqrt(2 * nInput)`. |

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
