# Inference

The `inference` module runs MCMC-style sampling for scalar `Problem` instances.

Use inference when you want to explore likely parameter values rather than find only one optimum. Typical examples include posterior sampling, uncertainty estimation, likelihood-based parameter fitting, or sampling around a scalar model score.

The core workflow is:

```text
define scalar Problem -> choose sampler -> method.run(problem, gamma=..., seed=...) -> InfResult
```

## What Inference Needs

Inference needs a `Problem` with exactly one output.

| Item | Meaning |
|---|---|
| `nInput` | Number of parameters to sample. |
| `lb`, `ub` | Lower and upper bounds of each parameter. They can be scalars shared by all variables or vectors with one value per variable. |
| `objFunc` | A batched function. It accepts an input matrix `X` and returns one scalar output per row. |
| `optType` | `"min"` or `"max"`. This tells UQPyL how to orient the objective before converting it to log probability. |
| `conFunc`, `nCon` | Optional constraints. Constraint values `<= 0` are feasible. |
| `logProbFunc` | Optional custom function when the objective is not already the score you want to sample from. |

Inference currently rejects multi-output problems. If you have several objectives, first combine them into one scalar score or use optimization instead.

## Basic Workflow

This example samples a two-parameter distribution implied by:

```text
f(x) = x1^2 + x2^2
```

Because this is a minimization problem, smaller objective values become more likely by default.

```python
import numpy as np

from UQPyL.inference import MH
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=-2.0, ub=2.0, objFunc=objFunc, optType="min", name="SpherePosterior")
method = MH(nChains=3, warmUp=5, maxIters=30, verboseFlag=False, logFlag=False, saveFlag=False)
result = method.run(problem, gamma=0.2, seed=123)

print(result.decs.shape)
print(result.objs.shape)
print(result.logProb.shape)
print(result.acceptanceRate)
print(result.bestDecs)
print(result.bestObjs)
print(round(result.summary()["acceptance_rate_mean"], 4))
```

Example output:

```text
(3, 30, 2)
(3, 30, 1)
(3, 30)
[0.5862 0.5517 0.5862]
[[ 0.2532 -0.222 ]]
[[0.1134]]
0.5747
```

Read this as:

| Output | Meaning |
|---|---|
| `(3, 30, 2)` | Three chains, thirty stored draws per chain, two parameters per draw. |
| `(3, 30, 1)` | The same chain/draw table, with one objective value per draw. |
| `acceptanceRate` | One acceptance rate per chain. |
| `bestDecs` | The sampled parameter vector with the best scalar score. |
| `bestObjs` | The corresponding objective value in the original problem direction. |

For this toy problem, good samples are near `[0, 0]`.

## Understand Batched Functions

`objFunc` receives a matrix, not one parameter vector at a time.

```text
X.shape = (n_samples, n_input)
X[0, :] = first parameter vector
X[1, :] = second parameter vector
```

The function should return:

```text
Y.shape = (n_samples, 1)
Y[0, 0] = output for X[0, :]
Y[1, 0] = output for X[1, :]
```

That is why examples use:

```text
X = np.atleast_2d(X)
return np.sum(X**2, axis=1, keepdims=True)
```

`np.atleast_2d(X)` makes the function work even if UQPyL passes only one row. `axis=1` means "sum across variables for each row". `keepdims=True` keeps the result as a two-dimensional column, which is the expected output shape.

## Objective and Log Probability

By default, inference converts the scalar objective to log probability with:

```text
log_prob = -oriented_objective
```

The "oriented objective" means UQPyL first applies `optType`.

| Problem direction | Effect |
|---|---|
| `optType="min"` | Smaller objective values become larger log probabilities. |
| `optType="max"` | Larger original objective values become larger log probabilities after orientation. |

This default is useful when your scalar objective is already a loss, error, negative score, or energy-like value. If your model has a real likelihood or posterior formula, pass `logProbFunc`.

## Use a Custom Log Probability

Use `logProbFunc(y, decs=None, cons=None)` when the objective output is not the actual log probability.

In this example, the sampler targets a Gaussian-shaped distribution centered near `[0.5, -0.25]`. The objective is still evaluated and stored, but the custom log probability controls acceptance.

```python
import numpy as np

from UQPyL.inference import MH
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


def logProbFunc(y, decs=None, cons=None):
    decs = np.atleast_2d(decs)
    target = np.array([0.5, -0.25])
    return -0.5 * np.sum((decs - target) ** 2, axis=1)


problem = Problem(nInput=2, nObj=1, lb=-2.0, ub=2.0, objFunc=objFunc, optType="min", name="SpherePosterior")
method = MH(nChains=3, warmUp=5, maxIters=30, logProbFunc=logProbFunc, verboseFlag=False, logFlag=False, saveFlag=False)
result = method.run(problem, gamma=0.2, seed=123)

print(result.logProb.shape)
print(result.bestDecs)
print(result.logProb[:, -3:])
```

Example output:

```text
(3, 30)
[[-0.0593  0.2672]]
[[-0.6783 -0.6783 -1.0424]
 [-0.3285 -0.0082 -0.1918]
 [-0.179  -0.0572 -0.7936]]
```

The custom function must return one log-probability value per evaluated row. Higher log probability means the proposal is more likely to be accepted.

## Choose an Inference Method

Start simple, then move to stronger samplers if the chains mix poorly.

| Method | Good first use |
|---|---|
| `MH` | Basic random-walk Metropolis-Hastings. Start here for simple scalar problems. |
| `AMH` | Adaptive Metropolis-Hastings. Useful when a fixed proposal scale is hard to tune. |
| `MH_Gibbs` | Coordinate-wise updates. Useful when changing one variable at a time is more stable. |
| `DEMC` | Differential-evolution MCMC. Useful when multiple chains can share information. |
| `DREAM_ZS` | DREAM(ZS)-style sampler with archive and adaptive crossover. Useful for harder posterior shapes. |

For constructor parameters, see [Inference API](api/inference.md).

A compact comparison run looks like this:

```python
import numpy as np

from UQPyL.inference import AMH, DEMC, DREAM_ZS, MH, MH_Gibbs
from UQPyL.problem import Problem

np.set_printoptions(precision=3, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=-2.0, ub=2.0, objFunc=objFunc, optType="min", name="SpherePosterior")
methods = [
    MH(nChains=3, warmUp=3, maxIters=12, verboseFlag=False, logFlag=False, saveFlag=False),
    AMH(nChains=3, warmUp=3, maxIterTimes=12, verboseFlag=False, logFlag=False, saveFlag=False),
    MH_Gibbs(nChains=3, warmUp=3, maxIters=12, verboseFlag=False, logFlag=False, saveFlag=False),
    DEMC(nChains=4, warmUp=3, maxIterTimes=12, verboseFlag=False, logFlag=False, saveFlag=False),
    DREAM_ZS(nChains=4, warmUp=3, maxIters=12, archSize=3, verboseFlag=False, logFlag=False, saveFlag=False),
]

for method in methods:
    result = method.run(problem, gamma=0.2, seed=123)
    print(result.method, result.decs.shape, result.acceptanceRate)
```

Example output:

```text
MH (3, 12, 2) [0.909 0.545 0.727]
AMH (3, 12, 2) [0.455 0.636 0.818]
MH-Gibbs (3, 12, 2) [0.545 0.909 0.545]
DEMC (4, 12, 2) [0.727 0.818 0.727 0.727]
DREAM-ZS (4, 12, 2) [0.818 1.    1.    1.   ]
```

Do not choose a method only because one short run has a higher acceptance rate. Acceptance rate is a diagnostic, not the final quality measure. Also inspect whether chains explore the parameter space and whether summaries are stable under a larger sampling budget.

## Set Proposal Scale

`gamma` controls proposal size.

| `gamma` form | Meaning |
|---|---|
| Scalar, such as `0.2` | Use the same proposal scale for every chain and variable. |
| Vector, such as `[0.1, 0.2]` | Use one proposal scale per input variable. |
| Matrix with shape `(nChains, nInput)` | Use a different per-variable scale for each chain. |

Example with a vector scale:

```python
import numpy as np

from UQPyL.inference import MH
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=-2.0, ub=2.0, objFunc=objFunc, optType="min", name="SpherePosterior")
method = MH(nChains=2, warmUp=2, maxIters=10, verboseFlag=False, logFlag=False, saveFlag=False)
result = method.run(problem, gamma=[0.1, 0.2], seed=123)

print(result.decs.shape)
print(result.acceptanceRate)
```

Example output:

```text
(2, 10, 2)
[0.8889 0.5556]
```

If acceptance is very low, `gamma` is often too large. If acceptance is very high but chains barely move, `gamma` may be too small. This is a tuning signal, not a guarantee; always look at the chain values and domain plausibility.

## Work With Constraints

Constraints use the same convention as optimization:

```text
cons <= 0 means feasible
```

During inference, infeasible proposals are hard rejected. The chain stores feasibility in `feasibleMask`.

```python
import numpy as np

from UQPyL.inference import MH
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


def conFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] + X[:, 1] - 0.5).reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, nCon=1, lb=-2.0, ub=2.0, objFunc=objFunc, conFunc=conFunc, optType="min", name="ConstrainedInference")
method = MH(nChains=2, warmUp=2, maxIters=10, verboseFlag=False, logFlag=False, saveFlag=False)
result = method.run(problem, gamma=0.1, seed=123)

print(result.feasibleMask.shape)
print(result.acceptanceRate)
print(result.bestFeasible)
print(result.bestCons)
```

Example output:

```text
(2, 10)
[0.7778 0.7778]
True
[[-1.2823]]
```

Here `bestCons` is negative, so the best sampled point is feasible.

## Use Verbose Output

Set `verboseFlag=True` to see progress during a run. Use `verboseFreq` to control how often progress is printed.

```python
import numpy as np

from UQPyL.inference import MH
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=-2.0, ub=2.0, objFunc=objFunc, optType="min", name="SpherePosterior")
method = MH(nChains=2, warmUp=2, maxIters=8, verboseFlag=True, verboseFreq=2, logFlag=False, saveFlag=False)
result = method.run(problem, gamma=0.2, seed=123)
```

Example verbose output:

```text
Inference: MH
Problem: SpherePosterior
nInput: 2
nOutput: 1
maxIters: 8
MH | iter=0 eval=6 curLogp=-2.0645e+00 accept=1.0000e+00 feasible=1.0000e+00 best=1.2846e-01 time=0.0s
MH | iter=2 eval=10 curLogp=-1.3276e+00 accept=7.5000e-01 feasible=1.0000e+00 best=1.2846e-01 time=0.0s
MH | iter=4 eval=14 curLogp=-1.1271e-01 accept=6.2500e-01 feasible=1.0000e+00 best=5.1974e-02 time=0.0s
MH | iter=6 eval=18 curLogp=-1.1043e+00 accept=5.8333e-01 feasible=1.0000e+00 best=5.1974e-02 time=0.0s
Inference finished
  method          : MH
  status          : finished
  iterations      : 7
  evaluations     : 20
  chains          : 2
  draws           : 8
  mean logProb    : -1.0207e+00
  acceptance mean : 6.4286e-01
  acceptance min  : 5.7143e-01
  acceptance max  : 7.1429e-01
  feasible rate   : 1.0000e+00
  best value      : 5.1974e-02
  bestX           : [3.8025e-03, 2.2795e-01]
  meanX           : [6.8361e-02, -4.4492e-01]
  stdX            : [5.4821e-01, 7.1938e-01]
  elapsed         : 0.0s
```

The most useful fields during early tuning are `accept`, `feasible`, `best`, `meanX`, and `stdX`.

## Read `InfResult`

`method.run()` returns an `InfResult`.

```python
import numpy as np

from UQPyL.inference import MH
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=-2.0, ub=2.0, objFunc=objFunc, optType="min", name="SpherePosterior")
method = MH(nChains=3, warmUp=5, maxIters=30, verboseFlag=False, logFlag=False, saveFlag=False)
result = method.run(problem, gamma=0.2, seed=123)

print(result.decs.shape)
print(result.objs.shape)
print(result.logProb.shape)
print(result.accepted.shape)
print(result.feasibleMask.shape)
print(result.acceptanceRate)
print(result.summary())
```

Important fields:

| Field | Meaning |
|---|---|
| `decs` | Decision draws with shape `(n_chains, draws, n_input)`. |
| `objs` | Objective values with shape `(n_chains, draws, n_output)`. |
| `cons` | Constraint values, or `None` for unconstrained problems. |
| `logProb` | Log-probability values with shape `(n_chains, draws)`. |
| `accepted` | Boolean mask showing which draws were accepted proposals. |
| `feasibleMask` | Boolean feasibility mask. |
| `acceptanceRate` | Acceptance rate per chain. |
| `bestDecs` | Best sampled parameter vector. |
| `bestObjs` | Objective value at `bestDecs`, reported in the original problem direction. |
| `bestCons` | Constraint values at `bestDecs`, or `None`. |
| `FEs` | Number of function evaluations. |
| `iters` | Final iteration count. |
| `history` | Runtime history snapshots. |

For example, to compute a simple posterior mean after dropping the first five draws:

```python
import numpy as np

from UQPyL.inference import MH
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=-2.0, ub=2.0, objFunc=objFunc, optType="min", name="SpherePosterior")
result = MH(nChains=3, warmUp=5, maxIters=30, verboseFlag=False, logFlag=False, saveFlag=False).run(problem, gamma=0.2, seed=123)
samples = result.decs[:, 5:, :].reshape(-1, result.nInput)
print(samples.mean(axis=0))
print(samples.std(axis=0))
```

This reshapes all chains into one sample table. Use a larger warm-up and sampling budget for real inference work.

## Read a Saved SQLite Result

Set `saveFlag=True` to save a sqlite result under `Result/`. Use `InfReader` to inspect it later.

For saved examples, prefer benchmark problems such as `Sphere`. They are easier to serialize than temporary functions typed in an interactive session.

```python
from pathlib import Path

import numpy as np

from UQPyL.inference import InfReader, MH
from UQPyL.problem import Sphere

np.set_printoptions(precision=4, suppress=True)

problem = Sphere(nInput=2, ub=2.0, lb=-2.0)
method = MH(nChains=2, warmUp=2, maxIters=10, verboseFlag=False, logFlag=False, saveFlag=True, saveFreq=2)
result = method.run(problem, gamma=0.2, seed=123)

dbPath = sorted(Path("Result").glob("mh_Sphere_*.sqlite3"))[-1]

with InfReader(dbPath) as reader:
    summary = reader.get_run_summary()
    params = reader.get_run_params()
    snapshots = reader.list_snapshots()
    members = reader.load_last_snapshot_members()
    loaded = reader.load_result()

print(dbPath.as_posix())
print(summary["method"], summary["problem_name"])
print(summary["final_fes"], summary["final_iters"])
print(params["seed"])
print(len(snapshots), len(members))
print(loaded.decs.shape)
print(loaded.acceptanceRate)
```

Example output:

```text
Result/mh_Sphere_20260510_1840_a780.sqlite3
MH Sphere
24 9
123
6 2
(2, 10, 2)
[0.6667 0.5556]
```

`snapshots` are saved runtime checkpoints. `members` are the chain members stored in the latest checkpoint. `loaded` is the final `InfResult`, so you can read it the same way as a result returned directly by `run()`.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Returning a one-dimensional objective such as `(n_samples,)`. | Return a column with shape `(n_samples, 1)`, for example `y.reshape(-1, 1)` or `keepdims=True`. |
| Passing a multi-objective problem to inference. | Combine outputs into one scalar score, or use optimization if you need a Pareto set. |
| Forgetting that `lb` and `ub` can be scalars or vectors. | Use scalars when all variables share the same bounds; use vectors when each variable has different bounds. |
| Using a very large `gamma`. | Reduce `gamma` when proposals are rejected too often. |
| Using a very small `gamma`. | Increase `gamma` when acceptance is high but chains barely move. |
| Treating `bestDecs` as the whole inference result. | Use the full chains in `result.decs` for uncertainty summaries. |
| Expecting constraints to be soft penalties. | Inference hard rejects infeasible proposals. Put soft penalties into the objective or `logProbFunc` if that is what you need. |
| Saving runs with locally defined custom functions and then moving code around. | For long-term saved runs, define problems and functions in importable Python files or use built-in benchmark problems. |

## Next Steps

| Goal | Read |
|---|---|
| Look up sampler constructors and result fields | [Inference API](api/inference.md) |
| Define scalar objectives, bounds, and constraints | [Problem](problem.md) |
| Compare optimization and inference workflows | [Optimization](optimization.md) |
| Build calibration workflows around parameter fitting | [Calibration](calibration.md) |
