# Examples

This page contains complete runnable workflows that combine `Problem` or `ModelProblem` with UQPyL modules.

Each example describes a concrete toy problem, the part you usually replace in real work, and how to read the printed output.

The examples use small budgets so they run quickly. For real work, usually increase sample counts, optimization budgets, chain lengths, or calibration candidate sizes after the workflow is correct.

## Workflow Map

| Workflow | Example problem | Replace in real use |
|---|---|---|
| Sampling and sensitivity analysis | Weighted two-variable sphere | `objFunc` |
| Single-objective optimization | Minimize distance to the origin | `objFunc`, bounds, optimizer settings |
| Multi-objective optimization | ZDT1 benchmark Pareto search | Benchmark problem or custom multi-objective `Problem` |
| MCMC inference | Sample parameters around a low objective region | `objFunc` or `logProbFunc` |
| Model calibration | Match two simulated time steps to observations | `simFunc`, `obs`, candidate parameter matrix |
| Surrogate training and validation | Learn a nonlinear response surface | `objFunc`, sampler size, surrogate model |
| Surrogate-assisted optimization | Optimize a surrogate, then check with the original model | `expensiveObjFunc`, infill/update strategy |

## Sampling and Sensitivity Analysis

This example asks which of two inputs matters more for:

```text
y = x1^2 + 0.2 * x2^2
```

Because `x1` has the larger coefficient, the first sensitivity value should dominate. Replace `objFunc` with your model and keep the `LHS -> evaluate -> RBDFAST` structure.

Replace in real use:

| Example part | Replace with |
|---|---|
| `objFunc` | Your model, simulator wrapper, or objective calculation. |
| `nSamples=256` | A sample size suitable for your model cost and input dimension. |
| `RBDFAST` | Another analysis method if your question needs it. |

```python
import numpy as np

from UQPyL.analysis import RBDFAST
from UQPyL.doe import LHS
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] ** 2 + 0.2 * X[:, 1] ** 2).reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, ub=1.0, lb=-1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D")

# Sample the input space, evaluate the model, then estimate first-order effects.
X = LHS("classic").sample(problem, nSamples=256, seed=123)
Y = problem.evaluate(X).objs
result = RBDFAST(verboseFlag=False).analyze(problem, X, Y=Y)

print(result.metricNames)
print(result.getMetric("S1").values)
```

Output: `metricNames` shows which metrics were computed. `S1` is a row of first-order sensitivity values, one value per input.

Example output:

```text
['S1']
[[0.95097836 0.04752271]]
```

## Single-Objective Optimization

This example minimizes the two-dimensional sphere function:

```text
f(x) = x1^2 + x2^2
```

The best solution should move toward `[0, 0]`. Replace `objFunc` and bounds with your own objective.

Replace in real use:

| Example part | Replace with |
|---|---|
| `objFunc` | Your scalar objective. |
| `ub`, `lb` | Physical or practical parameter bounds. |
| `GA(...)` | The optimizer and budget you want to use. |

```python
import numpy as np

from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, ub=1.0, lb=-1.0, objFunc=objFunc, optType="min", name="Sphere2D")

# Keep the evaluation budget small for documentation; increase maxFEs for real runs.
algorithm = GA(nPop=8, maxFEs=40, maxIters=5, verboseFlag=False, logFlag=False, saveFlag=False)
result = algorithm.run(problem, seed=123)

print(result.bestDecs)
print(result.bestObjs)
```

Output: `bestDecs` is the best decision row found. `bestObjs` is the corresponding objective value.

Example output:

```text
[[0.00304088 0.10493151]]
[[0.01101987]]
```

## Verbose Runtime Output

Most runnable methods accept `verboseFlag`, `verboseFreq`, `logFlag`, and `saveFlag`.

Use `verboseFlag=True` when you want terminal progress. Use `verboseFreq` to control how often progress lines are printed. Keep `logFlag=False` and `saveFlag=False` for quick examples unless you want files under `Result/`.

```python
import numpy as np

from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, ub=1.0, lb=-1.0, objFunc=objFunc, optType="min", name="Sphere2D")
algorithm = GA(nPop=8, maxFEs=24, maxIters=3, verboseFlag=True, verboseFreq=1, logFlag=False, saveFlag=False)
result = algorithm.run(problem, seed=123)
```

Example output:

```text
Algorithm: GA
Problem: Sphere2D
nInput: 2
nObj: 1
maxFEs: 24
maxIters: 3
GA | iter=0 eval=8 best=1.8884e-02 cv=0 time=0.0s
Optimization finished
  algorithm        : GA
  status           : finished
  iterations       : 3
  evaluations      : 24
  best value       : 1.1468e-02
  best X           : [3.0409e-03, 1.0704e-01]
  constraint viol. : 0
  elapsed          : 0.0s
```

For inference methods, verbose output shows chain progress, current log probability, acceptance rate, feasibility, and final chain summary.

```python
import numpy as np

from UQPyL.inference import MH
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, ub=2.0, lb=-2.0, objFunc=objFunc, optType="min", name="SpherePosterior")
method = MH(nChains=2, warmUp=2, maxIters=8, verboseFlag=True, verboseFreq=2, logFlag=False, saveFlag=False)
result = method.run(problem, gamma=0.2, seed=123)
```

Example output:

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

## Multi-Objective Optimization

This example uses the built-in `ZDT1` benchmark, a two-objective problem with a Pareto front. Use it to confirm the multi-objective workflow before replacing it with a custom multi-objective `Problem`.

Replace in real use:

| Example part | Replace with |
|---|---|
| `ZDT1(nInput=5)` | Your multi-objective `Problem`. |
| `NSGAII(...)` | A multi-objective optimizer and budget for your case. |
| Final solution choice | A domain decision rule, not simply the first Pareto row. |

```python
from UQPyL.optimization.moea import NSGAII
from UQPyL.problem import ZDT1


problem = ZDT1(nInput=5)
algorithm = NSGAII(nPop=12, maxFEs=48, maxIters=3, verboseFlag=False, logFlag=False, saveFlag=False)
result = algorithm.run(problem, seed=123)

print(result.bestDecs.shape)
print(result.bestObjs.shape)
print(result.bestMetric)
```

Output: `bestDecs` and `bestObjs` contain the current Pareto set. `bestMetric` is the multi-objective progress metric.

Example output:

```text
(8, 5)
(8, 2)
0.4890752707178604
```

## MCMC Inference

This example treats a scalar objective as an unnormalized score. Lower sphere values become higher default log probability, so chains tend to spend more time near the origin.

Replace `objFunc` with a negative log-likelihood or provide a custom `logProbFunc` when the default conversion is not what you want.

Replace in real use:

| Example part | Replace with |
|---|---|
| `objFunc` | A scalar loss, energy, negative log-likelihood, or score. |
| `gamma=0.2` | A proposal scale tuned for your parameter units. |
| `warmUp`, `maxIters`, `nChains` | Larger values for real inference. |

```python
import numpy as np

from UQPyL.inference import MH
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, ub=2.0, lb=-2.0, objFunc=objFunc, optType="min", name="SpherePosterior")

# gamma controls proposal scale; tune it if acceptance is too high or too low.
method = MH(nChains=3, warmUp=5, maxIters=30, verboseFlag=False, logFlag=False, saveFlag=False)
result = method.run(problem, gamma=0.2, seed=123)

print(result.decs.shape)
print(result.logProb.shape)
print(result.acceptanceRate)
```

Output: `decs.shape` is `(n_chains, draws, n_input)`. `acceptanceRate` reports one acceptance rate per chain.

Example output:

```text
(3, 30, 2)
(3, 30)
[0.5862069  0.55172414 0.5862069 ]
```

## Model Calibration

This example calibrates two parameters against two observed time steps:

```text
obs = [[1.0], [2.0]]
sim(t1) = x1
sim(t2) = x2
```

The candidate `[1.0, 2.0]` is a perfect match. Replace `simFunc`, `obs`, and candidate matrix `X` with your simulation model and parameter sets.

Replace in real use:

| Example part | Replace with |
|---|---|
| `simFunc` | Your simulation model wrapper. |
| `obs` | Observed time series or measured values. |
| `X` | Candidate parameter sets from DOE, prior samples, or an iterative method. |

```python
import numpy as np

from UQPyL.calibration import GLUE
from UQPyL.problem import ModelProblem


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))  # (n_samples, n_time, n_series)
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, simLabels=["Q"], name="ToyModel")
X = np.array([[1.0, 2.0], [1.0, 2.4], [0.0, 0.0]])

result = GLUE(metric="rmse", verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X, threshold=0.3)

print(result.bestDecs)
print(result.bestSim)
print(result.behavioralDecs)
```

Output: `bestDecs` is the best parameter row. `behavioralDecs` are the candidates accepted by the GLUE threshold.

Example output:

```text
[[1. 2.]]
[[1. 2.]]
[[1.  2. ]
 [1.  2.4]]
```

## Surrogate Training and Validation

This example samples a nonlinear response surface, trains an `RBF` surrogate, and validates it on a held-out split:

```text
y = sin(pi * x1) + x2^2
```

Replace `objFunc` with the expensive function you want to approximate.

Replace in real use:

| Example part | Replace with |
|---|---|
| `objFunc` | The expensive response you want to approximate. |
| `nSamples=30` | A training size appropriate for input dimension and model cost. |
| `RBF()` | A surrogate model that fits your response behavior. |

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.problem import Problem
from UQPyL.surrogate import RandSelect, mse, r_square
from UQPyL.surrogate.rbf import RBF

np.random.seed(123)


def objFunc(X):
    X = np.atleast_2d(X)
    return (np.sin(np.pi * X[:, 0]) + X[:, 1] ** 2).reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, ub=1.0, lb=0.0, objFunc=objFunc, optType="min", name="SurrogateToy")

# Generate training data, hold out 25%, then score the surrogate on unseen points.
X = LHS("classic").sample(problem, nSamples=30, seed=123)
Y = problem.evaluate(X).objs
trainIdx, testIdx = RandSelect(pTest=25).split(X)

model = RBF()
model.fit(X[trainIdx], Y[trainIdx])
pred = model.predict(X[testIdx])

print(pred.shape)
print(r_square(Y[testIdx], pred))
print(mse(Y[testIdx], pred))
```

Output: `pred.shape` is the prediction matrix for the test set. `r_square` should be closer to `1.0` when the surrogate fits well; `mse` should be small.

Example output:

```text
(7, 1)
0.9944563882440021
[0.00043401]
```

## Surrogate-Assisted Optimization Pattern

This example shows the basic manual pattern:

1. Evaluate the expensive model at DOE points.
2. Train a surrogate on those evaluations.
3. Optimize the surrogate cheaply.
4. Re-check the surrogate optimum with the expensive model.

It is intentionally a one-pass pattern. Real surrogate-assisted optimization usually repeats this loop with new infill points.

Replace in real use:

| Example part | Replace with |
|---|---|
| `expensiveObjFunc` | Your expensive objective. |
| Initial DOE size | Enough real evaluations to train a reliable first surrogate. |
| One-pass pattern | An iterative infill/update loop if the final accuracy matters. |

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem
from UQPyL.surrogate.rbf import RBF


def expensiveObjFunc(X):
    X = np.atleast_2d(X)
    return (np.sin(np.pi * X[:, 0]) + X[:, 1] ** 2).reshape(-1, 1)


expensiveProblem = Problem(nInput=2, nObj=1, ub=1.0, lb=0.0, objFunc=expensiveObjFunc, optType="min", name="ExpensiveToy")

# Train the surrogate from real evaluations.
X = LHS("classic").sample(expensiveProblem, nSamples=30, seed=123)
Y = expensiveProblem.evaluate(X).objs
surrogate = RBF()
surrogate.fit(X, Y)


def surrogateObjFunc(Xnew):
    return surrogate.predict(Xnew)


surrogateProblem = Problem(nInput=2, nObj=1, ub=1.0, lb=0.0, objFunc=surrogateObjFunc, optType="min", name="SurrogateToy")
result = GA(nPop=8, maxFEs=40, maxIters=5, verboseFlag=False, logFlag=False, saveFlag=False).run(surrogateProblem, seed=123)

# Always verify the surrogate optimum with the original expensive model.
checkedObj = expensiveProblem.evaluate(result.bestDecs).objs

print(result.bestDecs)
print(result.bestObjs)
print(checkedObj)
```

Output: `result.bestObjs` is the surrogate-predicted objective. `checkedObj` is the actual objective at the same decision row.

Example output:

```text
[[0.96350312 0.06192555]]
[[0.13448425]]
[[0.11824205]]
```

For built-in expensive optimization algorithms, see [Optimization](optimization.md) and [Optimization API](api/optimization.md).

## Next Steps

| Goal | Read |
|---|---|
| Understand the shared problem protocol | [Problem](problem.md) |
| Generate samples | [Design of Experiment](doe.md) |
| Run sensitivity analysis | [Analysis](analysis.md) |
| Optimize objectives | [Optimization](optimization.md) |
| Run inference | [Inference](inference.md) |
| Calibrate simulation models | [Calibration](calibration.md) |
| Train surrogate models | [Surrogate Modeling](surrogate.md) |
