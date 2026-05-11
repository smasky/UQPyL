# API Reference

This page is the API entry point for UQPyL.

Use it when you know what object you need, but not which module page contains it. For tutorials and workflows, start from [Quick Start](quick_start.md) or [Examples](examples.md).

## Module Pages

| Module | API page | Main purpose |
|---|---|---|
| `UQPyL.problem` | [Problem API](api/problem.md) | Modeling protocol, input spaces, evaluation results, decorators, and benchmark problems. |
| `UQPyL.doe` | [Design of Experiment API](api/doe.md) | Sampling classes, design metadata, and analysis-specific designs. |
| `UQPyL.analysis` | [Analysis API](api/analysis.md) | Sensitivity analysis methods, metrics, result objects, and saved-result reader. |
| `UQPyL.optimization` | [Optimization API](api/optimization.md) | Single-objective, multi-objective, and expensive optimization APIs. |
| `UQPyL.inference` | [Inference API](api/inference.md) | MCMC-style inference methods, chain results, and saved-run reader. |
| `UQPyL.calibration` | [Calibration API](api/calibration.md) | Calibration methods, metric handling, result objects, and saved-run reader. |
| `UQPyL.surrogate` | [Surrogate API](api/surrogate.md) | Surrogate models, scaling, splitting, metrics, uncertainty output, and tuning. |

## Common Workflow Protocols

Most public APIs follow one of these patterns.

| Workflow | Call pattern | Return |
|---|---|---|
| Problem evaluation | `problem.evaluate(X)` | `Eval` |
| Sampling | `sampler.sample(problem, nSamples=..., seed=...)` | `np.ndarray` |
| Sampling with metadata | `sampler.sampleWithMeta(problem, ...)` | `(X, meta)` |
| Analysis | `method.analyze(problem, X, Y=..., meta=...)` | `AnaResult` |
| Optimization | `algorithm.run(problem, seed=...)` | `OptResult` |
| Inference | `method.run(problem, gamma=..., seed=...)` | `InfResult` |
| Calibration | `method.run(modelProblem, ...)` | `CalResult` |
| Surrogate training | `model.fit(X, Y)` | fitted model |
| Surrogate prediction | `model.predict(Xnew)` | `np.ndarray` or `(mean, std/var)` |

## Core Data Objects

| Object | Where | Role |
|---|---|---|
| `Problem` | [Problem API](api/problem.md) | Defines objectives, constraints, bounds, labels, and variable types. |
| `ModelProblem` | [Problem API](api/problem.md) | Defines simulation models with observations, masks, and simulation context. |
| `Eval` | [Problem API](api/problem.md) | Standard return object from `evaluate()`. |
| `AnaResult` | [Analysis API](api/analysis.md) | Sensitivity-analysis result. |
| `OptResult` | [Optimization API](api/optimization.md) | Optimization result. |
| `InfResult` | [Inference API](api/inference.md) | Inference chain result. |
| `CalResult` | [Calibration API](api/calibration.md) | Calibration result. |

`Problem.evaluate()` returns an `Eval` object, not a dictionary. Use:

```python
res = problem.evaluate(X)
objs = res.objs
cons = res.cons
sim = res.sim
```

## Saved Result Readers

Runtime methods can save sqlite results when `saveFlag=True`.

| Module | Reader | Typical import |
|---|---|---|
| Analysis | `AnaReader` | `from UQPyL.analysis.runtime import AnaReader` |
| Optimization | `OptReader` | `from UQPyL.optimization.runtime import OptReader` |
| Inference | `InfReader` | `from UQPyL.inference import InfReader` |
| Calibration | `CalReader` | `from UQPyL.calibration import CalReader` |

Typical reader pattern:

```python
with Reader("Result/example.sqlite3") as reader:
    summary = reader.get_run_summary()
    result = reader.load_result()
```

The exact reader class and available methods are documented on each module API page.

## Runtime Controls

Runnable analysis, optimization, inference, and calibration methods commonly accept:

| Option | Meaning |
|---|---|
| `verboseFlag` | Print runtime progress or final summaries. |
| `verboseFreq` | Progress print interval where supported. |
| `logFlag` | Write text logs where supported. |
| `saveFlag` | Save structured sqlite output where supported. |
| `saveFreq` | Snapshot interval where supported. |

For quick examples, set `verboseFlag=False`, `logFlag=False`, and `saveFlag=False`. For reproducible runs, pass an explicit `seed` to `sample()`, `analyze()`, or `run()` when the method supports it.

## Import Cheat Sheet

| Need | Import |
|---|---|
| Define ordinary problems | `from UQPyL.problem import Problem` |
| Define simulation problems | `from UQPyL.problem import ModelProblem` |
| Generate LHS samples | `from UQPyL.doe import LHS` |
| Run free-design sensitivity analysis | `from UQPyL.analysis import RBDFAST` |
| Run single-objective optimization | `from UQPyL.optimization.soea import GA` |
| Run multi-objective optimization | `from UQPyL.optimization.moea import NSGAII` |
| Run MCMC inference | `from UQPyL.inference import MH` |
| Run GLUE calibration | `from UQPyL.calibration import GLUE` |
| Train an RBF surrogate | `from UQPyL.surrogate.rbf import RBF` |

## Naming Notes

| Name | Note |
|---|---|
| `objs` | Objective matrix, usually shape `(n_samples, n_obj)`. |
| `cons` | Constraint matrix. Values `<= 0` are feasible. |
| `decs` | Decision/input matrix or sampled decision chains, depending on result object. |
| `sims` / `sim` | Simulation output from `ModelProblem`; shape is usually `(n_samples, n_time, n_series)`. |
| `bestDecs` | Best decision row or Pareto decision matrix, depending on method type. |
| `bestObjs` | Objective value(s) associated with `bestDecs`. |

## Scope Status

All current core API modules are listed on this page. Module-specific parameters, fields, and reader methods live in the split API pages above.
