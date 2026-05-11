# UQPyL Documentation

UQPyL provides a shared problem protocol and a set of uncertainty quantification workflows for sampling, sensitivity analysis, optimization, inference, calibration, and surrogate modeling.

Most workflows start with a `Problem` or `ModelProblem`, then pass that object into one or more functional modules.

```text
Problem / ModelProblem -> Method -> Result
```

## Start Here

| Goal | Read |
|---|---|
| Run the shortest complete workflow | [Quick Start](quick_start.md) |
| Understand the shared modeling protocol | [Problem](problem.md) |
| Find complete workflow examples | [Examples](examples.md) |
| Look up classes, parameters, and result objects | [API Reference](api_reference.md) |

## Choose a Workflow

| Task | Module Guide | API |
|---|---|---|
| Define input spaces, objectives, constraints, simulations, and evaluation outputs | [Problem](problem.md) | [Problem API](api/problem.md) |
| Generate design samples for experiments, analysis, initialization, or modeling | [Design of Experiment](doe.md) | [DOE API](api/doe.md) |
| Analyze how inputs affect model or objective outputs | [Analysis](analysis.md) | [Analysis API](api/analysis.md) |
| Search for single-objective, multi-objective, or expensive-model optima | [Optimization](optimization.md) | [Optimization API](api/optimization.md) |
| Run MCMC-style parameter inference | [Inference](inference.md) | [Inference API](api/inference.md) |
| Calibrate simulation models against observations | [Calibration](calibration.md) | [Calibration API](api/calibration.md) |
| Train predictive surrogate models | [Surrogate Modeling](surrogate.md) | [Surrogate API](api/surrogate.md) |

## Recommended Learning Path

1. Read [Quick Start](quick_start.md) to see the main workflow shape.
2. Read [Problem](problem.md) before using any functional module.
3. Pick one task guide from the workflow table.
4. Use [API Reference](api_reference.md) when you need constructor parameters, return fields, or reader classes.
5. Use [Examples](examples.md) for end-to-end patterns.

## Core Concepts

| Concept | Where |
|---|---|
| `Problem` | Static objective and constraint problems. See [Problem](problem.md). |
| `ModelProblem` | Simulation models with observations, masks, and simulation context. See [Problem](problem.md). |
| `Eval` | Standard output object from `problem.evaluate()`. See [Problem API](api/problem.md). |
| Result objects | Module-specific outputs such as `AnaResult`, `OptResult`, `InfResult`, and `CalResult`. See [API Reference](api_reference.md). |
| Saved runs | Runtime sqlite readers such as `AnaReader`, `OptReader`, `InfReader`, and `CalReader`. See each API page. |

## API Reference

The API reference is split by module.

| Module | API Page |
|---|---|
| `UQPyL.problem` | [Problem API](api/problem.md) |
| `UQPyL.doe` | [DOE API](api/doe.md) |
| `UQPyL.analysis` | [Analysis API](api/analysis.md) |
| `UQPyL.optimization` | [Optimization API](api/optimization.md) |
| `UQPyL.inference` | [Inference API](api/inference.md) |
| `UQPyL.calibration` | [Calibration API](api/calibration.md) |
| `UQPyL.surrogate` | [Surrogate API](api/surrogate.md) |

## Project Notes

| Page | Purpose |
|---|---|
| [Changelog](changelog.md) | User-facing changes across releases. |
