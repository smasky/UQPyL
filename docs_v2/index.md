# UQPyL Documentation

UQPyL is organized around one core modeling module and six functional modules.

## Documentation Map

| Area | Module | Purpose |
|---|---|---|
| Core | `problem` | Define parameter spaces, objectives, constraints, simulations, and evaluation outputs. |
| Function | `doe` | Generate design samples for experiments, analysis, initialization, and modeling. |
| Function | `analysis` | Analyze how inputs affect model or objective outputs. |
| Function | `optimization` | Search for single-objective, multi-objective, or expensive-model optima. |
| Function | `inference` | Run MCMC-style parameter inference. |
| Function | `calibration` | Calibrate simulation models against observations. |
| Function | `surrogate` | Train and evaluate surrogate models. |

## Learning Path

1. Start with [Quick Start](quick_start.md).
2. Read [Problem](problem.md) to understand the shared modeling protocol.
3. Choose one functional module based on your workflow.
4. Use [Examples](examples.md) for complete end-to-end patterns.

