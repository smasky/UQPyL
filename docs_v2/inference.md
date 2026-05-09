# Inference

The `inference` module runs MCMC-style parameter inference on scalar-objective problems.

## Public Methods

| Method | Purpose |
|---|---|
| `MH` | Metropolis-Hastings sampler. |
| `AMH` | Adaptive Metropolis-Hastings sampler. |
| `MH_Gibbs` | Metropolis-Hastings within Gibbs sampler. |
| `DEMC` | Differential evolution MCMC sampler. |
| `DREAM_ZS` | DREAM(ZS) sampler. |

## Planned Sections

1. Inference problem requirements
2. Objective-to-log-probability convention
3. Chains, warm-up, and proposal scale
4. Constraints
5. `InfResult` and saved runs

