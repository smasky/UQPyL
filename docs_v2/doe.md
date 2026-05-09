# Design of Experiment

The `doe` module generates parameter samples from a `Problem` input space.

## Public Methods

| Method | Purpose |
|---|---|
| `LHS` | Latin hypercube sampling. |
| `FFD` | Full factorial design. |
| `Random` | Random sampling. |
| `Sobol` | Sobol sequence sampling. |
| `SaltelliDesign` | Sampling design for Sobol analysis. |
| `FASTDesign` | Sampling design for FAST analysis. |
| `MorrisDesign` | Sampling design for Morris analysis. |

## Planned Sections

1. Sampling from a `Problem`
2. Reproducibility with `seed`
3. Unit-space to problem-space mapping
4. Designs for analysis modules

