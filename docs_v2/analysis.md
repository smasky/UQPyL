# Analysis

The `analysis` module evaluates how input variables affect output behavior.

## Public Methods

| Method | Purpose |
|---|---|
| `Sobol` | Variance-based global sensitivity analysis. |
| `FAST` | Fourier amplitude sensitivity test. |
| `RBDFAST` | Random balance design FAST. |
| `Morris` | Screening method for factor effects. |
| `RSA` | Regional sensitivity analysis. |
| `DeltaTest` | Distribution-based sensitivity analysis. |
| `MARS` | MARS-based analysis when optional dependencies are available. |

## Planned Sections

1. Analysis workflow
2. Using `X`, `Y`, and sampling `meta`
3. Targets and output indexes
4. `AnaResult` and metrics

