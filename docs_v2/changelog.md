# Changelog

This page tracks user-facing changes in UQPyL and its documentation.

## 2.1.7 (Unreleased)

### Numerical and Runtime Fixes

- Fixed sensitivity-analysis scaling and extreme-range behavior, including DeltaTest cutoff ties; improved MARS holdout-validation warnings.
- Fixed optimizer selection, retention of evaluated best results, and numerical ranges in performance metrics.
- Fixed inference transition, adaptation, and probability-validation issues, with known-target distribution checks.
- Improved calibration posterior updates, score stability, and low-effective-sample-size diagnostics; unified calibration result fields.
- Improved surrogate defaults, solver stability, and uncertainty handling, with independent prediction accuracy checks.
- Fixed DoE edge cases, Problem range conversions, and reused buffers in single-point evaluation.
- Fixed SQLite creation-time consistency, sensitivity plot values, optimization history alignment, and reference-point plotting.

- Preserved masked-NaN handling in the canonical simulation matrix, fixed masked-objective documentation examples, and clarified surrogate prediction dimension errors.

### Interfaces and Environment

- `ModelProblem` now requires 1D `obs`/`mask` `(nObs,)` and 2D `simFunc` outputs `(nSamples,nObs)`. Users explicitly flatten source grids in a common observation order; legacy layouts are rejected.
- Removed observation labels (`obsLabels`). CalResult and SQLite summaries use observation counts (`nObs` / `n_obs`, `n_output=n_obs`); removed `nTime`, `nSeries`, `seriesLabels`, and `obsShape`. Old calibration databases require a new run.

- Python 3.10 is the minimum version; declared versions and build matrices now extend through Python 3.14.
- Surrogates use a single-output contract; compose multiple outputs with `MultiSurrogate`.
- Morris reports elementary effects relative to parameter ranges; MARS defaults to second-order interactions.
- `plot_sa` plots stored values without implicit normalization; select metrics and outputs using `metric` and `outputIndex`.
- This version includes interface and default-behavior changes. Check the relevant module guides when upgrading.

### Validation Status

- Current source passes 3175 tests in conda py312 with warnings treated as errors, including observation-vector contracts, coordinate-order checks, SQLite round trips and executable documentation. Four calibration methods retain exactly identical pre/post-migration numerical arrays.
- This is a release-preparation entry, not confirmation of a PyPI upload. Final 2.1.7 artifacts and cross-platform CI still require validation.

### Documentation

- Reworked the documentation into user-facing workflow guides.
- Split the API reference into module pages for `problem`, `doe`, `analysis`, `optimization`, `inference`, `calibration`, and `surrogate`.
- Expanded examples with runnable code, expected outputs, verbose output snippets, and common mistakes.
- Added more detailed explanations for `Problem` evaluation, batched objective functions, scalar/vector bounds, result objects, and saved sqlite readers.

### Notes

- `Problem.evaluate()` returns an `Eval` object. Use `res.objs` and `res.cons` instead of dictionary-style access.

## Released Versions

The following `v2.0.x` tags are visible in the local repository:

| Version | Date |
|---|---|
| `v2.0.11` | 2024-12-24 |
| `v2.0.10` | 2024-12-24 |
| `v2.0.9` | 2024-10-15 |
| `v2.0.8` | 2024-10-15 |
| `v2.0.7` | 2024-10-02 |
| `v2.0.6` | 2024-09-19 |
| `v2.0.5` | 2024-06-04 |
| `v2.0.4` | 2024-05-08 |
| `v2.0.3` | 2024-05-07 |
| `v2.0.2` | 2024-05-07 |
| `v2.0.1` | 2024-04-20 |