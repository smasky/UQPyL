# Changelog

This page tracks user-facing changes in UQPyL and its documentation.

## Unreleased

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