# Calibration

The `calibration` module calibrates simulation models represented by `ModelProblem`.

## Public Methods

| Method | Purpose |
|---|---|
| `GLUE` | Generalized likelihood uncertainty estimation. |
| `SUFI2` | Sequential uncertainty fitting. |
| `ES` | Ensemble smoother. |
| `IES` | Iterative ensemble smoother. |

## Runtime Persistence

Set `saveFlag=True` on a calibration method to write a sqlite file under
`<problem.workDir>/Result` or `<cwd>/Result`.

```python
from UQPyL.calibration import CalReader, GLUE

method = GLUE(saveFlag=True, verboseFlag=False)
result = method.run(problem, X, threshold=0.5)

runs = CalReader.list_runs(problem.workDir)
with CalReader(runs[-1]["dbPath"]) as reader:
    saved = reader.load_result()
```

Calibration persistence currently saves the final `CalResult` and related
artifacts only. It does not save per-iteration snapshots or support
`saveFreq`. Artifact payloads are stored as pickle blobs for exact Python
round-tripping; queryable SQL tables can be added later when concrete query
requirements are defined.

## Planned Sections

1. Building a `ModelProblem`
2. Observations, masks, and simulation labels
3. Calibration metrics
4. Method workflows
5. `CalResult` outputs
