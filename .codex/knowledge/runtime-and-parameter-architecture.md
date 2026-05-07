# Runtime And Parameter Architecture

## 1. Runtime naming rule

- Internal implementation uses camelCase.
- External protocol uses snake_case.
- SQLite historical column names stay unchanged unless there is a strong migration reason.

External protocol means:

- `summary()`
- `toDict()`
- reader summary payloads
- `list_runs()` results

Internal implementation means:

- mutable state fields
- result object fields
- runtime base/session members
- verbose/runtime helper internals

## 2. Runtime layering

Runtime is split into five layers:

1. base
2. state/result
3. storage
4. reader
5. verbose/viz

### 2.1 base

Base classes own run lifecycle:

- `setup(problem, seed)`
- core loop
- `finalize()`

Base classes should keep only one runtime handle:

- `session: RunSession | None`

Do not reintroduce duplicate state like `storageCtx`.

### 2.2 state/result

State is mutable during one run.
Result is the stable exported object built from state.

Current mapping:

- analysis: `AnaState -> AnaResult`
- inference: `InfState -> InfResult`
- optimization: `OptState -> OptResult`
- calibration: `CalState -> CalResult`

Result object fields stay camelCase.
Result export payload stays snake_case.

### 2.3 storage

Storage owns:

- sqlite schema creation
- run record creation
- snapshot persistence
- finalization

Shared storage behavior lives in `core.runtime_storage.BaseSqliteStorage`.

### 2.4 reader

Reader main entrypoints should use snake_case:

- `get_run()`
- `get_run_params()`
- `get_run_summary()`
- `load_problem()`
- `list_runs()`

Module-specific read APIs should also prefer snake_case:

- optimization:
  - `list_snapshots()`
  - `load_population()`
  - `load_best()`
  - `load_last_population()`
  - `load_last_best()`
- inference:
  - `list_snapshots()`
  - `load_snapshot_members()`
  - `load_last_snapshot_members()`
  - `load_result()`
- analysis:
  - `get_metrics()`
  - `get_metric()`
  - `get_artifacts()`
  - `load_result()`

### 2.5 verbose/viz

Verbose implementations remain separate per module.
Do not force a shared renderer abstraction if behavior differs materially.

Allowed shared behavior:

- resolve run id
- ensure result dir
- save log/save artifact helpers

Not worth forcing shared:

- summary formatting
- progress rendering
- final result rendering

## 3. Shared runtime export helpers

Shared helpers live in `core.runtime`:

- `export_runtime_meta(...)`
- `export_reader_summary(...)`

Use them when adding or changing:

- `summary()`
- reader summary payload assembly

Goal:

- keep common keys aligned
- avoid copy-paste drift across analysis/inference/optimization/calibration

## 4. Current runtime protocol contract

Common external summary keys should prefer:

- `run_id`
- `method`
- `problem_name`
- `n_input`
- `n_output`
- `n_con`
- `runtime`
- `created_at`

Module-specific keys are allowed, but should still use snake_case.

Examples:

- analysis:
  - `target`
  - `metric_names`
- inference:
  - `n_chains`
  - `draws`
  - `fes`
  - `best_feasible`
- optimization:
  - `best_feasible`
  - `appear_fes`
  - `appear_iters`
- calibration:
  - `n_time`
  - `n_series`
  - `n_obs`
  - `best_score`
  - `best_x`

## 5. Parameter architecture

There are now three layers:

1. `ParameterStore`
2. `Params`
3. `Setting`

### 5.1 ParameterStore

`core.parameter_store.ParameterStore` is the minimal shared base.

It defines:

- `dicts`
- `keys()`
- `values()`
- `items()`
- `asDict()`

It is intentionally small.
It should not absorb surrogate-specific tuning semantics.

### 5.2 Params

`core.params.Params` is the lightweight runtime container.

Use it for:

- runtime flags
- simple module settings
- flat parameter bags

It should stay simple.

### 5.3 Setting

`surrogate.setting.Setting` is not just a dict.
It is a surrogate-specific parameter-space object.

It owns:

- tunable parameter values
- constant parameter values
- bounds
- parameter types
- categorical sets
- log flags
- owner partitioning
- merge/remove operations
- tuning-space encoding and decoding

Do not collapse `Setting` into `Params`.

## 6. Parameter evolution rule

Safe direction:

- share minimal container behavior
- keep surrogate tuning semantics local to `Setting`

Unsafe direction:

- push surrogate tuning complexity into `Params`
- make runtime modules depend on bounds/types/categorical encoders

## 7. Setting synchronization rule

`Setting` must not maintain a stale duplicated flat map.

Current rule:

- authoritative state lives in `parVal` and `parCon`
- map-like views are materialized dynamically via `_mapping()`

If `setVals`, `mergeSetting`, `removeParas`, or owner-based removal changes parameter state,
`asDict()` and related map-like APIs must reflect it immediately.

## 8. Refactor guidance

When touching runtime:

- keep internal camelCase
- keep exported snake_case
- prefer adding shared export helpers over shared heavy abstractions

When touching surrogate parameter code:

- evolve `ParameterStore` carefully
- do not break `Setting` external behavior
- keep tuning-space logic local unless a truly shared abstraction emerges

## 9. What not to do

- Do not reintroduce `storageCtx`.
- Do not force verbose renderer unification.
- Do not rename sqlite columns just for style.
- Do not replace `Setting` with `Params`.
- Do not maintain parallel value stores that can drift silently.
