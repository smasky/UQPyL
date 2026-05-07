# Runtime 与 Parameter 架构约定

## 1. Runtime 命名规则

- 内部实现使用驼峰。
- 对外协议使用 `snake_case`。
- SQLite 历史列名默认保持不动，除非有明确迁移收益。

这里的“对外协议”包括：

- `summary()`
- `toDict()`
- reader summary payload
- `list_runs()` 返回结果

这里的“内部实现”包括：

- 可变运行态字段
- result 对象字段
- runtime base/session 成员
- verbose/runtime helper 内部细节

## 2. Runtime 分层

runtime 现在按五层理解：

1. base
2. state/result
3. storage
4. reader
5. verbose/viz

### 2.1 base

base 类负责一次 run 的生命周期：

- `setup(problem, seed)`
- 核心循环
- `finalize()`

base 层只保留一个运行句柄：

- `session: RunSession | None`

不要重新引入 `storageCtx` 这种重复状态。

### 2.2 state/result

- `State` 是运行中不断更新的可变状态。
- `Result` 是收尾时构造出的稳定结果对象。

当前映射：

- analysis: `AnaState -> AnaResult`
- inference: `InfState -> InfResult`
- optimization: `OptState -> OptResult`
- calibration: `CalState -> CalResult`

约定：

- result 对象内部字段继续用驼峰
- result 导出 payload 用 `snake_case`

### 2.3 storage

storage 层负责：

- sqlite schema 创建
- run 记录创建
- snapshot 持久化
- finalization

共享行为在 `core.runtime_storage.BaseSqliteStorage`。

### 2.4 reader

reader 主入口统一用 `snake_case`：

- `get_run()`
- `get_run_params()`
- `get_run_summary()`
- `load_problem()`
- `list_runs()`

模块专属读取接口也优先用 `snake_case`：

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

verbose 保持各模块独立实现。
如果行为差异明显，不强行抽共享 renderer。

可以共享的薄行为：

- resolve run id
- ensure result dir
- save log/save artifact helper

不值得强抽的部分：

- summary 格式化
- progress 渲染
- final result 渲染

## 3. Shared runtime export helper

共享 helper 放在 `core.runtime`：

- `export_runtime_meta(...)`
- `export_reader_summary(...)`

适用场景：

- `summary()`
- reader summary payload 组装

目标：

- 公共键保持一致
- 避免 analysis/inference/optimization/calibration 之间重复漂移

## 4. 当前 runtime 对外协议

公共 summary 键优先统一为：

- `run_id`
- `method`
- `problem_name`
- `n_input`
- `n_output`
- `n_con`
- `runtime`
- `created_at`

模块专属键允许扩展，但继续使用 `snake_case`。

示例：

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

## 5. Parameter 架构

现在的参数体系分三层：

1. `ParameterStore`
2. `Params`
3. `Setting`

### 5.1 ParameterStore

`core.parameter_store.ParameterStore` 是最小共享底座。

它定义：

- `dicts`
- `keys()`
- `values()`
- `items()`
- `asDict()`

这个类必须保持轻量。
不要把 surrogate 专属调参语义塞进来。

### 5.2 Params

`core.params.Params` 是 runtime 使用的轻量参数容器。

适合放：

- runtime flag
- 简单模块设置
- 扁平参数包

它应该一直保持简单。

### 5.3 Setting

`surrogate.setting.Setting` 不是普通 dict。
它是 surrogate 专属的参数空间对象。

它负责：

- tunable parameter value
- constant parameter value
- bounds
- parameter type
- categorical set
- log flag
- owner 分层
- merge/remove 操作
- tuning-space encode/decode

不要把 `Setting` 直接压扁成 `Params`。

## 6. 参数演进原则

安全方向：

- 共享最小容器能力
- surrogate 调参语义继续留在 `Setting`

危险方向：

- 把 surrogate 的调参复杂度推到 `Params`
- 让 runtime 模块依赖 bounds/type/categorical encoder

## 7. Setting 同步规则

`Setting` 不能维护一份会过期的重复平面 map。

当前规则：

- 权威状态在 `parVal` 和 `parCon`
- map-like 视图通过 `_mapping()` 动态物化

所以如果这些操作改变了参数状态：

- `setVals`
- `mergeSetting`
- `removeParas`
- owner-based removal

那么：

- `asDict()`
- `dicts`
- `keys()/values()/items()`

都必须立刻反映真实状态。

## 8. 重构指导

触达 runtime 时：

- 内部继续驼峰
- 导出继续 `snake_case`
- 优先补 shared export helper
- 谨慎引入重抽象

触达 surrogate parameter code 时：

- 小步演进 `ParameterStore`
- 不破坏 `Setting` 外部行为
- tuning-space 逻辑除非出现真正共享抽象，否则继续留在 surrogate 层

## 9. 不要做什么

- 不要重新引入 `storageCtx`
- 不要强行统一 verbose renderer
- 不要仅为了风格重命名 sqlite 列
- 不要把 `Setting` 替换成 `Params`
- 不要维护会 silently drift 的并行 value store
