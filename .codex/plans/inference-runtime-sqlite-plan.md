# Inference Runtime SQLite 统一设计方案

## 背景

`doe`、`analysis`、`problem`、`optimization` 模块已经逐步整理到统一的运行时协议。`inference` 也应采用类似结构，把算法逻辑、结果对象、verbose、sqlite 保存和读取分开，避免算法类继续承担过多运行时职责。

本轮设计只保留 `sqlite3` 作为持久化格式，暂不引入 `xarray`、`netCDF`、`npz` 等其他保存路径。

## 总体原则

1. 正式返回协议使用 `InfResult`，不再返回 `dict` 或 `xarray.Dataset`。
2. 内部数据天然按多链组织，单链只是 `nChains == 1` 的特例。
3. 第一阶段只支持单目标标量 inference，即 `problem.nOutput == 1`。
4. 目标值内部统一转成最小化方向存储：`res.objs * problem.opt`。
5. 默认 `log_prob = -oriented_obj`，高级用户可通过 `logProbFunc` 自定义。
6. 约束只按硬约束处理：初始样本必须可行，候选点不可行则直接拒绝。
7. `warmUp` 不进入正式 trace，但所有 `problem.evaluate` 都计入 `FEs`。
8. verbose 终端输出缩略信息，log 文件输出完整向量摘要。

## 目录结构

```text
UQPyL/inference/
  base.py
  chain.py
  mh.py
  amh.py
  demc.py
  dream_zs.py
  mh_gibbs.py
  runtime/
    __init__.py
    result.py
    storage.py
    reader.py
    verbose.py
```

## 多链数据模型

正式结果统一按多链数组保存：

```python
decs.shape == (nChains, draws, nInput)
objs.shape == (nChains, draws, 1)
cons.shape == (nChains, draws, nCons)  # 无约束时为 None
logProb.shape == (nChains, draws)
accepted.shape == (nChains, draws)
feasibleMask.shape == (nChains, draws)
```

算法链数约束：

- `MH`、`AMH`、`MH_Gibbs`: `nChains >= 1`
- `DEMC`、`DREAM_ZS`: `nChains >= 3`

## 运行时对象

### InfResult

`InfResult` 是 inference 的正式返回对象，类似 optimization 的 `OptResult`。核心字段包括：

```python
runId: str | None
method: str
problemName: str
nInput: int
nOutput: int
nCon: int
settings: dict
runtime: float
createdAt: str
decs: np.ndarray
objs: np.ndarray
cons: np.ndarray | None
logProb: np.ndarray
accepted: np.ndarray
feasibleMask: np.ndarray
acceptanceRate: np.ndarray
bestDecs: np.ndarray | None
bestObjs: np.ndarray | None
bestCons: np.ndarray | None
bestFeasible: bool
FEs: int
iters: int
history: InfHistory
diagnostics: dict
extra: dict
```

`InfResult.objs` 保存内部最小化方向目标值；`bestObjs` 按用户原始 `optType` 方向还原，便于与 problem 定义一致。

### InfHistory

`InfHistory` 只保存过程摘要，用于 verbose、sqlite snapshot 和后处理：

```python
snapshots
iterToFEs
meanLogProbHistory
acceptanceRateHistory
feasibleRateHistory
bestObjHistory
```

### InfState

`InfState` 是运行中的可变状态，负责从 chains 收集当前 trace、更新诊断指标，并构造最终 `InfResult`。

## InferenceABC 职责

`InferenceABC` 统一处理：

- `setup/reset/finalize`
- problem 校验：单目标标量
- 初始采样和硬约束可行性筛选
- `evaluate` 的 `Eval` 协议读取
- 内部目标方向转换
- `log_prob` 默认计算与用户自定义入口
- verbose、log、sqlite snapshot

算法类只保留 proposal、acceptance 循环和算法特有参数。

## logProb 语义

默认规则：

```python
orientedObjs = res.objs * problem.opt
log_prob = -orientedObjs[..., 0]
```

如果用户需要真实贝叶斯 posterior，可传入：

```python
logProbFunc(objs, decs=None, cons=None)
```

此函数接收已经按内部方向处理后的 `objs`。软约束、先验、似然等特殊语义也应在 `objFunc` 或 `logProbFunc` 中表达。

## 约束语义

第一阶段只支持硬约束：

- `initialSampling` 反复 LHS，直到收集到足够的可行初始链。
- `maxInitAttempts` 控制最大重采样批次数。
- 运行中候选点若 `cons > 0`，直接拒绝。
- 软约束不在框架层单独建模。

## sqlite3 保存

文件命名：

```text
Result/{method}_{problem}_{yyyymmdd_hhmm}_{suffix}.sqlite3
```

表结构：

- `run`: 一次运行的元信息和 problem payload。
- `runParam`: 参数快照。
- `snapshot`: 每个 `saveFreq` 时刻的摘要。
- `snapshotMember`: 每条链在 snapshot 时刻的位置、目标、约束和接受状态。
- `artifact`: 最终完整 `InfResult` 的 pickle payload。

`SqliteStorage` 使用 `PRAGMA journal_mode=MEMORY`，减少可见的 `sqlite3-journal` 临时文件。

## Reader 接口

`InfReader` 提供：

```python
InfReader.listRuns(resultDir)
reader.getRun()
reader.getRunParams()
reader.loadProblem()
reader.listSnapshots()
reader.loadSnapshotMembers(snapshotId)
reader.loadLastSnapshotMembers()
reader.loadResult()
```

## verbose 与 log

终端输出压缩进度：

```text
MH | iter=100 eval=404 curLogp=-1.23e+00 accept=2.70e-01 feasible=1.00e+00 best=3.21e-04 time=1.2s
```

log 文件输出完整 summary 和 final block，包括完整 `bestX`、`meanX`、`stdX` 向量。完整 trace 由 sqlite 保存，不在 log 中逐 draw 展开。

## 当前结论

- 框架完整性以 `InfResult + runtime + sqlite + reader + benchmark tests` 为第一阶段闭环。
- 暂不支持多目标和时间序列输出。
- 暂不提供 xarray/netCDF 导出。
- 后续可扩展方向是诊断指标、更多 posterior 示例，以及更严格的 MCMC benchmark。
