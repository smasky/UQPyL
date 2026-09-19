# Inference API

DEMC 默认 nChains=3；构造时要求整数且至少为 3，拒绝布尔值。 各 Reader 的 `list_runs()` 使用 `run_id`、`created_at`、`finished_at`、`final_fes`/`final_iters`（适用时）、`db_path`、`file_name`；数据库列名及内部对象字段保持原协议。

## `UQPyL.inference`

`inference` 模块对标量 `Problem` 运行 MCMC 风格采样。

## 导入

```python
from UQPyL.inference import MH, AMH, MH_Gibbs, DEMC, DREAM_ZS, InfReader
```

## 公共对象

| 对象 | 作用 |
|---|---|
| `MH` | Metropolis-Hastings。 |
| `AMH` | Adaptive Metropolis-Hastings。 |
| `MH_Gibbs` | 坐标式 Gibbs/MH 更新。 |
| `DEMC` | Differential Evolution MCMC。 |
| `DREAM_ZS` | DREAM(ZS) 风格采样器。 |
| `Chain` | 单条链的运行状态。 |
| `InfResult` | 推断结果对象。 |
| `InfHistory` | 推断历史。 |
| `InfReader` | 读取保存的推断 sqlite。 |

## 通用调用

```text
result = method.run(problem, gamma=0.2, seed=123)
```

| 参数 | 含义 |
|---|---|
| `problem` | 标量输出 `Problem`。 |
| `gamma` | proposal scale，可以是标量、向量或 `(nChains, nInput)` 矩阵。 |
| `seed` | 随机种子。 |

构造函数常用参数：

| 参数 | 含义 |
|---|---|
| `nChains` | 链数。 |
| `warmUp` | warm-up 长度。 |
| `maxIters` / `maxIterTimes` | 最大迭代次数。 |
| `logProbFunc` | 自定义对数概率函数。 |
| `verboseFlag`, `logFlag`, `saveFlag` | 运行输出和保存控制。 |

## 方法选择

| 方法 | 适合场景 |
|---|---|
| `MH` | 简单标量问题的默认起点。 |
| `AMH` | proposal scale 难手动固定时。 |
| `MH_Gibbs` | 每次更新一个变量更稳定时。 |
| `DEMC` | 多链共享差分 proposal。 |
| `DREAM_ZS` | 更复杂后验形状和 archive-based proposal。 |

`DREAM_ZS` 的提议 archive 使用内部连续坐标，独立保存初始状态、warm-up 和正式采样中接受的状态。
更新当前链不会改写已经存入的历史点。

## `InfResult`

| 字段 | 含义 |
|---|---|
| `decs` | 参数样本，shape 为 `(n_chains, draws, n_input)`。 |
| `objs` | 目标值，shape 为 `(n_chains, draws, n_output)`。 |
| `cons` | 约束值，无约束时为 `None`。 |
| `logProb` | 对数概率，shape 为 `(n_chains, draws)`。 |
| `accepted` | accepted proposal 标记。 |
| `feasibleMask` | 可行性标记。 |
| `acceptanceRate` | 每条链的接受率。 |
| `bestDecs`, `bestObjs`, `bestCons` | 最佳样本及其输出。 |
| `FEs`, `iters` | 函数评估次数和迭代数。 |
| `history` | 运行历史。 |
| `summary()` | 摘要字典。 |

## `InfReader`

| 方法 | 含义 |
|---|---|
| `list_runs(result_dir)` | 列出保存的推断结果。 |
| `get_run_summary()` | 读取摘要。 |
| `get_run_params()` | 读取参数。 |
| `list_snapshots()` | 列出快照。 |
| `load_last_snapshot_members()` | 读取最后快照的链成员。 |
| `load_result()` | 重建 `InfResult`。 |

## 下一步

| 目标 | 阅读 |
|---|---|
| 用户指南 | [Inference](../inference.md) |
| 标量问题定义 | [Problem API](problem.md) |
| 校准工作流 | [Calibration API](calibration.md) |


持久化结果带有模块标识；reader 会拒绝其他模块及没有标识的旧数据库。每次运行都有基于 UUID 的独立 ID，数据库和日志共用，即使关闭 SQLite 保存也有 ID。所有 reader 支持 `with` 和重复 `close()`。内部运行对象统一用 `state`、`params`，返回结果的正式字段不变。
