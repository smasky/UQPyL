# Optimization API

## `UQPyL.optimization`

`optimization` 模块运行单目标、多目标和昂贵模型优化算法。

## 导入

```python
from UQPyL.optimization.soea import GA, PSO, DE
from UQPyL.optimization.moea import NSGAII
from UQPyL.optimization.runtime import OptReader
```

## 公共对象

| 类别 | 对象 |
|---|---|
| 单目标算法 | `GA`, `PSO`, `DE`, `SCE_UA`, `ML_SCE_UA`, `CSA`, `ABC` |
| 多目标算法 | `NSGAII`, `NSGAIII`, `MOEAD`, `RVEA` |
| 昂贵优化算法 | `EGO`, `ASMO`, `MOASMO` |
| 结果对象 | `OptResult`, `OptHistory` |
| 种群对象 | `Population` |
| 保存结果 reader | `OptReader` |

## 通用调用

```text
result = algorithm.run(problem, seed=123)
```

构造函数通常接受：

| 参数 | 含义 |
|---|---|
| `nPop` | 种群规模。 |
| `maxFEs` | 最大函数评估次数。 |
| `maxIters` | 最大迭代次数。 |
| `verboseFlag`, `verboseFreq` | 终端输出控制。 |
| `logFlag` | 是否写文本日志。 |
| `saveFlag`, `saveFreq` | 是否保存 sqlite 和快照间隔。 |

## 算法选择

| 任务 | 推荐起点 |
|---|---|
| 普通单目标连续优化 | `GA`, `PSO`, `DE` |
| 水文/参数校准类全局搜索 | `SCE_UA`, `ML_SCE_UA` |
| 多目标 Pareto 搜索 | `NSGAII` |
| 参考方向多目标优化 | `NSGAIII`, `RVEA`, `MOEAD` |
| 昂贵单目标优化 | `EGO`, `ASMO` |
| 昂贵多目标优化 | `MOASMO` |

## `OptResult`

| 字段 | 含义 |
|---|---|
| `bestDecs` | 最佳决策行；多目标时通常是 Pareto 决策矩阵。 |
| `bestObjs` | `bestDecs` 对应目标值。 |
| `bestCons` | 对应约束值，无约束时为 `None`。 |
| `bestFeasible` | 最佳解是否可行。 |
| `bestMetric` | 多目标或特定算法的进展指标。 |
| `FEs` | 函数评估次数。 |
| `iters` | 迭代数。 |
| `history` | 运行历史。 |
| `summary()` | 摘要字典。 |

## `Population`

`Population` 存储优化中的成员。

| 字段 | 含义 |
|---|---|
| `decs` | 决策矩阵。 |
| `objs` | 目标矩阵。 |
| `cons` | 约束矩阵。 |
| `cv` | 约束违反度。 |
| `feasible` | 可行性标记。 |

## `OptReader`

用于读取 `saveFlag=True` 保存的优化 sqlite。

| 方法 | 含义 |
|---|---|
| `list_runs(result_dir)` | 列出优化结果。 |
| `get_run_summary()` | 读取运行摘要。 |
| `get_run_params()` | 读取运行参数。 |
| `list_snapshots()` | 列出保存的快照。 |
| `load_last_snapshot_members()` | 读取最后一个快照中的成员。 |
| `load_result()` | 重建 `OptResult`。 |

## 下一步

| 目标 | 阅读 |
|---|---|
| 用户指南 | [Optimization](../optimization.md) |
| 定义优化问题 | [Problem API](problem.md) |
| 代理辅助优化 | [Surrogate API](surrogate.md) |
