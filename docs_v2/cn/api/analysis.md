# Analysis API

## `UQPyL.analysis`

`analysis` 模块评估输入变量如何影响目标或约束输出。

## 导入

```python
from UQPyL.analysis import RBDFAST, Sobol, Morris
from UQPyL.analysis.runtime import AnaReader
```

## 公共对象

| 对象 | 作用 |
|---|---|
| `Sobol` | 基于 Saltelli 样本的方差分解敏感性分析。 |
| `FAST` | 基于 FAST 设计的 Fourier amplitude sensitivity test。 |
| `RBDFAST` | 可用于普通样本矩阵的一阶敏感性分析。 |
| `Morris` | 基于 elementary effects 的筛选方法。 |
| `RSA` | Regional sensitivity analysis。 |
| `DeltaTest` | 基于近邻的变量敏感性分析。 |
| `MARS` | MARS-based 分析；可选依赖不可用时可能为 `None`。 |
| `AnaResult` | `analyze()` 返回的标准结果对象。 |
| `AnaMetric` | `AnaResult` 中的一张指标矩阵。 |
| `AnaReader` | 读取 `saveFlag=True` 保存的 sqlite 结果。 |

## 通用调用

```text
result = method.analyze(
    problem,
    X,
    Y=None,
    meta=None,
    target="objs",
    index="all",
)
```

| 参数 | 含义 |
|---|---|
| `problem` | `ProblemBase` 实例。 |
| `X` | 输入样本矩阵。 |
| `Y` | 与 `X` 对应的输出矩阵；不传时由方法内部评估。 |
| `meta` | `sampleWithMeta()` 返回的采样元数据。部分方法必需。 |
| `target` | 分析输出块，通常为 `"objs"` 或 `"cons"`。 |
| `index` | 输出列选择：`"all"`、整数或整数列表。 |

Sobol、FAST 和 RBDFAST 在计算方差或频谱功率前，对有限输出做内部中心化和缩放。
改变输出单位（包括乘以负数）后，敏感度在浮点精度范围内保持一致。各输出列独立处理，
FAST 按各轨迹块处理；结果中的原始 `Y` 保留原值。真正恒定的输出沿用返回零指标的约定，
NaN 或无穷值会抛出 `ValueError`。输入数值舍入时已经丢失的变化无法通过内部缩放恢复。

运行控制参数：

| 参数 | 含义 |
|---|---|
| `verboseFlag` | 打印简洁运行摘要。 |
| `logFlag` | 写文本日志。 |
| `saveFlag` | 保存 sqlite 结果。 |

## 方法和设计匹配

| 方法 | 需要的样本设计 | 主要指标 |
|---|---|---|
| `Sobol` | `SaltelliDesign.sampleWithMeta()` | `S1`, `S1_norm`, `ST`, `ST_norm`, 可选 `S2` |
| `FAST` | `FASTDesign.sampleWithMeta()` | `S1`, `S1_norm`, `ST`, `ST_norm` |
| `Morris` | `MorrisDesign.sampleWithMeta()` | `mu`, `mu_star`, `sigma`, `S1_norm` |
| `RBDFAST` | 普通样本矩阵即可 | `S1` |
| `RSA` | 普通样本矩阵即可 | `S1`, `S1_norm` |
| `DeltaTest` | 普通样本矩阵即可 | `S1`, `S1_norm` |
| `MARS` | 普通训练式样本 | `S1`, `S1_norm` |

## `AnaResult`

| 字段或方法 | 含义 |
|---|---|
| `method` | 分析方法名。 |
| `problemName` | 问题名。 |
| `target` | 被分析的输出块。 |
| `settings` | 方法设置。 |
| `meta` | 采样元数据。 |
| `metrics` | `AnaMetric` 列表。 |
| `X`, `Y` | 记录的输入和输出矩阵。 |
| `runtime` | 运行时间。 |
| `metricNames` | 指标名列表。 |
| `getMetric(name)` / `result[name]` | 读取指定指标。 |
| `summary()` | 紧凑摘要。 |
| `toDict()` | 可序列化字典。 |

## `AnaMetric`

| 字段 | 含义 |
|---|---|
| `name` | 指标名，如 `S1`、`ST`、`mu_star`。 |
| `values` | 指标矩阵。行是输出，列是变量或变量组合。 |
| `rowLabels` | 输出标签。 |
| `colLabels` | 输入标签或输入组合标签。 |
| `colDim` | 列维度类型，如 `decsDim1` 或 `decsDim2`。 |

## `AnaReader`

用于读取 analysis sqlite 结果。

| 方法 | 含义 |
|---|---|
| `list_runs(result_dir)` | 列出结果目录中的 analysis runs。 |
| `get_run_summary()` | 读取运行摘要。 |
| `get_run_params()` | 读取运行参数。 |
| `get_metrics()` / `get_metric(name)` | 读取指标。 |
| `get_artifacts()` | 读取 `X`、`Y`、`settings`、`meta` 等 artifact。 |
| `load_problem()` | 读取保存的问题对象。 |
| `load_result()` | 重建完整 `AnaResult`。 |

## 下一步

| 目标 | 阅读 |
|---|---|
| 用户指南 | [Analysis](../analysis.md) |
| 生成兼容样本 | [DOE API](doe.md) |
| 建模协议 | [Problem API](problem.md) |


内置分析方法先将一维外部 Y 整理为列，再选择输出；统一检查 X/Y 行数及输入维度，保留所选的 problem 输出标签。导入请使用 `UQPyL.analysis` 公共入口或 `UQPyL.analysis.methods`，旧外层转发模块已删除。运行态为 `method.state`，运行参数为 `method.params`。


持久化结果带有模块标识；reader 会拒绝其他模块及没有标识的旧数据库。每次运行都有基于 UUID 的独立 ID，数据库和日志共用，即使关闭 SQLite 保存也有 ID。所有 reader 支持 `with` 和重复 `close()`。内部运行对象统一用 `state`、`params`，返回结果的正式字段不变。
