# Design of Experiment API

## `UQPyL.doe`

`doe` 模块从 `Problem` 或 `ModelProblem` 的输入空间生成样本。采样器通常先在 `[0, 1]` 单位空间生成点，再通过 `problem.unit_to_space()` 映射到真实边界。

## 导入

```python
from UQPyL.doe import LHS, Random, FFD, Sobol
```

## 公共对象

| 对象 | 作用 |
|---|---|
| `Sampler` | 大多数采样器的基础接口。 |
| `Random` | 均匀随机采样。 |
| `LHS` | Latin hypercube sampling。 |
| `FFD` | Full factorial design。 |
| `Sobol` | Sobol 低差异序列采样。 |
| `SaltelliDesign` | Sobol 敏感性分析专用设计。 |
| `FASTDesign` | FAST 敏感性分析专用设计。 |
| `MorrisDesign` | Morris 轨迹设计。 |

## 通用接口

| 方法 | 返回 | 含义 |
|---|---|---|
| `sample(problem, nSamples=None, seed=None, nt=None)` | `np.ndarray` | 生成问题空间样本。 |
| `sampleWithMeta(problem, nSamples=None, seed=None, nt=None)` | `(np.ndarray, dict)` | 生成样本和设计元数据。 |

| 参数 | 含义 |
|---|---|
| `problem` | `Problem`、`ModelProblem` 或 benchmark problem。 |
| `nSamples` | 样本数或基础样本量，具体含义取决于采样器。 |
| `seed` | 可选随机种子。 |
| `nt` | `nSamples` 的旧别名。 |

## 常用采样器

| 采样器 | 构造 | 说明 |
|---|---|---|
| `Random` | `Random()` | 随机均匀采样。 |
| `LHS` | `LHS(criterion="classic", iterations=5)` | Latin hypercube；`criterion` 可为 `"classic"`、`"center"`、`"maximin"` 等。 |
| `FFD` | `FFD()` | 全因子网格，使用 `levels` 而不是 `nSamples`。 |
| `Sobol` | `Sobol(scramble=True, skipValue=0)` | Sobol 低差异序列；`nSamples` 通常建议为 2 的幂。 |

## 分析专用设计

| 设计 | 用途 | 输出行数 |
|---|---|---|
| `SaltelliDesign(secondOrder=False)` | `analysis.Sobol` | `((D + 2) * N, D)` |
| `SaltelliDesign(secondOrder=True)` | 带二阶指标的 `analysis.Sobol` | `((2 * D + 2) * N, D)` |
| `FASTDesign(M=4)` | `analysis.FAST` | `(N * D, D)`，且 `N > 4 * M^2` |
| `MorrisDesign(numLevels=4)` | `analysis.Morris` | `(numTrajectory * (D + 1), D)` |

这些设计必须用 `sampleWithMeta()`，并把返回的 `meta` 传给对应分析方法。

## 元数据

`meta` 是描述设计结构的字典。普通采样中它用于记录；指定设计分析中它是必需输入。

| 采样器 | 关键字段 |
|---|---|
| `Random` | `designType`, `seed` |
| `LHS` | `designType`, `criterion`, `iterations`, `seed` |
| `FFD` | `designType`, `levels`, `seed` |
| `Sobol` | `designType`, `scramble`, `skipValue`, `seed` |
| `SaltelliDesign` | `designType`, `N`, `secondOrder`, `blockSize`, `seed` |
| `FASTDesign` | `designType`, `N`, `M`, `blockSize`, `seed` |
| `MorrisDesign` | `designType`, `numTrajectory`, `numLevels`, `trajectorySize`, `seed` |

## 下一步

| 目标 | 阅读 |
|---|---|
| 用户指南 | [Design of Experiment](../doe.md) |
| 敏感性分析 | [Analysis API](analysis.md) |
| 建模协议 | [Problem API](problem.md) |
