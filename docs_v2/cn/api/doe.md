# Design of Experiment API

LHS 的 `maximin` / `center_maximin` 在 `nSamples=1` 时发出 `RuntimeWarning`，分别使用 classic / center，元数据保留请求的 `criterion` 并以 `effective_criterion` 记录实际模式。三个优化 criterion 的 `iterations` 必须为正整数，拒绝布尔值；无效配置明确报参数错误。

Sobol 和 Saltelli 的 `skipValue` 必须为非负整数，但可超过本次基础样本数。基础样本数不是2的幂，或为2的幂但 `skipValue` 不是该样本数的整数倍时，发出一次 `UserWarning` 后继续生成，不裁剪或补齐样本。默认 `skipValue=0`；对齐配置不触发这项质量告警，不代表任意模型的积分/敏感性误差已收敛。Sobol 通过 fast_forward 跳过前缀，不再构造并保留全部被跳过的点。FFD 返回的 `levels` 元数据是独立副本。

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
| `sample(problem, nSamples=None, seed=None, nt=None, *, output="real")` | `np.ndarray` | 生成问题空间样本。 |
| `sampleWithMeta(problem, nSamples=None, seed=None, nt=None, *, output="real")` | `(np.ndarray, dict)` | 生成样本和设计元数据。 |

| 参数 | 含义 |
|---|---|
| `problem` | `ProblemBase` 协议对象，例如 `Problem`、`ModelProblem` 或内置 benchmark direct problem。 |
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

`LHS(criterion="correlation")` 在 unit 空间按输入列计算 Pearson 相关性，选择变量对中
最大绝对相关系数最小的候选，完全正相关或负相关也计入。`iterations` 必须为正整数。
一维或单样本没有可优化的相关性，直接返回第一份普通 LHS 设计，不打印候选搜索信息。
离散变量解码可能改变最终真实值的相关性。

## 分析专用设计

| 设计 | 用途 | 输出行数 |
|---|---|---|
| `SaltelliDesign(secondOrder=False)` | `analysis.Sobol` | `((D + 2) * N, D)` |
| `SaltelliDesign(secondOrder=True)` | 带二阶指标的 `analysis.Sobol` | `((2 * D + 2) * N, D)` |
| `FASTDesign(M=4)` | `analysis.FAST` | `(N * D, D)`，且 `N > 4 * M^2` |
| `MorrisDesign(numLevels=4)` | `analysis.Morris` | `(numTrajectory * (D + 1), D)` |

FAST 的 `N > 4 * M^2` 只是允许采样的下限。要为 `D` 个变量分配互不重复的辅助频率，还需满足 `N >= 4 * M^2 * (D - 1) + 1`。例如 `M=4、D=3` 时至少取 `N=129`，总样本数为 387。较小但合法的配置仍会提示频率复用风险；满足此条件只代表避免辅助频率复用，不保证敏感性估计已经收敛。

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


## 选择样本输出空间

所有内置采样器的 `sample()`、`sampleWithMeta()` 支持关键字参数 `output="real"`（默认）或 `output="unit"`。前者输出真实值，可直接评估；后者直接返回生成的 `[0,1]` 样本，用于优化种群等内部流程。相同采样器配置、样本数及 seed 下，两种输出来自相同的单位样本；元数据的 `output` 字段记录返回空间。

```python
U = LHS().sample(problem, 20, seed=123, output="unit")
X = LHS().sample(problem, 20, seed=123)  # 等于 problem.unit_to_space(U)
```

整数与离散变量使用等宽区间解码；离散真实值取自 `varSet`，不一定处于该列 `lb/ub` 的范围。独立 DOE/分析调用继续默认返回真实值。
