# Inference

`inference` 模块用于对标量 `Problem` 做 MCMC 风格采样。

当你关心的不是单个最优解，而是一组可能的参数值、参数不确定性、后验分布或围绕标量评分的采样结果时，使用推断模块。

核心流程：

```text
定义标量 Problem -> 选择采样方法 -> method.run(problem, gamma=..., seed=...) -> InfResult
```

## 推断需要什么

推断当前要求 `Problem` 只有一个输出。

| 项 | 含义 |
|---|---|
| `nInput` | 要采样的参数个数。 |
| `lb`, `ub` | 每个参数的下界和上界。可以是标量，也可以是向量。 |
| `objFunc` | 批量目标函数，输入矩阵 `X`，每行返回一个标量输出。 |
| `optType` | `"min"` 或 `"max"`，用于决定目标值如何转成默认 log probability。 |
| `conFunc`, `nCon` | 可选约束。约束值 `<= 0` 表示可行。 |
| `logProbFunc` | 可选自定义对数概率函数。 |

如果你有多个目标，先把它们组合成一个标量评分；如果目标本身不能合并，应该使用优化模块而不是推断模块。

## 基本工作流

这个例子对二维 sphere 问题采样：

```text
f(x) = x1^2 + x2^2
```

这是最小化问题，所以默认情况下，目标值越小，log probability 越高。

```python
import numpy as np

from UQPyL.inference import MH
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=-2.0, ub=2.0, objFunc=objFunc, optType="min", name="SpherePosterior")
method = MH(nChains=3, warmUp=5, maxIters=30, verboseFlag=False, logFlag=False, saveFlag=False)
result = method.run(problem, gamma=0.2, seed=123)

print(result.decs.shape)
print(result.objs.shape)
print(result.logProb.shape)
print(result.acceptanceRate)
print(result.bestDecs)
print(result.bestObjs)
```

Example output:

```text
(3, 30, 2)
(3, 30, 1)
(3, 30)
[0.5862 0.5517 0.5862]
[[ 0.2532 -0.222 ]]
[[0.1134]]
```

| 输出 | 含义 |
|---|---|
| `decs.shape` | `(n_chains, draws, n_input)`，即链数、每条链的样本数、参数维度。 |
| `objs.shape` | 每个样本对应的目标值。 |
| `logProb.shape` | 每个样本对应的对数概率。 |
| `acceptanceRate` | 每条链的接受率。 |
| `bestDecs` | 采样过程中评分最好的参数行。 |
| `bestObjs` | `bestDecs` 对应的原始目标值。 |

## 批量目标函数

`objFunc` 接收的是矩阵，不是单个参数向量。

```text
X.shape = (n_samples, n_input)
返回 shape = (n_samples, 1)
```

因此示例里使用：

```text
X = np.atleast_2d(X)
return np.sum(X**2, axis=1, keepdims=True)
```

`np.atleast_2d(X)` 让函数在只传入一行时也能工作。`keepdims=True` 保证输出仍是二维列。

## 目标值和 Log Probability

默认情况下，推断会把标量目标转换为：

```text
log_prob = -oriented_objective
```

| `optType` | 效果 |
|---|---|
| `"min"` | 目标越小，log probability 越高。 |
| `"max"` | 原始目标越大，log probability 越高。 |

如果你的模型已经有明确的似然、后验或能量函数，应该传入 `logProbFunc`，不要依赖默认转换。

## 自定义 Log Probability

`logProbFunc(y, decs=None, cons=None)` 必须为每个样本返回一个对数概率值。值越大，样本越容易被接受。

```python
import numpy as np

from UQPyL.inference import MH
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


def logProbFunc(y, decs=None, cons=None):
    decs = np.atleast_2d(decs)
    target = np.array([0.5, -0.25])
    return -0.5 * np.sum((decs - target) ** 2, axis=1)


problem = Problem(nInput=2, nObj=1, lb=-2.0, ub=2.0, objFunc=objFunc, optType="min", name="SpherePosterior")
method = MH(nChains=3, warmUp=5, maxIters=30, logProbFunc=logProbFunc, verboseFlag=False, logFlag=False, saveFlag=False)
result = method.run(problem, gamma=0.2, seed=123)

print(result.logProb.shape)
print(result.bestDecs)
print(result.logProb[:, -3:])
```

## 选择推断方法

| 方法 | 适合场景 |
|---|---|
| `MH` | 基础随机游走 Metropolis-Hastings。简单标量问题先用它。 |
| `AMH` | 自适应 Metropolis-Hastings。固定 proposal scale 难调时使用。 |
| `MH_Gibbs` | 坐标逐个更新。一次只改一个变量更稳定时使用。 |
| `DEMC` | 差分进化 MCMC。多链之间可以互相提供 proposal 信息。 |
| `DREAM_ZS` | DREAM(ZS) 风格采样器，用于更难的后验形状。 |

不要只因为一次短运行接受率更高就选择某个方法。接受率只是诊断信号，还要检查链是否充分探索参数空间，以及扩大预算后统计量是否稳定。

## 边界与预热

AMH、DEMC、DREAM 拒绝越界提议，保留当前状态，不调用用户模型；因此评估次数可能少于链数乘以总步数。MH/MH_Gibbs 保留各自对称提议下的反射处理。

DEMC 逐链更新。DREAM 在预热阶段调整步长和交叉概率，并保存包含重复点的参数档案；正式采样固定这些设置，只从档案选择供体。建议提供足够的 `warmUp`，零预热仍可运行，但未调整的提议可能探索较慢。AMH 仍采用随历史增长的协方差适应。

所有方法均通过 `result.diagnostics["sampler"]` 输出边界策略、更新方式、适应阶段、提议类型和提议参数；DREAM 另保留档案与冻结设置。诊断接近收敛不能替代已知分布对照；多峰或强相关模型仍需检查分布统计量和不同种子。

## 设置 Proposal Scale

`gamma` 控制 proposal 大小。

| `gamma` 形式 | 含义 |
|---|---|
| 标量，如 `0.2` | 所有链、所有变量使用同一个 proposal scale。 |
| 向量，如 `[0.1, 0.2]` | 每个变量一个 proposal scale。 |
| 矩阵，shape 为 `(nChains, nInput)` | 每条链、每个变量分别设置 proposal scale。 |

如果接受率很低，`gamma` 通常太大。如果接受率很高但链几乎不移动，`gamma` 可能太小。

## 处理约束

约束约定与优化模块一致：

```text
cons <= 0 表示可行
```

推断中不可行 proposal 会被硬拒绝。可行性保存在 `result.feasibleMask`。

```python
import numpy as np

from UQPyL.inference import MH
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


def conFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] + X[:, 1] - 0.5).reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, nCon=1, lb=-2.0, ub=2.0, objFunc=objFunc, conFunc=conFunc, optType="min", name="ConstrainedInference")
method = MH(nChains=2, warmUp=2, maxIters=10, verboseFlag=False, logFlag=False, saveFlag=False)
result = method.run(problem, gamma=0.1, seed=123)

print(result.feasibleMask.shape)
print(result.acceptanceRate)
print(result.bestFeasible)
print(result.bestCons)
```

## 读取 `InfResult`

`method.run()` 返回 `InfResult`。

| 字段 | 含义 |
|---|---|
| `decs` | 参数样本，shape 为 `(n_chains, draws, n_input)`。 |
| `objs` | 目标值，shape 为 `(n_chains, draws, n_output)`。 |
| `cons` | 约束值；无约束时为 `None`。 |
| `logProb` | 对数概率，shape 为 `(n_chains, draws)`。 |
| `accepted` | 每个样本是否来自 accepted proposal。 |
| `feasibleMask` | 每个样本是否可行。 |
| `acceptanceRate` | 每条链的接受率。 |
| `bestDecs` | 最佳采样参数行。 |
| `bestObjs` | `bestDecs` 对应的原始目标值。 |
| `FEs` | 函数评估次数。 |
| `iters` | 最终迭代数。 |
| `history` | 运行历史快照。 |

例如，丢弃前 5 个 draw 后计算简单均值：

```text
samples = result.decs[:, 5:, :].reshape(-1, result.nInput)
print(samples.mean(axis=0))
print(samples.std(axis=0))
```

真实推断任务应使用更长 warm-up 和更大的采样预算。

## 常见错误

| 错误 | 修正 |
|---|---|
| `objFunc` 返回 `(n_samples,)` | 返回 `(n_samples, 1)`，例如 `reshape(-1, 1)`。 |
| 把多目标问题传给推断 | 先合并成一个标量评分，或改用优化。 |
| `gamma` 太大 | proposal 经常被拒绝，减小 `gamma`。 |
| `gamma` 太小 | 接受率很高但链不移动，增大 `gamma`。 |
| 只看 `bestDecs` | 推断的核心是整条链，用 `result.decs` 做不确定性统计。 |
| 把约束当软惩罚 | 推断会硬拒绝不可行 proposal；软惩罚应放入目标或 `logProbFunc`。 |

## 下一步

| 目标 | 阅读 |
|---|---|
| 查构造参数和结果字段 | [Inference API](api/inference.md) |
| 定义标量目标、边界和约束 | [Problem](problem.md) |
| 对比优化和推断 | [Optimization](optimization.md) |
| 围绕参数拟合构建校准流程 | [Calibration](calibration.md) |


### 整数、离散变量与推断坐标

推断算法的内部提议保留在原 `lb/ub` 对应的连续坐标区间：连续变量仍使用物理尺度，整数/离散变量使用等宽区间表示每个合法取值。初始化先生成单位样本，再映射为内部连续坐标，避免 DOE 已解码后又重复映射。内部状态不会被取整或吸附到区间中点；这是为了保留原有连续提议的接受率计算。

实际 `Problem.evaluate()`、自定义 `logProbFunc` 的 `decs`、最终结果、打印及 SQLite 快照/结果文件均使用解码后的真实值。`initialSampling()`、`evaluate()`、`initChains()` 是算法内部接口，其决策参数按内部坐标解释；不要把对外结果的离散值直接传入这些内部接口。

等宽区间使常数目标对应各合法整数/离散取值的相同基准质量；若希望非均匀先验，应在目标或自定义 logProbFunc 中表达。正负目标方向及硬约束拒绝规则沿用既有逻辑。本修复核对了五算法的评估/记录一致性及一个 MH 已知离散分布，不代表所有算法和问题的收敛性都已得到证明。

变量边界须有限且有序。多选项离散变量需要正宽度的内部区间；固定连续维在提议和边界处理时保留固定值。提议、自适应协方差和 DREAM 历史提议档案仍使用内部连续坐标，对外不将这些内部坐标误标为实际参数。


自定义 `logProbFunc` 每行必须返回一个实数：批量形状为 `(n,)` 或 `(n,1)`，单点也可返回标量。允许 `-inf` 表示零概率；NaN、正无穷、复数或错误形状会停止运行，避免将无效链作为成功结果返回。初始化只保留约束可行且 logProb 有限的状态，最多尝试 `maxInitAttempts` 批 LHS；找不到足够初值会停止。零概率提议直接拒绝，不计算 `-inf - -inf`。回调也会用于初始化校验，同一状态可能调用多次，应返回确定的对数概率。

`DREAM_ZS(ps=1)` 在存在非固定维时默认混入 10% 的全维对称高斯随机游走，防止小档案使纯 snooker 永远困在低维空间。`snookerRefreshProb` 可设为 `(0,1]`，只在 `ps=1` 时生效；刷新标准差为各维跨度的 0.1 倍，越界直接拒绝。运行时发出 `RuntimeWarning` 并在 `diagnostics['sampler']['proposal_settings']` 保存 `full_support_refresh_probability`、`effective_snooker_probability` 和 `refresh_scale`。因此默认设置的有效 snooker 比例为 90%；常规 `ps<1` 路径不启用此刷新。

AMH 使用增量中心矩计算历史无偏协方差，保留拒绝后的重复状态、原有缩放和协方差下限。每步仅处理新追加行，额外缓存内存为每链一个均值和散布矩阵，重置时清空；完整采样历史仍保留。结果与全量重算可能有浮点舍入差异，不保证长链逐位一致。
