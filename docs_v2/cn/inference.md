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
