# 试验设计

`doe` 模块用于从 `Problem` 或 `ModelProblem` 的输入空间生成样本点。

当你需要为敏感性分析、代理模型训练、参数校准、绘图或优化算法初始化准备模型评估点时，就会用到 DOE。

## 采样到底在做什么

采样器通常先在单位空间生成点，再根据问题中定义的边界把这些点映射到真实变量范围。

```text
unit samples in [0, 1] -> problem.unit_to_space(...) -> samples in [lb, ub]
```

也就是说，DOE 返回给你的 `X` 已经是可以直接送入模型的真实输入值，不需要再手动乘以范围或加上下界。

例如：

| 变量 | 下界 `lb` | 上界 `ub` | 返回样本中的含义 |
|---|---:|---:|---|
| `x1` | `0` | `10` | 第一列会落在 `[0, 10]` |
| `x2` | `-1` | `1` | 第二列会落在 `[-1, 1]` |

`lb` 和 `ub` 可以像 `Problem` 页面里一样给标量，也可以给向量：

```text
Problem(nInput=3, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)
Problem(nInput=3, nObj=1, lb=[0.0, -1.0, 10.0], ub=[1.0, 1.0, 20.0], objFunc=objFunc)
```

第一种写法表示所有变量都用同一个边界；第二种写法表示每个变量有自己的边界。

## 基本流程

最常见的流程是：

1. 定义 `Problem`。
2. 选择一个采样器。
3. 用采样器生成 `X`。
4. 调用 `problem.evaluate(X)` 得到模型输出。

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=[0.0, -1.0], ub=[10.0, 1.0], objFunc=objFunc, name="BoundedSphere")

X = LHS("classic").sample(problem, nSamples=5, seed=123)
Y = problem.evaluate(X).objs

print(X)
print(Y.shape)
```

Example output:

```text
[[ 7.84669     0.1248378 ]
 [ 4.3518118   0.95595708]
 [ 1.36470373 -0.97847159]
 [ 2.44071975 -0.52625128]
 [ 9.63950912  0.31062976]]
(5, 1)
```

这里要特别注意两个形状：

| 对象 | 形状 | 含义 |
|---|---|---|
| `X` | `(n_samples, n_input)` | 每一行是一个样本点，每一列是一个输入变量 |
| `Y` | `(n_samples, n_obj)` | 每一行对应一个样本点的输出，每一列是一个目标 |

如果你对“批量评估”还不熟，可以把 `X` 理解成一张表：每一行是一组参数。`problem.evaluate(X)` 会把这张表里的多组参数一次性送进评估函数，返回同样行数的结果。

## 选择采样器

建议从任务出发，而不是从类名出发。

| 任务 | 推荐采样器 | 原因 |
|---|---|---|
| 普通模型探索 | `LHS("classic")` | 比完全随机采样更均匀地覆盖每个变量范围。 |
| 快速检查代码是否能跑通 | `Random()` | 简单、便宜，适合 smoke test。 |
| 少量变量上做规则网格 | `FFD()` | 确定性的全因子网格，适合低维、小水平数问题。 |
| 低差异序列采样 | `Sobol()` | 适合需要结构化空间填充样本的场景。 |
| Sobol 敏感性分析 | `SaltelliDesign()` | 生成 `Sobol` 分析所需的专用设计和元信息。 |
| FAST 敏感性分析 | `FASTDesign()` | 生成 `FAST` 分析所需的专用设计和元信息。 |
| Morris 筛选 | `MorrisDesign()` | 生成 Morris 轨迹和元信息。 |

实用默认选择：如果下游方法没有明确要求专用设计，优先用 `LHS("classic")`。

## 样本量怎么选

样本量没有统一答案，取决于输入维度、模型运行成本和后续分析方法。

| 目标 | 起步建议 |
|---|---|
| 检查流程能否跑通 | `nSamples = 5` 到 `20` |
| 绘图或粗略探索 | `nSamples = 20` 到 `100` |
| 训练代理模型 | 从 `10 * nInput` 左右开始，再用独立验证集检查精度 |
| RBDFAST / RSA / DeltaTest | 根据 `nInput`，从 `100` 到 `500` 左右开始 |
| Sobol 分析 | 使用 `SaltelliDesign`，基础样本量 `N` 可从 `128`、`256` 或 `512` 开始 |
| FAST 分析 | 使用 `FASTDesign`，并满足 `N > 4 * M^2` |
| Morris 筛选 | 使用 `numTrajectory`，初次分析常从 `10` 到 `50` 条轨迹开始 |

如果模型很贵，先用小样本跑通完整流程，再逐步增加样本量。这样更容易尽早发现边界、维度、输出形状或下游接口的问题。

## 可复现实验

需要复现实验结果时，传入 `seed`。

```python
import numpy as np

from UQPyL.doe import Random
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

X1 = Random().sample(problem, nSamples=4, seed=42)
X2 = Random().sample(problem, nSamples=4, seed=42)

print(X1)
print((X1 == X2).all())
```

Example output:

```text
[[0.77395605 0.43887844]
 [0.85859792 0.69736803]
 [0.09417735 0.97562235]
 [0.7611397  0.78606431]]
True
```

同一个采样器、同一个问题、同一个 `seed` 会生成相同样本。想要另一组随机设计时，换一个 `seed` 即可。

## `sample()` 和 `sampleWithMeta()`

如果你只需要样本矩阵，用 `sample()`。

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

X = LHS("classic").sample(problem, nSamples=6, seed=123)
print(X.shape)
```

如果下游方法需要采样设计的元信息，用 `sampleWithMeta()`。

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

X, meta = LHS("classic").sampleWithMeta(problem, nSamples=6, seed=123)

print(X.shape)
print(meta)
```

Example output:

```text
(6, 2)
{'designType': 'lhs', 'criterion': 'classic', 'iterations': 5, 'seed': 123}
```

`meta` 记录的是样本如何生成，不是模型输出。普通采样时它主要用于记录和复现；Sobol、FAST、Morris 这类方法特定的设计中，它是分析方法需要读取的必要信息。

## `meta` 应该怎么用

`meta` 是一个小字典，描述采样设计。它告诉下游方法应该如何理解 `X` 的行排列、样本块结构和设计参数。

普通采样中，`meta` 常用于检查和记录：

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

X, meta = LHS("classic").sampleWithMeta(problem, nSamples=6, seed=123)

print(meta["designType"])
print(meta["criterion"])
print(meta["seed"])
```

Example output:

```text
lhs
classic
123
```

方法特定的敏感性分析中，`meta` 是必须保留的输入。例如 `Sobol` 需要知道 `X` 是否来自 `SaltelliDesign`、是否生成了二阶样本、基础样本量是多少。

```python
import numpy as np

from UQPyL.analysis import Sobol
from UQPyL.doe import SaltelliDesign
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] + X[:, 1]).reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

X, meta = SaltelliDesign(secondOrder=False).sampleWithMeta(problem, N=64, seed=123)
Y = problem.evaluate(X).objs

result = Sobol(verboseFlag=False).analyze(problem, X, Y=Y, meta=meta)

print(meta)
print(result.metricNames)
```

Example output:

```text
{'designType': 'saltelli', 'N': 64, 'secondOrder': False, 'skipValue': 0, 'scramble': True, 'blockSize': 4, 'seed': 123}
['S1', 'S1_norm', 'ST', 'ST_norm']
```

可以用下面这张表判断是否需要 `meta`：

| 场景 | 是否需要 `meta` |
|---|---|
| 只需要 `X` 去做模型评估、优化初始化或代理模型训练 | 可选 |
| 需要记录采样设计，方便复现实验 | 建议保留 |
| 后续要运行 `Sobol`、`FAST` 或 `Morris` 分析 | 必须保留 |
| 不确定下游方法是否需要元信息 | 用 `sampleWithMeta()`，把 `meta` 一起保存 |

## 全因子设计

`FFD` 会生成规则网格。它使用 `levels`，不是 `nSamples`。

```python
import numpy as np

from UQPyL.doe import FFD
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=[0.0, -1.0], ub=[1.0, 1.0], objFunc=objFunc)

X, meta = FFD().sampleWithMeta(problem, levels=[3, 4])

print(X.shape)
print(X[:5])
print(meta["levels"])
```

Example output:

```text
(12, 2)
[[ 0.         -1.        ]
 [ 0.         -0.33333333]
 [ 0.          0.33333333]
 [ 0.          1.        ]
 [ 0.5        -1.        ]]
[3, 4]
```

两个变量分别给 `[3, 4]` 个水平时，总样本数是 `3 * 4 = 12`。全因子设计在维度升高时会增长得很快，例如 6 个变量、每个变量 5 个水平，总行数就是 `5^6 = 15625`。

## 敏感性分析专用设计

有些敏感性分析方法要求特定采样结构。此时要使用对应的 DOE 类，并保留 `meta`。

### Sobol

```python
import numpy as np

from UQPyL.analysis import Sobol
from UQPyL.doe import SaltelliDesign
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] + 0.1 * X[:, 1]
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

X, meta = SaltelliDesign(secondOrder=False).sampleWithMeta(problem, N=64, seed=123)
Y = problem.evaluate(X).objs
result = Sobol(verboseFlag=False).analyze(problem, X, Y=Y, meta=meta)

print(X.shape)
print(meta["designType"])
print(result.metricNames)
```

Example output:

```text
(256, 2)
saltelli
['S1', 'S1_norm', 'ST', 'ST_norm']
```

对 `D` 个输入变量和基础样本量 `N`，`SaltelliDesign(secondOrder=False)` 会返回 `(D + 2) * N` 行。所以这里 `D=2`、`N=64`，总行数是 `256`，不是 `64`。

### FAST

```python
import numpy as np

from UQPyL.doe import FASTDesign
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X, axis=1, keepdims=True)


problem = Problem(nInput=3, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

X, meta = FASTDesign(M=4).sampleWithMeta(problem, N=256, seed=123)

print(X.shape)
print(meta["designType"])
print(meta["M"])
```

Example output:

```text
(768, 3)
fast
4
```

FAST 中 `N` 必须满足 `N > 4 * M^2`。当 `M=4` 时，`N` 需要大于 `64`。

### Morris

```python
import numpy as np

from UQPyL.doe import MorrisDesign
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X, axis=1, keepdims=True)


problem = Problem(nInput=3, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

X, meta = MorrisDesign(numLevels=4).sampleWithMeta(problem, numTrajectory=5, seed=123)

print(X.shape)
print(meta["designType"])
print(meta["trajectorySize"])
```

Example output:

```text
(20, 3)
morris
4
```

对 `D` 个输入变量和 `numTrajectory` 条轨迹，Morris 会返回 `numTrajectory * (D + 1)` 行。所以这里是 `5 * (3 + 1) = 20` 行。

## 方法匹配

| 下游方法 | 应该使用的设计 | 是否保留 `meta` |
|---|---|---|
| `RBDFAST` | `LHS`、`Random` 或其他普通样本矩阵 | 可选 |
| `RSA` | `LHS`、`Random` 或其他普通样本矩阵 | 可选 |
| `DeltaTest` | `LHS`、`Random` 或其他普通样本矩阵 | 可选 |
| `Sobol` | `SaltelliDesign` | 必须 |
| `FAST` | `FASTDesign` | 必须 |
| `Morris` | `MorrisDesign` | 必须 |

如果分析方法需要 `meta`，就用 `sampleWithMeta()` 生成样本，并把 `X` 和 `meta` 一起传给分析方法。

## 常见错误

| 错误 | 为什么有问题 | 修正方式 |
|---|---|---|
| 用 `sample()` 生成 Sobol/FAST/Morris 分析样本 | 这些分析方法需要设计元信息。 | 用 `sampleWithMeta()`，并把 `meta` 传给分析方法。 |
| 把 Saltelli 的 `N` 当成最终样本行数 | Saltelli 会把基础样本量 `N` 扩展成更多行。 | 采样后查看 `X.shape`。 |
| 在高维问题中使用 `FFD` | 网格大小等于各变量水平数的乘积，增长很快。 | 高维探索优先使用 `LHS` 或 `Sobol`。 |
| 忘记设置 `seed` | 每次运行样本不同，文档示例、测试和复现实验会变得困难。 | 需要复现时传入 `seed`。 |
| 没检查 `lb` 和 `ub` 就开始采样 | 错误边界会直接产生错误样本。 | 打印 `problem.lb`、`problem.ub` 和前几行样本检查。 |
| 把 `meta` 当成模型输出 | `meta` 只是采样设计说明，不是 `Y`。 | 模型输出来自 `problem.evaluate(X).objs` 或相应返回字段。 |

## 下一步

| 目标 | 阅读 |
|---|---|
| 查看采样器构造参数和元信息字段 | [DOE API](api/doe.md) |
| 定义输入边界和变量类型 | [Problem](problem.md) |
| 用样本做敏感性分析 | [Analysis](analysis.md) |
| 用样本训练代理模型 | [Surrogate Modeling](surrogate.md) |
