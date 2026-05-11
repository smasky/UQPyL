# Problem

`problem` 模块是 UQPyL 的建模核心。采样、分析、优化、推断、校准和代理建模等工作流，都是从一个 `Problem` 类对象开始的。

如果你想把自己的数学模型、仿真模型或目标函数接入 UQPyL，这一页就是入口。

## 你需要先定义什么

大多数用户只需要先回答四个问题：

| 问题 | 对应到哪里 |
|---|---|
| 输入变量有哪些？ | `nInput`、`lb`、`ub`，可选 `xLabels` |
| 一次模型评估会计算什么？ | `objFunc`、`conFunc`、`simFunc` 或 `evaluate` |
| 目标是最小化还是最大化？ | `optType` |
| 这是普通目标问题，还是带观测的仿真问题？ | `Problem` 或 `ModelProblem` |

普通目标和约束问题用 `Problem`。如果主要输出是时间序列仿真或多序列仿真，尤其是校准场景，用 `ModelProblem`。

## 从一个简单的 `Problem` 开始

这个例子定义了一个双变量目标：

```text
f(x) = x1^2 + x2^2
```

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, ub=1.0, lb=-1.0, objFunc=objFunc, optType="min", name="Sphere2D")

res = problem.evaluate([[0.2, 0.3]])
print(res.objs)
```

Example output:

```text
[[0.13]]
```

最重要的契约是：

```text
input X -> objFunc(X) -> objective values
```

## 定义输入空间

`nInput` 表示输入变量个数，`lb` 和 `ub` 分别是下界和上界。

### 标量边界

如果所有变量都共享同一个范围，可以直接给标量。

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=3, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc)

print(problem.lb)
print(problem.ub)
```

Example output:

```text
[[0. 0. 0.]]
[[1. 1. 1.]]
```

这里三个变量都在 `[0.0, 1.0]` 内。

### 按变量分别设边界

如果每个变量范围不同，就传列表或 NumPy 数组。

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(
    nInput=3,
    nObj=1,
    lb=[0.0, -5.0, 50.0],
    ub=[1.0, 10.0, 100.0],
    objFunc=objFunc,
    xLabels=["width", "slope", "storage"],
)

print(problem.lb)
print(problem.ub)
print(problem.xLabels)
```

Example output:

```text
[[ 0. -5. 50.]]
[[  1.  10. 100.]]
['width', 'slope', 'storage']
```

可以把它理解成：

| 变量 | 下界 | 上界 |
|---|---:|---:|
| `width` | `0.0` | `1.0` |
| `slope` | `-5.0` | `10.0` |
| `storage` | `50.0` | `100.0` |

## 理解批量评估

UQPyL 通常一次会评估很多候选点，所以函数一般都要接受表格形式的 `X`。

| 形状 | 含义 |
|---|---|
| `(n_samples, n_input)` | 一批输入样本 |
| `X` 的一行 | 一个候选输入向量 |
| `X` 的一列 | 一个输入变量 |

例如：

```python
import numpy as np


single_x = np.array([0.2, 0.3])
batch_x = np.array([
    [0.2, 0.3],
    [0.5, 0.1],
    [0.0, 1.0],
])

print(np.atleast_2d(single_x).shape)
print(np.atleast_2d(batch_x).shape)
```

Example output:

```text
(1, 2)
(3, 2)
```

`np.atleast_2d(X)` 很有用，因为无论你传单个样本还是一批样本，都会变成统一的二维表。

做完 `X = np.atleast_2d(X)` 之后：

| 表达式 | 含义 |
|---|---|
| `X[:, 0]` | 所有样本的第一个变量 |
| `X[:, 1]` | 所有样本的第二个变量 |
| `X.shape[0]` | 样本数 |
| `reshape(-1, 1)` | 转成一列输出 |
| `keepdims=True` | 让 NumPy 规约结果保持列形状 |

## 编写目标函数

### 单目标

单目标时，返回形状应为 `(n_samples, 1)`。

```python
import numpy as np


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + X[:, 1] ** 2
    return y.reshape(-1, 1)


print(objFunc([0.2, 0.3]))
print(objFunc([[0.2, 0.3], [0.5, 0.1]]))
```

Example output:

```text
[[0.13]]
[[0.13]
 [0.26]]
```

### 多目标

多目标时，每个目标占一列。

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    f1 = np.sum(X**2, axis=1)
    f2 = np.sum((X - 0.5) ** 2, axis=1)
    return np.vstack([f1, f2]).T


problem = Problem(nInput=2, nObj=2, ub=1.0, lb=0.0, objFunc=objFunc, optType=["min", "min"])

print(problem.evaluate([[0.2, 0.3], [0.5, 0.1]]).objs)
```

Example output:

```text
[[0.13 0.13]
 [0.26 0.16]]
```

第一行输出属于第一行输入，第二行输出属于第二行输入。

## 编写约束函数

如果问题有约束，用 `conFunc`。约束满足的判定规则是：

```text
cons <= 0
```

这个例子表示 `x1 + x2 <= 1.0`。

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


def conFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] + X[:, 1] - 1.0).reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, nCon=1, ub=1.0, lb=0.0, objFunc=objFunc, conFunc=conFunc)

res = problem.evaluate([[0.2, 0.3], [0.8, 0.4]])
print(res.objs)
print(res.cons)
```

Example output:

```text
[[0.13]
 [0.8 ]]
[[-0.5]
 [ 0.2]]
```

第一行是可行的，因为 `-0.5 <= 0`；第二行不可行，因为 `0.2 > 0`。

如果有两个约束，就返回两列：

```python
def conFunc(X):
    X = np.atleast_2d(X)
    c1 = X[:, 0] + X[:, 1] - 1.0
    c2 = 0.2 - X[:, 0]
    return np.vstack([c1, c2]).T
```

这里 `c1 <= 0` 表示 `x1 + x2 <= 1.0`，`c2 <= 0` 表示 `x1 >= 0.2`。

## 检查返回形状

UQPyL 期望这些函数返回如下形状：

| 函数 | 输入形状 | 返回形状 |
|---|---|---|
| 单目标 `objFunc(X)` | `(n_samples, n_input)` | `(n_samples, 1)` |
| 双目标 `objFunc(X)` | `(n_samples, n_input)` | `(n_samples, 2)` |
| 单约束 `conFunc(X)` | `(n_samples, n_input)` | `(n_samples, 1)` |
| 仿真模型 `simFunc(X)` | `(n_samples, n_input)` | `(n_samples, n_time, n_series)` |

常见形状错误：

| 错误 | 为什么会出问题 | 修法 |
|---|---|---|
| 返回 `0.13` | 这是标量，不是按样本逐行返回。 | 返回 `[[0.13]]`。 |
| 返回 `(n_samples,)` | 这是扁平向量，不是列矩阵。 | 用 `reshape(-1, 1)` 或 `keepdims=True`。 |
| 把 `X[0]` 当成第一个变量 | `X[0]` 是第一行，不是第一列。 | 用 `X[:, 0]`。 |
| 忘记 `np.atleast_2d(X)` | 单样本和批量输入行为会不一致。 | 一开始就写 `X = np.atleast_2d(X)`。 |

## 使用 `evaluate()`

`problem.evaluate(X)` 返回的是 `Eval` 对象。

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


def conFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] + X[:, 1] - 1.0).reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, nCon=1, ub=1.0, lb=0.0, objFunc=objFunc, conFunc=conFunc)
res = problem.evaluate([[0.2, 0.3], [0.8, 0.4]])

print(res.objs)
print(res.cons)
```

| 字段 | 含义 |
|---|---|
| `objs` | 目标值；如果没请求则为 `None`。 |
| `cons` | 约束值；如果没有或未请求则为 `None`。 |
| `sim` | `ModelProblem` 的仿真输出；普通 `Problem` 一般为 `None`。 |

可以用 `target` 只请求其中一个输出块。

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


def conFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] + X[:, 1] - 1.0).reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, nCon=1, ub=1.0, lb=0.0, objFunc=objFunc, conFunc=conFunc)
obj_res = problem.evaluate([[0.2, 0.3]], target="objs")
con_res = problem.evaluate([[0.2, 0.3]], target="cons")

print(obj_res.objs)
print(obj_res.cons)
print(con_res.objs)
print(con_res.cons)
```

Example output:

```text
[[0.13]]
None
None
[[-0.5]]
```

## 什么时候用组合式 `evaluate`

如果目标和约束共享昂贵的中间计算，建议直接写 `evaluate`。

```python
import numpy as np

from UQPyL.problem import Eval, Problem


def evaluate(X):
    X = np.atleast_2d(X)
    total = np.sum(X, axis=1, keepdims=True)
    return Eval(objs=total**2, cons=total - 1.0)


problem = Problem(nInput=2, nObj=1, nCon=1, ub=1.0, lb=0.0, evaluate=evaluate)

res = problem.evaluate([[0.2, 0.3], [0.8, 0.4]])
print(res.objs)
print(res.cons)
```

Example output:

```text
[[0.25]
 [1.44]]
[[-0.5]
 [ 0.2]]
```

`Problem` 只接受一种 callable 配置：

| 配置方式 | 适用场景 |
|---|---|
| `objFunc` | 只有目标，没有约束。 |
| `objFunc` + `conFunc` | 有目标，也有约束。 |
| `evaluate` | 目标和约束要一起计算。 |

不要把 `evaluate` 和 `objFunc` / `conFunc` 混用。

## 使用单样本函数

如果你觉得写 batched 函数不自然，可以写单样本函数，然后用 `singleFunc` 包装。

```python
import numpy as np

from UQPyL.problem import Problem, singleFunc


@singleFunc
def objFunc(x):
    return float(np.sum(x**2))


problem = Problem(nInput=2, nObj=1, ub=1.0, lb=-1.0, objFunc=objFunc)

print(problem.evaluate([[0.2, 0.3], [0.5, 0.1]]).objs)
```

Example output:

```text
[[0.13]
 [0.26]]
```

如果是目标和约束一起返回，则用 `singleEval`。

```python
import numpy as np

from UQPyL.problem import Eval, Problem, singleEval


@singleEval
def evaluate(x):
    return Eval(objs=float(np.sum(x**2)), cons=np.array([x[0] + x[1] - 1.0]))


problem = Problem(nInput=2, nObj=1, nCon=1, ub=1.0, lb=0.0, evaluate=evaluate)

print(problem.evaluate([[0.2, 0.3], [0.8, 0.4]]).objs)
print(problem.evaluate([[0.2, 0.3], [0.8, 0.4]]).cons)
```

## 使用变量类型

默认所有变量都是连续变量。整数变量和离散变量通过 `varType` 指定。

| 值 | 类型 | 含义 |
|---|---|---|
| `0` | Continuous | 边界内任意连续值。 |
| `1` | Integer | 边界内整数值。 |
| `2` | Discrete | 从 `varSet` 中映射的离散值。 |

离散变量必须配合 `varSet` 使用。

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X, axis=1, keepdims=True)


problem = Problem(
    nInput=3,
    nObj=1,
    ub=[1.0, 10.0, 1.0],
    lb=[0.0, 0.0, 0.0],
    varType=[0, 1, 2],
    varSet={2: [0.1, 0.5, 0.9]},
    objFunc=objFunc,
)

X = np.array([[0.2, 3.7, 0.8]])
print(problem.apply_var_type(X))
```

Example output:

```text
[[0.2 4.  0.9]]
```

## 使用 `ModelProblem` 处理仿真

如果主要 callable 是仿真模型，就用 `ModelProblem`。这在校准、时间序列模型、水文模型等场景很常见。

仿真函数同样接收 batched 输入，它通常返回：

```text
(n_samples, n_time, n_series)
```

下面这个例子有两个参数、两个时刻、一个仿真序列。

```python
import numpy as np

from UQPyL.problem import ModelProblem


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, simLabels=["Q"], name="ToyModel")

res = problem.evaluate([[1.0, 2.0], [1.5, 2.5]], target="sim")
print(res.sim.shape)
print(res.sim)
```

Example output:

```text
(2, 2, 1)
[[[1. ]
  [2. ]]

 [[1.5]
  [2.5]]]
```

这三个维度分别表示：

| 下标 | 含义 |
|---|---|
| `sim[i, :, :]` | 第 `i` 个样本的全部仿真输出 |
| `sim[:, t, :]` | 所有样本在第 `t` 个时刻的输出 |
| `sim[:, :, j]` | 第 `j` 个序列在所有样本和时刻上的输出 |

### 用仿真误差构造目标

如果提供了 `obs`，那么 `objFunc` 可以通过 `context` 访问仿真结果和观测值。

```python
import numpy as np

from UQPyL.problem import ModelProblem


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


def objFunc(X, context):
    err = context.sim - context.obs
    return np.mean(err**2, axis=(1, 2)).reshape(-1, 1)


problem = ModelProblem(nInput=2, nObj=1, ub=3.0, lb=0.0, simFunc=simFunc, objFunc=objFunc, obs=obs, simLabels=["Q"])

res = problem.evaluate([[1.0, 2.2], [0.0, 0.0]])
print(res.objs)
```

Example output:

```text
[[0.02]
 [2.5 ]]
```

### ModelProblem 检查清单

| 检查项 | 期望 |
|---|---|
| `obs` | 二维数组，通常是 `(n_time, n_series)` |
| `mask` | 与 `obs` 同形状；`True` 表示忽略该位置 |
| `simFunc(X).shape[0]` | 必须等于 `np.atleast_2d(X).shape[0]` |
| `simFunc(X).shape[1:]` | 如果用了观测，应该与 `obs.shape` 匹配 |
| `objFunc(X, context)` | 返回 `(n_samples, n_obj)` |

## 核心对象

| 概念 | 作用 |
|---|---|
| `Space` | 定义输入维度、边界、标签和变量类型。 |
| `Problem` | 定义静态问题的目标和可选约束。 |
| `ModelProblem` | 定义带观测、掩码和仿真上下文的仿真问题。 |
| `Eval` | `evaluate()` 的标准返回对象。 |
| `singleFunc` | 把单样本目标函数包装成 batched 接口。 |
| `singleEval` | 把单样本组合评估函数包装成 batched 接口。 |

## 常见错误

| 错误 | 会发生什么 | 修法 |
|---|---|---|
| 目标函数返回 `(n_samples,)` | 下游方法通常希望读到二维目标表，可能报错或解释错形状。 | 返回 `(n_samples, n_obj)`，例如 `y.reshape(-1, 1)` 或 `keepdims=True`。 |
| `objFunc` 只按单样本写，却直接接收 batched `X` | 单个样本能跑，多行样本就坏。 | 用 `np.atleast_2d(X)` 并按行计算，或者用 `@singleFunc` 包装。 |
| 仍然写 `res["objs"]` 这种字典式访问 | `Problem.evaluate()` 返回的是 `Eval`，不是字典。 | 用 `res.objs`、`res.cons` 和 `res.sim`。 |
| 本来每个变量范围不同，却把 `lb`、`ub` 写成标量 | 所有变量会共享同一组边界。 | 用向量形式，例如 `lb=[0.0, -5.0]`、`ub=[1.0, 10.0]`。 |
| 把约束符号写反 | 可行和不可行会被颠倒。 | 记住规则是 `cons <= 0` 才可行。 |
| `nCon=1`，但没有返回约束值 | 需要约束信息的方法无法判断可行性。 | 提供 `conFunc`，或在组合式 `evaluate()` 里返回 `Eval(cons=...)`。 |
| `ModelProblem` 的 `simFunc(X)` 返回 `(n_time, n_series)` | 校准期望每个输入样本都有一份仿真结果。 | 返回 `(n_samples, n_time, n_series)`。 |
| 把需要参与计算的观测位置设成 `mask=True` | 这些位置会被忽略。 | 只有缺失值或明确想忽略的位置才设 `True`。 |
| `optType` 写反 | 优化和推断内部会把目标方向理解错。 | 损失/误差用 `"min"`，收益/分数用 `"max"`。 |

## 基准问题

UQPyL 在 `UQPyL.problem` 下内置了一些 benchmark problem。

| 类型 | 例子 |
|---|---|
| 单目标 | `Sphere`、`Ackley`、`Rosenbrock`、`Rastrigin`、`Griewank`、`Trid` |
| 带约束单目标 | `RosenbrockWithCon` |
| 多目标 | `ZDT1`、`ZDT2`、`ZDT3`、`ZDT4`、`ZDT6`、`DTLZ1`-`DTLZ7` |

```python
from UQPyL.problem import Sphere, ZDT1


sphere = Sphere(nInput=10)
zdt1 = ZDT1(nInput=5)

print(sphere.evaluate([[0.0] * 10]).objs)
print(zdt1.evaluate([[0.5] * 5]).objs)
```

## 下一步

| 目标 | 阅读 |
|---|---|
| 查构造参数 | [Problem API](api/problem.md) |
| 从问题对象生成样本 | [Design of Experiment](doe.md) |
| 查看完整工作流示例 | [Examples](examples.md) |
