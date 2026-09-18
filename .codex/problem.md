# Problem模块

`problem` 模块用于将实际问题抽象为 UQPyL 可以统一调用的问题对象。通过这一层抽象，采样、优化、分析、推断、校准和代理建模等功能模块不需要分别适配具体应用场景，而可以通过同一套接口接入不同问题，形成一致的工作流。

使用 `problem` 模块描述一个实际问题，本质上需要明确三件事：

1. 输入变量如何定义
2. 给定一批输入样本，如何完成评估
3. 评估结果需要输出哪些内容

在 UQPyL 中，这三部分统一概括为：

```text
Problem = Space + Evaluation + Eval
```

其中：

- `Space` 用于描述输入空间
- `Evaluation` 用于描述从输入样本到结果的评估过程
- `Eval` 用于描述评估结果的统一返回格式

从评估环节看，一个实际问题通常至少需要考虑以下几类输出：

1. 目标值 `objs`
2. 约束值 `cons`
3. 在仿真问题中，必要时还需要保留模拟输出，例如时间序列或多序列结果 `sims`

这也是为什么 `Eval` 不只是一个单一标量的容器，而是一个统一承载评估结果的数据对象。

后文将先用一个最小例子说明这三部分在代码中的对应关系，再进一步讨论 `Evaluation` 的不同组织方式，以及由此形成的两类问题对象。

## 最小例子

下面用一个最简单的 `Problem` 例子说明这三个部分如何对应到代码。

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(
    nInput=2,
    nObj=1,
    lb=-1.0,
    ub=1.0,
    objFunc=objFunc,
)

res = problem.evaluate([[0.2, 0.3]])

print(problem.xLabels)
print(res.objs)
```

在这个例子中：

- `nInput`、`lb`、`ub` 共同定义了输入空间 `Space`
- `objFunc` 定义了评估过程 `Evaluation`
- `problem.evaluate(...)` 返回的 `res` 是统一结果对象 `Eval`

也就是说，其下游的功能模块不需要关心这个问题对应的具体应用场景，只需要按照统一接口调用 `evaluate()` 并读取 `Eval` 即可。

这个例子同时也说明，问题抽象里的三部分分工并不相同。`Space` 用来描述输入空间，`Eval` 用来组织返回结果，而 `Evaluation` 则连接输入和输出，决定一批样本究竟如何完成计算。

在很多问题中，`Space` 和 `Eval` 的外部形式往往比较稳定，真正变化更明显的部分通常是 `Evaluation`。也就是说，不同实际问题最主要的差异，往往不在“输入怎么定义”或“结果怎么承载”，而在“计算过程如何展开”。

沿着这个思路继续看，就会发现 `Evaluation` 内部通常有两种比较稳定的组织方式。

一种是直接评估：

```text
X -> objFunc/conFunc -> Eval
```

这类问题中，输入样本进入后就可以直接计算目标和约束，中间不需要单独保留一个仿真结果层。

另一种是仿真后评估：

```text
X -> simFunc -> objFunc/conFunc -> Eval
```

这类问题中，输入样本进入后需要先运行仿真模型，再由仿真输出构造目标、约束或误差指标。此时仿真输出本身通常也需要作为正式结果的一部分被保留和传递。

因此，虽然问题抽象在外部接口层是统一的，`Evaluation` 这一层内部仍然可以自然分出两种不同的组织方式。对应到实际使用中，最常见的就是下面两种情况。

## 问题类型

第一种情况是，目标和约束可以直接由输入样本计算得到。这类问题中，评估过程主要围绕 `objFunc` 和 `conFunc` 展开，典型例子包括数学测试函数、常规优化问题和黑箱打分问题。

第二种情况是，输入样本进入后还需要先运行一个仿真模型，再由仿真输出构造目标、约束或误差指标。这类问题中，评估过程通常会进一步引入 `simFunc`，典型例子包括校准、时序仿真、过程模型和多序列误差分析等问题。

为了分别承载这两种情况，UQPyL 提供了两个正式的问题对象：

| 对象 | 适用情况 | 评估链条 |
|---|---|---|
| `Problem` | 输入后可直接完成目标/约束计算 | `X -> objFunc/conFunc` |
| `ModelProblem` | 输入后需要先仿真，再完成后续评估 | `X -> simFunc -> objFunc/conFunc` |

## 共同接口

`Problem` 和 `ModelProblem` 的内部评估链条不同，但它们提供给下游功能模块是同一组基础接口。共享部分包括：输入空间、批量输入约定、评估入口和返回协议。

### `Space`

`Space` 用于描述输入空间。常见字段包括：

| 字段 | 含义 |
|---|---|
| `nInput` | 输入变量个数 |
| `lb`, `ub` | 输入变量上下界 |
| `varType` | 变量类型编码 |
| `varSet` | 离散变量取值集合 |
| `xLabels` | 输入变量标签 |

若未显式传入 `space=...`，两个问题对象会根据构造参数自动生成 `Space`。

### batched `X`

UQPyL 默认按批量评估设计，标准输入形状为：

```text
(nSamples, nInput)
```

因此，用户定义的目标函数、约束函数与仿真函数通常都应接收二维输入。常见写法为：

```python
X = np.atleast_2d(X)
```

### `evaluate(X, target=None)`

两类问题都通过 `evaluate()` 进入正式评估流程。

| 对象 | 方法签名 |
|---|---|
| `Problem` | `evaluate(X, target=None)` |
| `ModelProblem` | `evaluate(X, target=None)` |

`target` 用于声明本次请求希望返回的结果子集，包括四个固定字段：`None`、`"objs"`、`"cons"`、`"sims"`

### `Eval`

两类问题统一返回 `Eval` 对象。

| 字段 | 含义 |
|---|---|
| `objs` | 目标矩阵 |
| `cons` | 约束矩阵 |
| `sims` | 仿真输出 |

推荐的返回方式为：

```python
return Eval(objs=objs, cons=cons, sims=sims, target=target)
```

## 两类问题的边界

共享的是外部接口，不同的是内部评估链。

| 对象 | 内部主链 | 关键语义 |
|---|---|---|
| `Problem` | `X -> objFunc/conFunc -> Eval` | 无独立仿真阶段 |
| `ModelProblem` | `X -> simFunc -> simContext -> objFunc/conFunc -> Eval` | `sims` 是正式中间结果 |

这一区别进一步体现在三个方面。

### callable 签名

| 对象 | 基础 callable |
|---|---|
| `Problem` | `objFunc(X)`、`conFunc(X)` |
| `ModelProblem` | `simFunc(X)`、`objFunc(X, simContext)`、`conFunc(X, simContext)` |

### evaluator 接口

| 对象 | evaluator 类型 | 方法签名 |
|---|---|---|
| `Problem` | `Evaluator` | `evaluate(X, target=None)` |
| `ModelProblem` | `ModelEvaluator` | `evaluate(X, simContext, target=None)` |

### 可返回结果

| 对象 | 合法 `target` |
|---|---|
| `Problem` | `None` / `"objs"` / `"cons"` |
| `ModelProblem` | `None` / `"objs"` / `"cons"` / `"sims"` |

## `Problem`：直接评估型问题

`Problem` 用于表达目标与约束可以直接由输入样本计算的问题。其标准评估链为：

```text
X -> objFunc(X) / conFunc(X) -> Eval
```

### 基础定义方式

`Problem` 的基础定义方式有两层：

1. 定义 `objFunc`
2. 定义 `objFunc + conFunc`

示例：

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


def conFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] + X[:, 1] - 1.0).reshape(-1, 1)


problem = Problem(
    nInput=2,
    nObj=1,
    nCon=1,
    lb=0.0,
    ub=1.0,
    objFunc=objFunc,
    conFunc=conFunc,
)
```

约束统一按 `cons <= 0` 判定可行。

### 高级定义方式

当 `objFunc / conFunc` 不能完整表达评估逻辑时，可以自定义 `Evaluator`。

```python
import numpy as np

from UQPyL.problem import Eval, Evaluator, Problem


class QuadraticEvaluator(Evaluator):
    def evaluate(self, X, target=None):
        X = np.atleast_2d(X)
        objs = np.sum(X**2, axis=1, keepdims=True)
        cons = (X[:, 0] + X[:, 1] - 1.0).reshape(-1, 1)
        return Eval(objs=objs, cons=cons, target=target)


problem = Problem(
    nInput=2,
    nObj=1,
    nCon=1,
    lb=0.0,
    ub=1.0,
    evaluator=QuadraticEvaluator(),
)
```

## `ModelProblem`：仿真评估型问题

`ModelProblem` 用于表达必须先运行仿真模型，再依据仿真结果构造目标或约束的问题。其标准评估链为：

```text
X -> simFunc(X) -> simContext -> objFunc(X, simContext) / conFunc(X, simContext) -> Eval
```

### 基础定义方式

`ModelProblem` 的基础定义方式通常分为三层：

1. 定义 `simFunc`
2. 定义 `simFunc + objFunc`
3. 定义 `simFunc + objFunc + conFunc`

示例：

```python
import numpy as np

from UQPyL.problem import ModelProblem


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sims = np.zeros((X.shape[0], 2, 1))
    sims[:, 0, 0] = X[:, 0]
    sims[:, 1, 0] = X[:, 1]
    return sims


def objFunc(X, simContext):
    err = simContext.sims - simContext.obs
    return np.mean(err**2, axis=(1, 2)).reshape(-1, 1)


problem = ModelProblem(
    nInput=2,
    nObj=1,
    lb=0.0,
    ub=3.0,
    simFunc=simFunc,
    objFunc=objFunc,
    obs=obs,
    seriesLabels=["Q"],
)
```

### `simContext`

`ModelProblem` 在完成仿真后，会构造 `simContext` 并传入目标函数、约束函数或 `ModelEvaluator`。当前常用字段包括：

- `simContext.sims`
- `simContext.obs`
- `simContext.mask`

### 高级定义方式

当基础的 `simFunc + objFunc (+ conFunc)` 不能完整表达后处理逻辑时，可以自定义 `ModelEvaluator`。

```python
import numpy as np

from UQPyL.problem import Eval, ModelEvaluator, ModelProblem


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sims = np.zeros((X.shape[0], 2, 1))
    sims[:, 0, 0] = X[:, 0]
    sims[:, 1, 0] = X[:, 1]
    return sims


class MSEEvaluator(ModelEvaluator):
    def evaluate(self, X, simContext, target=None):
        err = simContext.sims - simContext.obs
        objs = np.mean(err**2, axis=(1, 2)).reshape(-1, 1)
        return Eval(objs=objs, sims=simContext.sims, target=target)


problem = ModelProblem(
    nInput=2,
    nObj=1,
    lb=0.0,
    ub=3.0,
    simFunc=simFunc,
    obs=obs,
    evaluator=MSEEvaluator(),
)
```

此外，`ModelProblem` 还支持：

- `simulate(X)`：仅运行仿真并返回 `simContext`
- `evaluate(X, target="sims")`：仅返回仿真输出

## 下一步

这一版主要用于对比文档结构。若这一主线你认可，后续可以再继续补两部分：

1. 将 `Space`、`Problem`、`ModelProblem` 的完整参数表补齐
2. 将标签系统与 `target` 规则单独整理成一节
