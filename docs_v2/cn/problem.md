# Problem 模块

`Problem` 模块用于将实际问题抽象为UQPyL各功能模块可以统一调用的问题对象，从而为采样、优化、分析、推断、校准和代理建模等任务提供一致的接入方式，进而构建通用的工作流。

## 单位区间与真实值转换

| 接口 | 含义 |
|---|---|
| `problem.unit_to_space(U)` | 解码单位区间样本，返回真实值副本。 |
| `problem.space_to_unit(X)` | 编码真实值，返回单位区间副本。 |
| `problem.canonicalize_unit(U)` | 将等价整数/离散编码归并到唯一代表点，返回单位区间副本。 |

转换由 `Space` 实现，`Problem` 提供统一入口。连续变量按有限上下界线性转换；固定连续变量编码为 `0.5`。整数变量在 `ceil(lb)` 到 `floor(ub)` 的合法整数间等宽分段，离散变量按 `varSet` 顺序等宽分段，编码时返回区间中点；`U=1` 对应最后一个值。这也意味着整数采样采用等宽整数区间，而非先线性缩放再取整。

离散 `varSet` 当前要求互不重复的有限数值；离散真实取值由 `varSet` 决定，不必落在该列用于旧编码的 `lb/ub` 内。真实值编码时检查边界、整数合法性和候选成员关系。默认解码与编码满足 `unit_to_space(space_to_unit(X)) == X`（浮点误差范围内）；反向仅得到规范代表点，不保证恢复原始整数/离散编码。

`evaluate(X)` 接收真实值，不会猜测是否为单位区间编码。旧的 `apply_var_type()` 仍供已有路径使用，但不再用于优化评估入口。


从UQPyL视角看，一个实际问题进入框架，至少需要回答三个基本问题：

1. 输入变量如何定义
2. 给定一批输入参数样本，如何完成评估
3. 评估结果需要输出哪些内容

因此，UQPyL将问题拆分为三部分：

```text
Problem = Space + Evaluation + Eval
```

其中：

- `Space` 用于描述输入空间
- `Evaluation` 用于描述从输入样本到结果的评估过程
- `Eval` 用于描述评估结果的统一返回格式

在这个抽象中，`Space` 和 `Eval` 提供稳定的外部结构，使得UQPyL的功能模块通用地接入。

不同实际问题最主要的差异通常体现在 `Evaluation` ，即“输入之后怎样完成计算”。

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

也就是说，功能模块不需要关心问题的具体评估过程，只需要通过统一接口调用 `evaluate()`，并从 `Eval` 中读取结果即可。

由这个例子可以看到，`Space`、`Evaluation` 和 `Eval` 虽然共同构成了问题抽象，但承担的角色并不相同。

- `Space` 负责定义输入变量的维度、范围、类型与标签
- `Evaluation` 负责定义输入样本如何被计算为结果
- `Eval` 负责统一组织目标、约束和仿真输出等结果内容

其中，`Space` 和 `Eval` 的外部形式通常较为稳定，而 `Evaluation` 的组织方式会随着问题本身而变化。由此可以自然引出两类常见的评估链条。

介绍两类问题对象前，还需要先建立一个非常重要的输入约定：UQPyL 默认按批量评估组织问题评估。

## Batched `X`

在 UQPyL 中，输入 `X` 的标准形状为：

```text
(nSamples, nInput)
```

也就是说：

- `X` 的每一行对应一个输入样本
- `X` 的每一列对应一个输入变量

例如，对于一个二输入问题：

```python
singleX = np.array([0.2, 0.3])
batchX = np.array([
    [0.2, 0.3],
    [0.5, 0.1],
    [0.0, 1.0],
])

print(np.atleast_2d(singleX).shape)
print(np.atleast_2d(batchX).shape)
```

输出为：

```text
(1, 2)
(3, 2)
```

因此，在自定义 `objFunc`、`conFunc` 或 `simFunc` 时，通常建议先写：

```python
X = np.atleast_2d(X)
```

这样无论传入单个样本还是一批样本，函数内部都可以按统一的二维输入处理。

在这个约定下，一些常见写法的含义也会更加清楚：

| 表达式 | 含义 |
|---|---|
| `X[:, 0]` | 所有样本的第 1 个输入变量 |
| `X[:, 1]` | 所有样本的第 2 个输入变量 |
| `X.shape[0]` | 样本个数 |
| `reshape(-1, 1)` | 将一维结果整理为单列输出 |
| `keepdims=True` | 在规约运算后保留列维度 |

## 两类问题对象

第一类情况中，输入样本进入后可以直接计算目标与约束：

```text
X -> objFunc/conFunc -> Eval
```

这类问题中，评估过程主要围绕目标函数与约束函数展开，中间不需要单独保留一个独立的仿真输出层。

第二类情况中，输入样本进入后需要先运行仿真模型，再由仿真输出构造目标、约束或误差指标：

```text
X -> simFunc -> objFunc/conFunc -> Eval
```

这类问题中，仿真输出本身往往也是正式结果的一部分，需要在后续模块中继续使用或保留。

因此，虽然问题抽象在外部接口层是统一的，但在 `Evaluation` 这一层内部，仍然会稳定地形成两种不同的评估组织方式。为了分别承载这两种情况，UQPyL 提供了两个正式的问题对象。

## 两个问题对象

| 对象 | 适用情况 | 评估链条 |
|---|---|---|
| `Problem` | 输入后可直接完成目标或约束计算 | `X -> objFunc/conFunc -> Eval` |
| `ModelProblem` | 输入后需要先运行仿真，再完成后续评估 | `X -> simFunc -> objFunc/conFunc -> Eval` |

这两个对象共享统一的外部接口，但内部评估结构不同。

### `Problem`

`Problem` 用于表达直接评估型问题。典型场景包括数学测试函数、常规优化问题以及可直接由输入样本计算评分结果的黑箱问题。

其标准评估链条为：

```text
X -> objFunc(X) / conFunc(X) -> Eval
```

### `ModelProblem`

`ModelProblem` 用于表达仿真评估型问题。典型场景包括模型校准、时序仿真、过程模型、多序列误差分析等问题。

其标准评估链条为：

```text
X -> simFunc(X) -> simContext -> objFunc(X, simContext) / conFunc(X, simContext) -> Eval
```

其中，`simFunc` 负责生成仿真输出，框架随后基于仿真结果构造 `simContext`，再将其传入目标函数与约束函数。

## 共同接口

`Problem` 与 `ModelProblem` 虽然内部链条不同，但对下游模块暴露的是同一套正式接口。共享部分主要包括输入空间、统一评估入口与统一返回协议。

### Space

`Space` 用于描述输入空间。若不显式传入 `space=...`，`Problem` 和 `ModelProblem` 会根据构造参数自动生成。

常见字段包括：

| 字段 | 含义 |
|---|---|
| `nInput` | 输入变量个数 |
| `lb`, `ub` | 输入变量上下界 |
| `varType` | 变量类型编码 |
| `varSet` | 离散变量取值集合 |
| `xLabels` | 输入变量标签 |

其中，最基本的三个量是 `nInput`、`lb` 和 `ub`，它们共同定义输入空间的维度与范围。

如果所有变量共用相同的上下界，可以直接传入标量：

```python
problem = Problem(
    nInput=3,
    nObj=1,
    lb=0.0,
    ub=1.0,
    objFunc=objFunc,
)
```

这表示 3 个输入变量都位于 `[0.0, 1.0]`。

如果每个变量的范围不同，则逐维传入：

```python
problem = Problem(
    nInput=3,
    nObj=1,
    lb=[0.0, -5.0, 50.0],
    ub=[1.0, 10.0, 100.0],
    xLabels=["width", "slope", "storage"],
    objFunc=objFunc,
)
```

此时可以理解为：

| 变量 | 下界 | 上界 |
|---|---:|---:|
| `width` | `0.0` | `1.0` |
| `slope` | `-5.0` | `10.0` |
| `storage` | `50.0` | `100.0` |

`Space` 还可以进一步表达变量类型。`varType` 采用逐维编码：

| 编码 | 类型 | 含义 |
|---|---|---|
| `0` | 连续变量 | 在边界内连续取值 |
| `1` | 整数变量 | 在边界内取整 |
| `2` | 离散变量 | 由 `varSet` 指定合法取值 |

离散变量需要与 `varSet` 配合使用。例如：

```python
problem = Problem(
    nInput=3,
    nObj=1,
    lb=[0.0, 0.0, 0.0],
    ub=[1.0, 10.0, 1.0],
    varType=[0, 1, 2],
    varSet={2: [0.1, 0.5, 0.9]},
    objFunc=objFunc,
)
```

这里第 1 维是连续变量，第 2 维是整数变量，第 3 维是离散变量。

`Space` 还提供若干与变量变换相关的方法，这些方法可通过问题对象直接调用：

| 方法 | 作用 |
|---|---|
| `validate(X)` | 检查输入形状是否为 `(nSamples, nInput)` |
| `cast_int_vars(X)` | 对整数变量执行取整 |
| `map_discrete_vars(X)` | 对离散变量映射到 `varSet` 指定取值 |
| `apply_var_type(X)` | 同时执行整数与离散变量处理 |
| `unit_to_space(X)` | 将 `[0, 1]` 单位超立方缩放到真实输入空间 |

### evaluate()

两类问题都通过 `evaluate()` 进入正式评估流程。

| 对象 | 方法签名 |
|---|---|
| `Problem` | `evaluate(X, target=None)` |
| `ModelProblem` | `evaluate(X, target=None)` |

`evaluate()` 的职责是统一完成输入检查、评估执行、结果校验与 `Eval` 封装。

### Eval

两类问题统一返回 `Eval` 对象。其标准结果字段为：

```python
@dataclass
class Eval:
    objs: np.ndarray | None = None
    cons: np.ndarray | None = None
    sims: np.ndarray | None = None
```

未提供的结果字段必须为 `None`，而不是空数组。标准形状如下：

| 字段 | 标准形状 |
|---|---|
| `objs` | `(nSamples, nObj)` |
| `cons` | `(nSamples, nCon)` |
| `sims` | `(nSamples, nTime, nSeries)` |

`Eval` 还提供以下便捷属性：

```python
res = problem.evaluate(X)
print(res.hasObjs)
print(res.hasCons)
print(res.hasSims)
```

UQPyL 的大多数下游模块都默认按“样本逐行、结果逐列”读取输出，因此单目标函数在接口上仍应返回列矩阵 `(nSamples, 1)`，而不是一维数组 `(nSamples,)`。

### target

`evaluate(X, target=None)` 中的 `target` 用于声明本次调用关注的结果子集。不请求的结果字段会被置为 `None`。

| target | 含义 | Problem | ModelProblem |
|---|---|:---:|:---:|
| `None` | 返回所有可用结果 | ✅ | ✅ |
| `"objs"` | 仅返回目标值 | ✅ | ✅ |
| `"cons"` | 仅返回约束值 | ✅ | ✅ |
| `"sims"` | 仅返回仿真输出 | ❌ | ✅ |

对 `ModelProblem` 而言，仿真阶段始终校验 `sims`；最终返回由 `target` 决定。`None` 返回所有可用字段，`"objs"` / `"cons"` 仅返回所请求字段，`"sims"` 仅返回仿真输出并跳过目标与约束计算。

```python
res = problem.evaluate(X, target="objs")
print(res.objs)
print(res.cons)
print(res.sims)
```

## `Problem`：直接评估型问题

`Problem` 适用于目标与约束可以直接由输入样本计算的问题。

### 基础定义方式

基础用法中，用户通过 `objFunc` 和可选的 `conFunc` 定义评估过程。

| callable | 说明 |
|---|---|
| `objFunc(X)` | 必需，返回目标矩阵 `(nSamples, nObj)` |
| `conFunc(X)` | 可选，返回约束矩阵 `(nSamples, nCon)` |

约束统一按 `cons <= 0` 判定为可行。

例如，若实际约束为：

```text
x1 + x2 <= 1.0
```

则可以写为：

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

如果有多个目标，则每个目标占据 `objs` 的一列；如果有多个约束，则每个约束占据 `cons` 的一列。

### 高级定义方式

当 `objFunc` 与 `conFunc` 不能完整表达评估流程时，可以继承 `Evaluator` 自定义评估器。

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

`evaluator` 与 `objFunc` / `conFunc` 互斥，不能同时传入。

## `ModelProblem`：仿真评估型问题

`ModelProblem` 适用于需要先运行仿真模型，再基于仿真输出计算目标或约束的问题。

### 基础定义方式

基础用法中，用户需要提供 `simFunc`，并可进一步提供依赖 `simContext` 的 `objFunc` 与 `conFunc`。

| callable | 说明 |
|---|---|
| `simFunc(X)` | 必需，返回仿真输出 `sims` |
| `objFunc(X, simContext)` | 可选，返回目标矩阵 |
| `conFunc(X, simContext)` | 可选，返回约束矩阵 |

`simContext` 由框架在 `simFunc` 执行后自动构造，用于向后续评估步骤传递仿真结果与观测信息。

从结果结构上看，`simFunc(X)` 的标准返回形状为：

```text
(nSamples, nTime, nSeries)
```

其中：

- 第 1 维对应样本
- 第 2 维对应时间步或过程步
- 第 3 维对应输出序列

```python
import numpy as np

from UQPyL.problem import ModelProblem


obs = np.array([[1.0], [0.8], [2.0]])
mask = np.array([[False], [True], [False]])


def simFunc(X):
    X = np.atleast_2d(X)
    nSamples = X.shape[0]
    sims = np.zeros((nSamples, 3, 1))
    sims[:, 0, 0] = X[:, 0]
    sims[:, 1, 0] = 0.5 * X[:, 0] + 0.5 * X[:, 1]
    sims[:, 2, 0] = X[:, 1]
    return sims


def objFunc(X, simContext):
    err = simContext.sims - simContext.obs
    if simContext.mask is not None:
        err = err[:, ~simContext.mask]
    return np.mean(err**2, axis=(1, 2)).reshape(-1, 1)


problem = ModelProblem(
    nInput=2,
    nObj=1,
    lb=0.0,
    ub=3.0,
    simFunc=simFunc,
    objFunc=objFunc,
    obs=obs,
    mask=mask,
    seriesLabels=["Q"],
)
```

### SimContext

`ModelProblem` 在执行 `simFunc` 后会构造 `SimContext`，用于向后续评估过程传递与仿真相关的上下文信息。

```python
@dataclass(frozen=True)
class SimContext:
    sims: np.ndarray
    obs: np.ndarray | None
    mask: np.ndarray | None
```

其中：

- `sims` 为仿真输出
- `obs` 为观测数据，形状通常为 `(nTime, nSeries)`
- `mask` 为缺测掩码，形状与 `obs` 一致

如果提供了 `obs`，那么 `objFunc(X, simContext)` 或 `conFunc(X, simContext)` 就可以基于仿真输出与观测数据之间的偏差构造目标值或约束值。这也是 `ModelProblem` 与 `Problem` 在评估结构上的核心差异之一。

### simulate()

若只希望执行仿真，而不进入目标或约束计算，可以调用 `simulate()`：

```python
simContext = problem.simulate(X)
print(simContext.sims)
print(simContext.obs)
```

与 `evaluate(X, target="sims")` 相比，`simulate()` 直接返回 `SimContext`，适合需要继续访问 `obs` 或 `mask` 的场景。

### 高级定义方式

当基础的 `simFunc + objFunc + conFunc` 组合仍不足以表达完整流程时，可以继承 `ModelEvaluator` 自定义评估器。

```python
import numpy as np

from UQPyL.problem import Eval, ModelEvaluator, ModelProblem


class MSEEvaluator(ModelEvaluator):
    def evaluate(self, X, simContext, target=None):
        err = simContext.sims - simContext.obs
        if simContext.mask is not None:
            err = err[:, ~simContext.mask]
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

`evaluator` 与 `objFunc` / `conFunc` 互斥，不能同时传入。

## 其他常用内容

### optType

`optType` 用于声明每个目标的优化方向。支持两种形式：

- 字符串：`"min"` 或 `"max"`
- 列表：如 `["min", "max", "min"]`

框架内部会自动转换为统一的最小化形式。

### singleFunc

`singleFunc` 可将只处理单个样本的函数包装为批量函数。

```python
from UQPyL.problem import Problem, singleFunc


@singleFunc
def objFunc(x):
    return x[0] ** 2 + x[1] ** 2


problem = Problem(
    nInput=2,
    nObj=1,
    lb=-1.0,
    ub=1.0,
    objFunc=objFunc,
)
```

## 常见错误

以下问题在初次定义 `Problem` 或 `ModelProblem` 时最常见：

| 错误 | 影响 | 修法 |
|---|---|---|
| `objFunc` 返回 `(nSamples,)` | 下游模块通常期望二维目标矩阵 | 改为 `reshape(-1, 1)` 或使用 `keepdims=True` |
| 把 `X[0]` 当作第 1 个变量 | `X[0]` 实际是第 1 个样本 | 使用 `X[:, 0]` |
| 忘记写 `np.atleast_2d(X)` | 单样本与批量输入的行为可能不一致 | 在函数开头统一转为二维 |
| 约束符号写反 | 可行与不可行会被颠倒 | 统一按 `cons <= 0` 表示可行 |
| `simFunc(X)` 返回 `(nTime, nSeries)` | 无法与 batched 输入对齐 | 返回 `(nSamples, nTime, nSeries)` |
| 仍然使用 `res["objs"]` 访问结果 | `evaluate()` 返回的是 `Eval` 而不是字典 | 使用 `res.objs`、`res.cons`、`res.sims` |

## 构造参数总览

下面给出 `Problem` 与 `ModelProblem` 的完整构造参数列表，便于在理解问题抽象之后直接查阅实际建模入口。

### `Problem` 构造参数

| 参数 | 类型 | 是否必需 | 默认值 | 说明 |
|---|---|:---:|---|---|
| `nInput` | `int` | 是 | `None` | 输入变量个数 |
| `nObj` | `int` | 是 | `None` | 目标个数 |
| `ub` | `int` / `float` / `list` / `np.ndarray` | 是 | `None` | 输入上界，可为标量或逐维数组 |
| `lb` | `int` / `float` / `list` / `np.ndarray` | 是 | `None` | 输入下界，可为标量或逐维数组 |
| `objFunc` | `callable` | 条件必需 | `None` | 目标函数，基础模式下必须提供；与 `evaluator` 互斥 |
| `conFunc` | `callable` | 否 | `None` | 约束函数，提供时必须与 `objFunc` 一起使用 |
| `conWgt` | `list` | 否 | `None` | 约束权重 |
| `nCon` | `int` | 否 | `0` | 约束个数 |
| `varType` | `list` | 否 | `None` | 变量类型编码列表：`0` 连续，`1` 整数，`2` 离散 |
| `varSet` | `list` | 否 | `None` | 离散变量取值集合，通常与 `varType=2` 配合使用 |
| `optType` | `list` / `str` | 否 | `"min"` | 优化方向，可统一指定或逐目标指定 |
| `xLabels` | `list` | 否 | `None` | 输入变量标签 |
| `name` | `str` | 否 | 类名 | 问题名称 |
| `space` | `SpaceBase` | 否 | `None` | 自定义输入空间对象 |
| `objLabels` | `list` | 否 | `None` | 目标标签 |
| `conLabels` | `list` | 否 | `None` | 约束标签 |
| `evaluator` | `EvaluatorBase` | 条件必需 | `None` | 自定义评估器；与 `objFunc` / `conFunc` 互斥 |

其中有三条使用规则需要特别注意：

1. `Problem` 必须提供 `objFunc`，或者提供自定义 `evaluator`
2. `conFunc` 不能单独出现，必须与 `objFunc` 一起使用
3. `evaluator` 与 `objFunc` / `conFunc` 互斥，不能同时传入

### `ModelProblem` 构造参数

`ModelProblem` 继承了大部分与输入空间和目标定义相关的参数，同时额外增加了仿真问题所需的参数。

| 参数 | 类型 | 是否必需 | 默认值 | 说明 |
|---|---|:---:|---|---|
| `nInput` | `int` | 是 | `None` | 输入变量个数 |
| `nObj` | `int` | 否 | `1` | 目标个数 |
| `ub` | `int` / `float` / `list` / `np.ndarray` | 是 | `None` | 输入上界 |
| `lb` | `int` / `float` / `list` / `np.ndarray` | 是 | `None` | 输入下界 |
| `simFunc` | `callable` | 是 | `None` | 仿真函数，返回 `sims` |
| `objFunc` | `callable` | 否 | `None` | 目标函数，签名为 `objFunc(X, simContext)`；与 `evaluator` 互斥 |
| `conFunc` | `callable` | 否 | `None` | 约束函数，签名为 `conFunc(X, simContext)` |
| `obs` | `np.ndarray` | 否 | `None` | 观测数据，标准形状为 `(nTime, nSeries)` |
| `mask` | `np.ndarray` | 否 | `None` | 缺测掩码，形状需与 `obs` 一致 |
| `conWgt` | `list` | 否 | `None` | 约束权重 |
| `nCon` | `int` | 否 | `0` | 约束个数 |
| `varType` | `list` | 否 | `None` | 变量类型编码 |
| `varSet` | `list` | 否 | `None` | 离散变量取值集合 |
| `optType` | `list` / `str` | 否 | `"min"` | 优化方向 |
| `xLabels` | `list` | 否 | `None` | 输入变量标签 |
| `name` | `str` | 否 | 类名 | 问题名称 |
| `space` | `SpaceBase` | 否 | `None` | 自定义输入空间对象 |
| `objLabels` | `list` | 否 | `None` | 目标标签 |
| `conLabels` | `list` | 否 | `None` | 约束标签 |
| `evaluator` | `ModelEvaluatorBase` | 否 | `None` | 自定义仿真评估器；与 `objFunc` / `conFunc` 互斥 |
| `seriesLabels` | `list` | 否 | `None` | 仿真输出序列标签 |

`ModelProblem` 还需要额外注意以下规则：

1. `simFunc` 是必需参数
2. `conFunc` 不能单独出现，必须与 `objFunc` 一起使用
3. `evaluator` 与 `objFunc` / `conFunc` 互斥，不能同时传入
4. `mask` 不能脱离 `obs` 单独使用
5. 当未显式提供 `seriesLabels` 且 `obs` 存在时，会按观测序列数自动生成

### 内置测试问题

`UQPyL.problem` 提供了一系列经典测试问题，可直接实例化使用。

```python
from UQPyL.problem import Ackley, DTLZ2, Rosenbrock, Sphere, ZDT1

problem = Sphere(nInput=10)
problem = Rosenbrock(nInput=5)
problem = ZDT1(nInput=30)
problem = DTLZ2(nInput=12, nObj=3)
```

更多构造参数与完整字段说明可参考 [API 文档](./api/problem.md)。

### 约束权重

`Problem(conWgt=[10, 1], nCon=2, ...)` 为每个约束指定一个有限非负权重，长度必须等于 `nCon`。`None` 表示不加权；零权重表示忽略该约束，包括其可行性判断。

```python
CV = np.sum(np.maximum(0, cons * conWgt), axis=1)
```
