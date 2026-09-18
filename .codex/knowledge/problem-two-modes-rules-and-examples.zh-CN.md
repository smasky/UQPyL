# UQPyL `problem` 两种模式规则与示例

## 目的

这份文档只回答一件事：

**当前 UQPyL 的两种问题模式，到底应该怎么定规则。**

这里先不讨论大改造，只先把现阶段最重要的使用规则定清楚。

---

## 两种模式

当前建议明确保留两种模式：

### 1. 普通模式

主链：

```text
X -> objs/cons
```

对应对象：

- `Problem`

适用场景：

- 数学测试函数
- 一般优化问题
- 敏感性分析问题
- 代理建模训练数据生成
- 内部虽然有计算过程，但中间结果不需要暴露

### 2. 仿真模式

主链：

```text
X -> sim -> objs/cons
```

对应对象：

- `ModelProblem`

适用场景：

- 水文模型
- 过程模型
- 率定问题
- 需要保留模拟序列的分析问题
- 目标和约束依赖仿真结果的问题

---

## 最核心的分界线

分界线不是：

`内部有没有模拟`

真正的分界线是：

`模拟结果是否需要成为正式接口的一部分`

### 用 `Problem`

如果你只关心最终目标或约束，不关心：

- `sim`
- `obs`
- `mask`
- 中间仿真结果复用

那就用 `Problem`。

即使内部真的跑了模型，也可以用 `Problem`。

### 用 `ModelProblem`

如果模拟结果需要被正式保留、复用、对比、输出，就用 `ModelProblem`。

比如：

- `objFunc` 要用 `sim`
- `conFunc` 也要用 `sim`
- 率定方法要比较 `sim` 和 `obs`
- 后续要输出模拟序列

---

## 规则一：用户接口仍然分两种

这是当前最重要的结论：

**用户实现时，仍然需要区分两种模式。**

不建议强行统一签名。

### `Problem` 的用户函数

```python
objFunc(X)
conFunc(X)
evaluate(X)
```

### `ModelProblem` 的用户函数

```python
simFunc(X)
objFunc(X, context)
conFunc(X, context)
evaluate(X, context)
```

也就是说：

- 普通模式：无 `context`
- 仿真模式：显式 `context`

---

## 规则二：`objFunc` / `conFunc` 必须保留

后续即使底层抽象改成 `Space + Simulator + Evaluator`，高层接口仍然建议保留：

- `objFunc`
- `conFunc`
- `evaluate`

原因：

- 这是最简单的用户入口
- 普通问题没必要理解 `Evaluator`
- 大多数用户只想定义目标和约束

所以要明确：

**内部可以统一，外部不必统一。**

---

## 规则三：普通模式的正式约定

### 适用对象

- `Problem`

### 输入函数签名

```python
objFunc(X)
conFunc(X)
evaluate(X)
```

### 输入语义

- `X` 是二维批量数组
- 形状为 `(n_samples, n_input)`

### 输出语义

- `objFunc(X)` 返回 `(n_samples, n_obj)`
- `conFunc(X)` 返回 `(n_samples, n_con)`
- `evaluate(X)` 返回 `Eval`

### 约束规则

默认约定：

```text
cons <= 0
```

### 目标选择规则

- 只有目标：`objFunc`
- 目标和约束分开：`objFunc + conFunc`
- 必须一起算：`evaluate`

---

## 规则四：仿真模式的正式约定

### 适用对象

- `ModelProblem`

### 输入函数签名

```python
simFunc(X)
objFunc(X, context)
conFunc(X, context)
evaluate(X, context)
```

### 标准链路

```text
X -> simFunc(X) -> context.sim -> objFunc/conFunc -> Eval
```

### `context` 的定位

`context` 是仿真问题的正式中间结果视图。

当前建议稳定保留这几个字段：

- `context.sim`
- `context.obs`
- `context.mask`

含义：

- `sim`：模拟结果
- `obs`：观测数据
- `mask`：有效位置 / 缺测掩码

### 输出语义

- `simFunc(X)` 返回批量 `sim`
- `objFunc(X, context)` 返回 `(n_samples, n_obj)`
- `conFunc(X, context)` 返回 `(n_samples, n_con)`
- `evaluate(X, context)` 返回 `Eval`

---

## 规则五：什么时候内部有模拟也仍然用 `Problem`

这是一个容易混淆的点。

下面这种情况仍然应该允许：

```text
X -> 内部模拟 -> 内部算分 -> objs/cons
```

即使内部跑了模型，只要：

- 中间 `sim` 不需要暴露
- `objFunc` 自己就能封装完
- `conFunc` 不需要复用仿真结果

那仍然可以定义成 `Problem`。

这是合法且合理的。

### 典型例子

- 黑箱模型只返回一个评分
- 一个外部程序执行完后只读出单个目标值
- 仿真过程只是实现细节，不属于公共协议

---

## 规则六：什么时候必须考虑 `ModelProblem`

下面这些情况，建议优先用 `ModelProblem`：

- `objFunc` 和 `conFunc` 都依赖同一个 `sim`
- 需要观测对比
- 需要 `obs` / `mask`
- 需要输出序列结果
- 需要避免重复仿真
- 需要和 hydroPilot 这类执行器自然联动

一句话：

**只要仿真结果本身开始成为公共语义，就不应继续藏在普通 `Problem` 里。**

---

## 规则七：`evaluate` 仍然保留两种版本

### 普通模式

```python
evaluate(X)
```

### 仿真模式

```python
evaluate(X, context)
```

适用原则：

- 如果目标和约束必须一起算，用 `evaluate`
- 否则优先 `objFunc` / `conFunc`

---

## 规则八：`context` 不要失控

`context` 是正式接口，但不应该变成无限制杂物箱。

当前建议先只把下面几个字段当成正式协议：

- `sim`
- `obs`
- `mask`

其他运行信息如果以后需要，再单独讨论，不要现在一开始就把：

- 日志
- 警告
- 运行目录
- 调试信息
- 派生结果

全部塞进 `context`。

---

## 情况总表

| 情况 | 推荐对象 | 用户函数形式 |
| --- | --- | --- |
| 直接数学目标 | `Problem` | `objFunc(X)` |
| 直接目标 + 约束 | `Problem` | `objFunc(X)` + `conFunc(X)` |
| 目标约束必须一起算 | `Problem` | `evaluate(X)` |
| 内部跑模型但只关心最终分数 | `Problem` | `objFunc(X)` 或 `evaluate(X)` |
| 先仿真再算目标 | `ModelProblem` | `simFunc(X)` + `objFunc(X, context)` |
| 先仿真再算目标和约束 | `ModelProblem` | `simFunc(X)` + `objFunc(X, context)` + `conFunc(X, context)` |
| 需要统一控制仿真输出与目标约束 | `ModelProblem` | `simFunc(X)` + `evaluate(X, context)` |

---

## 示例一：普通单目标问题

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
    optType="min",
)
```

适用原因：

- 目标直接由 `X` 算出
- 不需要 `sim`

---

## 示例二：普通目标 + 约束问题

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
    optType="min",
)
```

---

## 示例三：内部有模拟，但仍然用 `Problem`

```python
import numpy as np

from UQPyL.problem import Problem


def runModel(X):
    X = np.atleast_2d(X)
    return np.sin(X[:, 0]) + X[:, 1] ** 2


def objFunc(X):
    sim = runModel(X)
    return sim.reshape(-1, 1)


problem = Problem(
    nInput=2,
    nObj=1,
    lb=0.0,
    ub=1.0,
    objFunc=objFunc,
)
```

适用原因：

- 虽然内部有模型计算
- 但 `sim` 不需要成为公共接口

---

## 示例四：仿真型单目标问题

```python
import numpy as np

from UQPyL.problem import ModelProblem

obs = np.array([[1.0], [2.0], [3.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 3, 1))
    sim[:, 0, 0] = X[:, 0] * 1.0
    sim[:, 1, 0] = X[:, 0] * 2.0
    sim[:, 2, 0] = X[:, 0] * 3.0
    return sim


def objFunc(X, context):
    diff = context.sim - context.obs[None, :, :]
    rmse = np.sqrt(np.mean(diff**2, axis=(1, 2)))
    return rmse.reshape(-1, 1)


problem = ModelProblem(
    nInput=1,
    nObj=1,
    lb=0.0,
    ub=2.0,
    simFunc=simFunc,
    objFunc=objFunc,
    obs=obs,
    optType="min",
)
```

适用原因：

- 目标依赖 `sim`
- 目标还要和 `obs` 对比

---

## 示例五：仿真型目标 + 约束问题

```python
import numpy as np

from UQPyL.problem import ModelProblem

obs = np.array([[1.0], [2.0], [3.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 3, 1))
    sim[:, 0, 0] = X[:, 0] * 1.0
    sim[:, 1, 0] = X[:, 0] * 2.0
    sim[:, 2, 0] = X[:, 0] * 3.0
    return sim


def objFunc(X, context):
    diff = context.sim - context.obs[None, :, :]
    mse = np.mean(diff**2, axis=(1, 2))
    return mse.reshape(-1, 1)


def conFunc(X, context):
    peak = np.max(context.sim, axis=(1, 2))
    return (peak - 5.0).reshape(-1, 1)


problem = ModelProblem(
    nInput=1,
    nObj=1,
    nCon=1,
    lb=0.0,
    ub=2.0,
    simFunc=simFunc,
    objFunc=objFunc,
    conFunc=conFunc,
    obs=obs,
    optType="min",
)
```

---

## 示例六：仿真型组合 `evaluate`

```python
import numpy as np

from UQPyL.problem import Eval, ModelProblem

obs = np.array([[1.0], [2.0], [3.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 3, 1))
    sim[:, :, 0] = X[:, 0:1] * np.array([[1.0, 2.0, 3.0]])
    return sim


def evaluate(X, context):
    diff = context.sim - context.obs[None, :, :]
    objs = np.sqrt(np.mean(diff**2, axis=(1, 2), keepdims=True))
    cons = (np.max(context.sim, axis=(1, 2)) - 5.0).reshape(-1, 1)
    return Eval(objs=objs, cons=cons, sim=context.sim)


problem = ModelProblem(
    nInput=1,
    nObj=1,
    nCon=1,
    lb=0.0,
    ub=2.0,
    simFunc=simFunc,
    evaluate=evaluate,
    obs=obs,
    optType="min",
)
```

适用原因：

- 目标和约束依赖同一组中间量
- 一起组织更自然

---

## 当前推荐结论

现阶段建议把规则定成下面这样：

### 对外规则

- `Problem`：无 `context`
- `ModelProblem`：有 `context`

### 对内理解

- 普通问题：`X -> objs/cons`
- 仿真问题：`X -> sim -> objs/cons`

### 选择标准

- 模拟结果不进入公共语义：`Problem`
- 模拟结果进入公共语义：`ModelProblem`

一句话总结：

**当前最稳的规则不是强行统一用户接口，而是明确承认两种问题模式，并把它们的函数签名、适用场景和边界定清楚。**
