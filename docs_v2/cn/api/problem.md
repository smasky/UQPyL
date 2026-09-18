# Problem API

## `UQPyL.problem`

`UQPyL.problem` 定义 UQPyL 的统一问题协议。所有采样、优化、分析、推断、校准与代理建模流程，最终都通过 `Problem` 或 `ModelProblem` 消费问题对象。

## 导入

```python
from UQPyL.problem import (
    Problem,
    ModelProblem,
    Eval,
    Space,
    Evaluator,
    ModelEvaluator,
)
```

## 公共对象

| 对象 | 作用 |
|---|---|
| `ProblemBase` / `ProblemABC` | 问题对象共享的协议基类 |
| `Problem` | 直接评估型问题对象 |
| `ModelProblem` | 仿真后评估型问题对象 |
| `Eval` | `evaluate()` 的统一返回对象 |
| `Space` / `SpaceBase` | 输入空间对象及其基类 |
| `EvaluatorBase` / `Evaluator` | `Problem` 使用的 evaluator 接口与默认实现 |
| `ModelEvaluatorBase` / `ModelEvaluator` | `ModelProblem` 使用的 evaluator 接口与默认实现 |
| `SimulatorBase` | 高级仿真器接口 |
| `singleFunc` | 单样本目标函数到批量目标函数的适配装饰器 |

## 共享协议

### 批量输入约定

所有正式评估接口都按批量输入设计。标准输入形状为：

```text
(n_samples, n_input)
```

### 统一评估入口

| 对象 | 方法签名 |
|---|---|
| `Problem` | `evaluate(X, target=None)` |
| `ModelProblem` | `evaluate(X, target=None)` |
| `Evaluator` | `evaluate(X, target=None)` |
| `ModelEvaluator` | `evaluate(X, simContext, target=None)` |

### 统一返回协议：`Eval`

```python
Eval(objs=None, cons=None, sims=None, target=None)
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `objs` | `np.ndarray` 或 `None` | 目标矩阵 |
| `cons` | `np.ndarray` 或 `None` | 约束矩阵 |
| `sims` | `np.ndarray` 或 `None` | 仿真输出 |
| `target` | `str` 或 `None` | 用于构造时过滤结果字段 |

#### `Eval` 的 `target` 规则

| `target` | 保留字段 | 过滤字段 |
|---|---|---|
| `None` | 已提供的全部字段 | 无 |
| `"objs"` | `objs` | `cons`、`sims` |
| `"cons"` | `cons` | `objs`、`sims` |
| `"sims"` | `sims` | `objs`、`cons` |

#### `Eval` 的只读语义

`Eval` 在设计上应被视为一次评估的结果对象。推荐的构造方式是：

```python
return Eval(objs=objs, cons=cons, sims=sims, target=target)
```

推荐直接在构造阶段完成字段筛选，而不再通过后续赋值修改结果对象。

## `Space`

### 构造签名

```python
Space(
    nInput,
    ub,
    lb,
    varType=None,
    varSet=None,
    xLabels=None,
)
```

### 参数

| 参数 | 类型 | 含义 |
|---|---|---|
| `nInput` | `int` | 输入变量个数 |
| `ub` | `int` / `float` / `list` / `np.ndarray` | 上界 |
| `lb` | `int` / `float` / `list` / `np.ndarray` | 下界 |
| `varType` | `list` 或 `None` | 变量类型编码：`0` 连续，`1` 整数，`2` 离散 |
| `varSet` | `dict` 或 `None` | 离散变量的取值集合，键为变量索引 |
| `xLabels` | `list` 或 `None` | 输入变量标签 |

### 属性

| 属性 | 含义 |
|---|---|
| `nInput` | 输入变量个数 |
| `lb`, `ub` | 形状为 `(1, nInput)` 的上下界矩阵 |
| `varType` | 变量类型向量 |
| `idxF`, `idxI`, `idxD` | 连续、整数、离散变量索引 |
| `varSet` | 离散变量取值集合 |
| `xLabels` | 输入变量标签 |
| `encoding` | `"real"` 或 `"mix"` |

### 常用方法

| 方法 | 返回 | 含义 |
|---|---|---|
| `validate(X)` | `np.ndarray` | 将输入转为二维并校验维度 |
| `transform(X)` | `np.ndarray` | 应用空间定义后的标准输入 |
| `unit_to_space(X, IFlag=True, DFlag=True)` | `np.ndarray` | 将单位空间样本映射到真实空间 |
| `apply_var_type(X, IFlag=True, DFlag=True)` | `np.ndarray` | 应用整数与离散变量变换 |
| `cast_int_vars(X)` | `np.ndarray` | 对整数变量取整 |
| `map_discrete_vars(X)` | `np.ndarray` | 将离散变量映射到真实取值 |

## `Problem`

### 构造签名

```python
Problem(
    nInput=None,
    nObj=None,
    ub=None,
    lb=None,
    objFunc=None,
    conFunc=None,
    conWgt=None,
    nCon=0,
    varType=None,
    varSet=None,
    optType="min",
    xLabels=None,
    name=None,
    space=None,
    objLabels=None,
    conLabels=None,
    evaluator=None,
)
```

### 参数

| 参数 | 类型 | 含义 |
|---|---|---|
| `nInput` | `int` 或 `None` | 输入变量个数。未提供 `space` 时通常必填 |
| `nObj` | `int` | 目标个数 |
| `ub`, `lb` | 标量 / 列表 / 数组 | 输入变量上下界 |
| `objFunc` | callable 或 `None` | 目标函数，签名为 `objFunc(X)` |
| `conFunc` | callable 或 `None` | 约束函数，签名为 `conFunc(X)` |
| `conWgt` | `list` 或 `None` | 约束权重 |
| `nCon` | `int` | 约束个数，默认 `0` |
| `varType` | `list` 或 `None` | 变量类型编码 |
| `varSet` | `dict` 或 `None` | 离散变量取值集合 |
| `optType` | `str` / `list` / `None` | 优化方向，支持 `"min"`、`"max"` 或逐目标列表 |
| `xLabels` | `list` 或 `None` | 输入变量标签 |
| `name` | `str` 或 `None` | 问题名称 |
| `space` | `SpaceBase` 或 `None` | 显式输入空间对象 |
| `objLabels` | `list` 或 `None` | 目标标签 |
| `conLabels` | `list` 或 `None` | 约束标签 |
| `evaluator` | `EvaluatorBase` 或 `None` | 高级 evaluator 实例 |

### 语义规则

1. `Problem` 至少需要定义 `objFunc`，或提供一个 `evaluator`
2. `conFunc` 不能脱离 `objFunc` 单独存在
3. `evaluator` 不能与 `objFunc` / `conFunc` 同时使用
4. `Problem.evaluate()` 的正式返回值不得包含 `sims`

### 标签与默认值

| 字段 | 默认行为 |
|---|---|
| `xLabels` | 未提供时自动生成 `x_1, ..., x_n` |
| `objLabels` | 未提供时自动生成 `obj_1, ..., obj_n` |
| `conLabels` | 当 `nCon > 0` 且未提供时自动生成 `con_1, ..., con_n` |

### 推荐定义层次

1. `objFunc`
2. `objFunc + conFunc`
3. 自定义 `Evaluator`

### 常用方法

| 方法 | 返回 | 含义 |
|---|---|---|
| `evaluate(X, target=None)` | `Eval` | 正式评估入口 |
| `objFunc(X)` | `np.ndarray` | 调用目标函数 |
| `conFunc(X)` | `np.ndarray` / `None` | 调用约束函数 |
| `validate(X)` | `np.ndarray` | 输入校验 |
| `unit_to_space(X, IFlag=True, DFlag=True)` | `np.ndarray` | 空间映射 |
| `apply_var_type(X, IFlag=True, DFlag=True)` | `np.ndarray` | 应用变量类型变换 |
| `cast_int_vars(X)` | `np.ndarray` | 整数变量取整 |
| `map_discrete_vars(X)` | `np.ndarray` | 离散变量映射 |
| `getOptimum()` | 依实现而定 | 返回已知最优解 |

### `target`

| 取值 | 含义 |
|---|---|
| `None` | 返回所有可用字段 |
| `"objs"` | 只返回目标 |
| `"cons"` | 只返回约束 |

`evaluate` 会先检查 `target`，将一维单样本规整为二维数组，再执行评估并校验 `Eval`；覆写 `evaluate` 的子类也遵循这一规则。三维及更高维输入会被拒绝。普通 `Problem` 在 `target=None` 或 `"objs"` 时必须返回 `objs`；两类问题在 `nCon > 0` 且请求全部结果或 `"cons"` 时必须返回 `cons`。无约束问题单独请求 `"cons"` 可以返回空字段。纯模拟 `ModelProblem` 的全部结果允许只有 `sims`，显式请求 `"objs"` 时才要求目标值存在。

### 示例

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
    xLabels=["x1", "x2"],
    objLabels=["f"],
)

res = problem.evaluate([[0.2, 0.3]])
print(res.objs)
```

## `Evaluator`

### 构造签名

```python
Evaluator(objFunc=None, conFunc=None)
```

### 方法签名

```python
evaluate(X, target=None)
```

### 语义

- `Evaluator` 用于 `Problem`
- 若仅需标准行为，可直接传 `objFunc` / `conFunc`
- 若需要复杂评估控制，则继承 `Evaluator` 并重载 `evaluate()`

### 示例

```python
import numpy as np

from UQPyL.problem import Eval, Evaluator


class QuadraticEvaluator(Evaluator):
    def evaluate(self, X, target=None):
        X = np.atleast_2d(X)
        objs = np.sum(X**2, axis=1, keepdims=True)
        cons = (X[:, 0] + X[:, 1] - 1.0).reshape(-1, 1)
        return Eval(objs=objs, cons=cons, target=target)
```

## `ModelProblem`

### 构造签名

```python
ModelProblem(
    nInput=None,
    nObj=1,
    ub=None,
    lb=None,
    simFunc=None,
    objFunc=None,
    conFunc=None,
    obs=None,
    mask=None,
    conWgt=None,
    nCon=0,
    varType=None,
    varSet=None,
    optType="min",
    xLabels=None,
    name=None,
    space=None,
    objLabels=None,
    conLabels=None,
    evaluator=None,
    seriesLabels=None,
)
```

### 参数

| 参数 | 类型 | 含义 |
|---|---|---|
| `nInput` | `int` 或 `None` | 输入变量个数 |
| `nObj` | `int` | 目标个数，默认 `1` |
| `ub`, `lb` | 标量 / 列表 / 数组 | 输入变量上下界 |
| `simFunc` | callable | 仿真函数，签名为 `simFunc(X)` |
| `objFunc` | callable 或 `None` | 目标函数，签名为 `objFunc(X, simContext)` |
| `conFunc` | callable 或 `None` | 约束函数，签名为 `conFunc(X, simContext)` |
| `obs` | `np.ndarray` 或 `None` | 观测矩阵，形状为 `(n_time, n_series)` |
| `mask` | `np.ndarray` 或 `None` | 观测掩码，形状需与 `obs` 一致 |
| `conWgt` | `list` 或 `None` | 约束权重 |
| `nCon` | `int` | 约束个数 |
| `varType` | `list` 或 `None` | 变量类型编码 |
| `varSet` | `dict` 或 `None` | 离散变量取值集合 |
| `optType` | `str` / `list` / `None` | 优化方向 |
| `xLabels` | `list` 或 `None` | 输入变量标签 |
| `name` | `str` 或 `None` | 问题名称 |
| `space` | `SpaceBase` 或 `None` | 显式输入空间对象 |
| `objLabels` | `list` 或 `None` | 目标标签 |
| `conLabels` | `list` 或 `None` | 约束标签 |
| `evaluator` | `ModelEvaluatorBase` 或 `None` | 高级 model evaluator 实例 |
| `seriesLabels` | `list` 或 `None` | 仿真序列标签 |

### 语义规则

1. `ModelProblem` 必须定义 `simFunc`
2. 若定义 `conFunc`，则通常也应定义 `objFunc`
3. `evaluator` 不能与 `objFunc` / `conFunc` 同时使用
4. `ModelProblem.evaluate()` 在 `target=None` 或 `target="sims"` 时必须返回 `sims`；`target="objs"` / `target="cons"` 时仅返回所请求字段，`sims` 必须为 `None`
5. `mask` 只能在提供 `obs` 时使用

### simulation-only 与 full model 两种形态

| 形态 | 最小配置 | 适用场景 |
|---|---|---|
| simulation-only | `simFunc` | 仅消费仿真输出的流程 |
| full model | `simFunc + objFunc` | 需要正式目标值的优化、分析、推断或校准流程 |

### `simContext`

`ModelProblem` 的仿真上下文对象为 `SimContext`，当前字段如下：

| 字段 | 类型 | 含义 |
|---|---|---|
| `sims` | `np.ndarray` | 仿真输出 |
| `obs` | `np.ndarray` 或 `None` | 观测数据 |
| `mask` | `np.ndarray` 或 `None` | 观测掩码 |

### 常用方法

| 方法 | 返回 | 含义 |
|---|---|---|
| `evaluate(X, target=None)` | `Eval` | 正式评估入口 |
| `simulate(X)` | `SimContext` | 仅运行仿真并返回上下文 |
| `simFunc(X)` | `np.ndarray` | 调用仿真函数并校验输出 |
| `objFunc(X, simContext)` | `np.ndarray` | 调用目标函数 |
| `conFunc(X, simContext)` | `np.ndarray` / `None` | 调用约束函数 |
| `flattenSim(sim)` | `np.ndarray` | 将仿真输出展平为 `(n_samples, n_obs)` |
| `flattenObs()` | `np.ndarray` | 将观测展平为 `(n_obs,)` |
| `flattenMask()` | `np.ndarray` | 将掩码展平为 `(n_obs,)` |

### `target`

| 取值 | 含义 |
|---|---|
| `None` | 返回所有可用字段 |
| `"objs"` | 只返回目标 |
| `"cons"` | 只返回约束 |
| `"sims"` | 只返回仿真输出 |

### 示例

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

## `ModelEvaluator`

### 构造签名

```python
ModelEvaluator(objFunc=None, conFunc=None)
```

### 方法签名

```python
evaluate(X, simContext, target=None)
```

### 语义

- `ModelEvaluator` 用于 `ModelProblem`
- 其职责是消费 `simContext`，并构造 `Eval`
- 返回结果按 `target` 筛选：`None` 或 `"sims"` 时保留 `sims`，`"objs"` / `"cons"` 时将 `sims` 置为 `None`

### 示例

```python
import numpy as np

from UQPyL.problem import Eval, ModelEvaluator


class MSEEvaluator(ModelEvaluator):
    def evaluate(self, X, simContext, target=None):
        err = simContext.sims - simContext.obs
        objs = np.mean(err**2, axis=(1, 2)).reshape(-1, 1)
        return Eval(objs=objs, sims=simContext.sims, target=target)
```

## `singleFunc`

`singleFunc` 用于将按单样本编写的目标函数适配为批量目标函数。

### 示例

```python
from UQPyL.problem import singleFunc


@singleFunc
def objFunc(x):
    return x[0] ** 2 + x[1] ** 2
```

## 下一步

| 目标 | 阅读 |
|---|---|
| 设计语义与抽象说明 | [Problem](../problem.md) |
| DOE 采样流程 | [DOE API](doe.md) |
| 校准流程 | [Calibration API](calibration.md) |


## 单位区间与真实值转换

| 接口 | 含义 |
|---|---|
| `problem.unit_to_space(U)` | 解码单位区间样本，返回真实值副本。 |
| `problem.space_to_unit(X)` | 编码真实值，返回单位区间副本。 |
| `problem.canonicalize_unit(U)` | 将等价整数/离散编码归并到唯一代表点，返回单位区间副本。 |

转换由 `Space` 实现，`Problem` 提供统一入口。连续变量按有限上下界线性转换；固定连续变量编码为 `0.5`。整数变量在 `ceil(lb)` 到 `floor(ub)` 的合法整数间等宽分段，离散变量按 `varSet` 顺序等宽分段，编码时返回区间中点；`U=1` 对应最后一个值。这也意味着整数采样采用等宽整数区间，而非先线性缩放再取整。

离散 `varSet` 当前要求互不重复的有限数值；离散真实取值由 `varSet` 决定，不必落在该列用于旧编码的 `lb/ub` 内。真实值编码时检查边界、整数合法性和候选成员关系。默认解码与编码满足 `unit_to_space(space_to_unit(X)) == X`（浮点误差范围内）；反向仅得到规范代表点，不保证恢复原始整数/离散编码。

`evaluate(X)` 接收真实值，不会猜测是否为单位区间编码。旧的 `apply_var_type()` 仍供已有路径使用，但不再用于优化评估入口。

### 约束权重

`Problem(conWgt=[10, 1], nCon=2, ...)` 为每个约束指定一个有限非负权重，长度必须等于 `nCon`。`None` 表示不加权；零权重表示忽略该约束，包括其可行性判断。

```python
CV = np.sum(np.maximum(0, cons * conWgt), axis=1)
```


`varType` 必须是一维向量，长度等于 `nInput`，各项只能取整数值 0、1、2；非法值在转换前拒绝，不会截断小数。旧 `_transform_*` 包装已删除，请使用 `unit_to_space`、`apply_var_type`、`cast_int_vars`、`map_discrete_vars`。独立目标/约束回调按 `target` 执行，默认完整评价仍执行两者。
