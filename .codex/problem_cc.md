# Problem 模块

`problem` 模块用于将实际问题抽象为 UQPyL 可以统一调用的问题对象。抽象的核心在于将实际问题拆解为三个标准化的部分：**Space**（输入空间）、**Evaluation**（评估过程）、**Eval**（结果容器）。基于统一的描述协议，不同实际问题转化为统一的问题对象后可以无缝对接采样、优化、校准、分析等功能模块的任意算法，从而构建通用的工作流。

`Problem = Space + Evaluation + Eval`

## Space：输入空间

`Space` 描述输入变量的维度、范围与类型。若不显式传入 `space=`，`Problem` 和 `ModelProblem` 会根据构造参数自动生成。

### 基础定义

`nInput`、`lb`、`ub` 三者定义了输入空间的基本形态：

- `nInput`：输入变量的个数。
- `lb` / `ub`：各维度的下界与上界。传入标量时广播到所有维度，传入 list 或 array 时逐维指定。
- `xLabels`：输入变量的名称标签，默认自动生成 `['x_1', 'x_2', ...]`。

### 变量类型

`varType` 是一个长度为 `nInput` 的列表，逐维声明变量类型：

| 编码 | 类型 | 说明 |
|:----:|------|------|
| `0` | 连续 | 默认行为，在 `[lb, ub]` 内连续取值 |
| `1` | 整数 | 自动取整（`np.round`） |
| `2` | 离散 | 从 `varSet` 中按区间映射到预设的离散值 |

`varSet` 是一个 `dict`，键为离散变量所在的维度索引，值为该维度的合法取值列表。离散变量在区间 `[lb_i, ub_i]` 内被均分后映射到 `varSet[i]` 中的对应值。

```python
problem = Problem(
    nInput=3, nObj=1,
    lb=[0.0, 0.0, 0.0],
    ub=[1.0, 5.0, 1.0],
    varType=[0, 1, 2],                              # 连续 / 整数 / 离散
    varSet={2: [0.1, 0.3, 0.5, 0.7, 0.9]},          # 第 3 维的可选离散值，Python index习惯
    objFunc=objFunc,
)
```

### 自动变换方法

`Space` 提供一组方法，将采样点变换到合法空间。这些方法可通过 `Problem` / `ModelProblem` 实例直接调用（代理到内部 `self.space`）：

| 方法 | 作用 |
|------|------|
| `validate(X)` | 检查维度，确保为 `(nSamples, nInput)` 二维数组 |
| `cast_int_vars(X)` | 对整数类型变量执行取整 |
| `map_discrete_vars(X)` | 对离散类型变量映射到 `varSet` 中的取值 |
| `apply_var_type(X)` | 同时执行取整与离散映射 |
| `unit_to_space(X)` | 将 `[0, 1]` 单位超立方缩放到 `[lb, ub]`，并施加变量类型变换 |

典型用法——采样算法在单位超立方上生成点，再由 `unit_to_space` 变换到实际空间：

```python
X_unit = np.random.rand(100, problem.nInput)   # (100, nInput)，值域 [0, 1]
X_real = problem.unit_to_space(X_unit)          # 缩放 + 整数取整 + 离散映射
```

---

## Evaluation：评估过程

`Evaluation` 描述"给定输入后如何得到输出"。用户在定义问题时，通过传入一个或多个可调用对象来描述评估逻辑，框架则通过 `evaluate()` 统一入口在内部编排这些调用。

从输入参数评估过程看，实际问题大致分为两种情况。一种是基于输入参数直接计算目标与约束，中间不需要显式仿真环节；另一种则需要先运行仿真模型，再由仿真输出构造目标、约束或误差指标。为此，UQPyL 分别用 `Problem` 和 `ModelProblem` 来承载这两种模式。

### 直接评估：`Problem`

适用于目标与约束可以直接由输入计算的问题，如数学测试函数、黑箱评分等。评估过程无需独立的仿真环节。

#### objFunc

`objFunc` 是必须提供的可调用对象，描述从输入到目标值的映射：

| 项目 | 约定 |
|------|------|
| 输入 | `X`，形状 `(nSamples, nInput)` 的 `np.ndarray` |
| 输出 | `objs`，形状 `(nSamples, nObj)` 的 `np.ndarray` |
| 批量 | 始终按二维处理，推荐首行加 `X = np.atleast_2d(X)` |

框架会在 `evaluate()` 内部自动校验输出形状是否与 `nObj` 一致。

```python
def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)   # 沿样本维求和，keepdims 保持列向量形状 (nSamples, nObj)
```

#### conFunc

`conFunc` 为可选参数，描述约束函数。若提供，必须同时声明 `nCon`：

| 项目 | 约定 |
|------|------|
| 输入 | `X`，形状 `(nSamples, nInput)` |
| 输出 | `cons`，形状 `(nSamples, nCon)` |
| 约束判定 | 统一按 `cons <= 0` 视为可行。若实际约束为 `g(x) >= 0`，应在函数内取反 |

```python
def conFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] + X[:, 1] - 1.0).reshape(-1, 1)   # reshape 为列向量 (nSamples, nCon)
```

#### 完整示例

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
    nInput=2,              # 输入维度
    nObj=1,                # 目标维度
    nCon=1,                # 约束维度
    lb=0.0, ub=1.0,        # 输入上下界
    objFunc=objFunc,       # 目标函数（必需）
    conFunc=conFunc,       # 约束函数（可选，提供时需同时声明 nCon）
    optType='min',         # 优化方向：'min' / 'max' / ['min', 'max', ...]
    varType=[0, 0],        # 变量类型，默认全为连续
    xLabels=['x1', 'x2'],  # 输入标签
    name='MyProblem',
)

res = problem.evaluate([[0.2, 0.3], [0.5, 0.6]])
```

#### optType：优化方向

`optType` 声明每个目标的优化方向，框架内部自动转换为最小化问题。支持两种形式：

- 字符串：`'min'` 或 `'max'`，适用于所有目标方向相同的情况。
- 列表：`['min', 'max', 'min']`，逐目标指定，长度必须等于 `nObj`。

实例化后可通过 `problem.opt` 获取数值形式（`1` 表示最小化，`-1` 表示最大化）。

#### 自定义 Evaluator

当 `objFunc` 与 `conFunc` 的形式无法满足需求（如需预处理、缓存、外部进程调用等），可以继承 `Evaluator` 对象并实现 `evaluate(self, X, target)` 方法，将整个评估逻辑封装在一个类中。

`Evaluator.evaluate()` 的契约：

| 项目 | 约定 |
|------|------|
| 输入 | `X` 形状 `(nSamples, nInput)`，`target` 为 `None` / `"objs"` / `"cons"` |
| 输出 | 必须返回 `Eval` 实例，`target` 参数需透传以触发自动清理 |

```python
from UQPyL.problem import Eval, Evaluator, Problem

class MyEvaluator(Evaluator):
    def evaluate(self, X, target=None):
        X = np.atleast_2d(X)
        objs = np.sum(X**2, axis=1, keepdims=True)
        cons = (X[:, 0] + X[:, 1] - 1.0).reshape(-1, 1)
        return Eval(objs=objs, cons=cons, target=target)

problem = Problem(
    nInput=2, nObj=1, nCon=1,
    lb=0.0, ub=1.0,
    evaluator=MyEvaluator(),
)
```

> **注意**：`evaluator` 不能与 `objFunc` / `conFunc` 同时传入，二者互斥。

---

### 仿真评估：`ModelProblem`

适用于需要先运行仿真模型、再从仿真输出构造目标或约束的问题，如模型校准、时序仿真、过程模型等。

`ModelProblem` 的评估过程分为两步：先由 `simFunc` 执行仿真得到 `sims`，再由 `objFunc` / `conFunc` 从仿真结果计算目标与约束。这两步通过 `simContext` 衔接。

#### simFunc

`simFunc` 是 `ModelProblem` 必须提供的可调用对象，运行仿真模型：

| 项目 | 约定 |
|------|------|
| 输入 | `X`，形状 `(nSamples, nInput)` |
| 输出 | `sims`，形状 `(nSamples, nTime, nSeries)` 的三维数值数组 |
| NaN | 仿真输出原则上不允许包含 NaN。仅当提供了与 `obs` 形状一致的 `mask` 时，缺测标记位置允许为 NaN |

```python
def simFunc(X):
    X = np.atleast_2d(X)
    nSamples = X.shape[0]
    sims = np.zeros((nSamples, 3, 1))            # (nSamples, nTime=3, nSeries=1)
    sims[:, 0, 0] = X[:, 0]                      # t=0：第 1 个参数
    sims[:, 1, 0] = 0.5 * X[:, 0] + 0.5 * X[:, 1]  # t=1：两个参数的均值
    sims[:, 2, 0] = X[:, 1]                      # t=2：第 2 个参数
    return sims
```

#### objFunc（ModelProblem）

`ModelProblem` 的 `objFunc` 签名与 `Problem` 不同，即除了 `X` 外，额外接收 `simContext`：

| 项目 | 约定 |
|------|------|
| 输入 | `X` 形状 `(nSamples, nInput)`，`simContext` 为 `SimContext` 实例 |
| 输出 | `objs`，形状 `(nSamples, nObj)` |

`simContext` 由框架在 `simFunc` 执行后自动构造并传入，用户无需手动创建。

#### conFunc（ModelProblem）

与 `Problem` 的 `conFunc` 类似，但同样额外接收 `simContext`：

| 项目 | 约定 |
|------|------|
| 输入 | `X` 形状 `(nSamples, nInput)`，`simContext` 为 `SimContext` 实例 |
| 输出 | `cons`，形状 `(nSamples, nCon)` |
| 约束判定 | 统一按 `cons <= 0` 视为可行 |

#### SimContext

`SimContext` 是一个 frozen dataclass，在 `simFunc` 返回后由框架自动构造，随后传入 `objFunc` 和 `conFunc`：

```python
@dataclass(frozen=True)
class SimContext:
    sims: np.ndarray           # 仿真输出，形状 (nSamples, nTime, nSeries)
    obs: np.ndarray | None     # 观测数据，形状 (nTime, nSeries)
    mask: np.ndarray | None    # 缺测掩码，形状同 obs，True 表示该位置缺失
```

典型用法，如计算仿真与观测的均方误差，并排除缺测位置：

```python
def objFunc(X, simContext):
    err = simContext.sims - simContext.obs          # 逐元素误差
    if simContext.mask is not None:
        err = err[:, ~simContext.mask]              # 排除缺测位置
    return np.mean(err**2, axis=(1, 2)).reshape(-1, 1)  # 对时间和序列维取均值
```

#### obs 与 mask

- **`obs`**：观测矩阵，必须为二维 `(nTime, nSeries)`。例如在水文模型中，`nTime` 为时间步数，`nSeries` 为观测站点数。
- **`mask`**：缺测掩码，bool 型二维数组，形状必须与 `obs` 完全一致。`True` 表示对应位置数据缺失。提供了 `mask` 时，仿真输出在标记为缺失的位置允许为 NaN。

#### 完整示例

```python
import numpy as np
from UQPyL.problem import ModelProblem

obs = np.array([[1.0], [0.8], [2.0]])            # 观测数据 (nTime=3, nSeries=1)
mask = np.array([[False], [True], [False]])       # 第 2 个时间步缺失

def simFunc(X):
    X = np.atleast_2d(X)
    nSamples = X.shape[0]
    sims = np.zeros((nSamples, 3, 1))            # (nSamples, nTime=3, nSeries=1)
    sims[:, 0, 0] = X[:, 0]                      # t=0
    sims[:, 1, 0] = 0.5 * X[:, 0] + 0.5 * X[:, 1]  # t=1
    sims[:, 2, 0] = X[:, 1]                      # t=2
    return sims

def objFunc(X, simContext):
    err = simContext.sims - simContext.obs        # 逐元素误差
    if simContext.mask is not None:
        err = err[:, ~simContext.mask]            # 排除缺测位置
    return np.mean(err**2, axis=(1, 2)).reshape(-1, 1)  # 对时间和序列维取均值

problem = ModelProblem(
    nInput=2, nObj=1,
    lb=0.0, ub=3.0,
    simFunc=simFunc,             # 仿真函数（必需）
    objFunc=objFunc,             # 目标函数
    obs=obs,                     # 观测数据
    mask=mask,                   # 缺测掩码
    seriesLabels=['Q'],          # 序列标签
    name='MyModel',
)

res = problem.evaluate([[0.5, 1.5]])
```

#### simulate()：仅运行仿真

若只需要仿真输出而不需要计算目标或约束，可以使用 `simulate()` 方法：

```python
simContext = problem.simulate(X)                   # 返回 SimContext，可访问 .obs / .mask
sims = problem.evaluate(X, target="sims").sims     # 返回 Eval，仅包含 .sims
```

两者的区别：`simulate()` 返回完整的 `SimContext`，可直接访问观测数据与掩码；`evaluate(X, target="sims")` 返回 `Eval`，其中仅有 `sims` 字段。

#### 自定义 ModelEvaluator

继承 `ModelEvaluator` 可实现更复杂的后处理逻辑。`ModelEvaluator.evaluate()` 的契约：

| 项目 | 约定 |
|------|------|
| 输入 | `X`、`simContext`、`target` |
| 输出 | 必须返回 `Eval` 实例 |

```python
from UQPyL.problem import Eval, ModelEvaluator, ModelProblem

class MyModelEvaluator(ModelEvaluator):
    def evaluate(self, X, simContext, target=None):
        err = simContext.sims - simContext.obs          # 逐元素误差
        if simContext.mask is not None:
            err = err[:, ~simContext.mask]              # 排除缺测位置
        objs = np.mean(err**2, axis=(1, 2)).reshape(-1, 1)  # 对时间和序列维取均值
        return Eval(objs=objs, sims=simContext.sims, target=target)

problem = ModelProblem(
    nInput=2, nObj=1,
    lb=0.0, ub=3.0,
    simFunc=simFunc,
    obs=obs,
    evaluator=MyModelEvaluator(),
)
```

---

## Eval 与 evaluate()

`Eval` 是评估结果的统一容器，`evaluate()` 是所有下游模块访问问题的唯一入口。

### evaluate()：统一入口

用户在定义问题时传入 `objFunc`、`conFunc`、`simFunc` 等可调用对象来描述评估逻辑，但下游模块（优化器、分析器、校准器等）并不直接调用这些函数，而是通过 `evaluate()` 这一统一入口来访问问题。

`evaluate()` 内部的工作流程为：

```text
校验输入维度 (validate) → 调用用户定义函数 → 校验输出形状 → 封装为 Eval 返回
```

对 `Problem` 而言，`evaluate()` 直接调用 `objFunc` / `conFunc`；对 `ModelProblem` 而言，`evaluate()` 先调用 `simFunc` 获得仿真输出，构造 `simContext`，再将其传入 `objFunc` / `conFunc`。

### Eval 字段

```python
@dataclass
class Eval:
    objs: np.ndarray | None = None   # 目标值 (nSamples, nObj)
    cons: np.ndarray | None = None   # 约束值 (nSamples, nCon)
    sims: np.ndarray | None = None   # 仿真输出 (nSamples, nTime, nSeries)
```

**形状规则**：

| 字段 | 形状 | 要求 |
|------|------|------|
| `objs` | `(nSamples, nObj)` | 二维数值数组 |
| `cons` | `(nSamples, nCon)` | 二维数值数组 |
| `sims` | `(nSamples, nTime, nSeries)` | 三维数值数组，不含 NaN（缺测位置除外） |

**空值规则**：未提供的输出块必须为 `None`，禁止以空数组代替。

`Eval` 实例上的便捷属性：

```python
res = problem.evaluate(X)
print(res.objs)        # np.ndarray 或 None
print(res.hasObjs)     # bool，等价于 res.objs is not None
print(res.hasCons)     # bool
print(res.hasSims)     # bool
```

### target：按需返回

`evaluate(X, target=None)` 中的 `target` 参数控制本次调用需要返回哪些输出块。不请求的输出块会被强制置为 `None`，下游模块可借此跳过不关心的计算。

两类对象支持的 `target` 值：

| target | 含义 | Problem | ModelProblem |
|--------|------|:-------:|:------------:|
| `None` | 返回所有可用输出块 | ✅ | ✅ |
| `"objs"` | 仅返回目标值，其余为 `None` | ✅ | ✅ |
| `"cons"` | 仅返回约束值，其余为 `None` | ✅ | ✅ |
| `"sims"` | 仅返回仿真输出，其余为 `None` | ❌ | ✅ |

两类对象的详细行为差异：

**Problem**：

| target | objs | cons | 额外校验 |
|--------|:----:|:----:|------|
| `None` | 必返回 | nCon > 0 时返回 | — |
| `"objs"` | ✅ | 强制为 None | 若 cons 非空则报错 |
| `"cons"` | 强制为 None | ✅ | 若 objs 非空则报错 |

**ModelProblem**：

| target | objs | cons | sims | 额外校验 |
|--------|:----:|:----:|:----:|------|
| `None` | 有 objFunc 时返回 | 有 conFunc 时返回 | 必返回 | — |
| `"objs"` | ✅ | 强制为 None | 强制为 None | 若 objs 为空则报错 |
| `"cons"` | 强制为 None | ✅ | 强制为 None | 若 nCon>0 且 cons 为空则报错 |
| `"sims"` | 强制为 None | 强制为 None | ✅ | 若 objs 或 cons 非空则报错 |

> **核心约束**：仿真阶段校验 `sims`；最终仅在 `target=None` 或 `target="sims"` 时返回 `sims`，`"objs"` / `"cons"` 仅返回所请求字段。

`target` 的使用示例：

```python
# 假设 problem 是一个 Problem 实例，nObj=1, nCon=1

# target=None：返回所有输出
res = problem.evaluate(X)
print(res.objs)   # (nSamples, 1)
print(res.cons)   # (nSamples, 1)

# target="objs"：仅计算目标，cons 强制为 None
res = problem.evaluate(X, target="objs")
print(res.objs)   # (nSamples, 1)
print(res.cons)   # None

# target="cons"：仅计算约束，objs 强制为 None
res = problem.evaluate(X, target="cons")
print(res.objs)   # None
print(res.cons)   # (nSamples, 1)
```

对于 `ModelProblem`：

```python
# target="sims"：仅返回仿真输出，跳过 objFunc / conFunc 的计算
res = problem.evaluate(X, target="sims")
print(res.sims)   # (nSamples, nTime, nSeries)
print(res.objs)   # None
print(res.cons)   # None
```

---

## 内置测试问题

`UQPyL.problem` 提供一系列经典测试函数，可直接实例化使用。

### 单目标 (SOP)

```python
from UQPyL.problem import Sphere, Rosenbrock, Ackley, Griewank, Rastrigin

problem = Sphere(nInput=10)
problem = Rosenbrock(nInput=5)
```

完整列表：`Sphere`, `Schwefel_2_22`, `Schwefel_1_22`, `Schwefel_2_21`, `Rosenbrock`, `Step`, `Quartic`, `Schwefel_2_26`, `Rastrigin`, `Ackley`, `Griewank`, `Trid`, `Bent_Cigar`, `Discus`, `Weierstrass`, `RosenbrockWithCon`

### 多目标 (MOP)

```python
from UQPyL.problem import ZDT1, ZDT2, DTLZ1, DTLZ2

problem = ZDT1(nInput=30)
problem = DTLZ2(nInput=12, nObj=3)
```

ZDT 族：`ZDT1`, `ZDT2`, `ZDT3`, `ZDT4`, `ZDT6`
DTLZ 族：`DTLZ1`—`DTLZ7`

---

## singleFunc 装饰器

`singleFunc` 将只处理单个样本的函数包装为批量函数，省去用户手动处理 `np.atleast_2d` 和维度变换：

```python
from UQPyL.problem import singleFunc, Problem

@singleFunc
def objFunc(x):                        # x: (nInput,)  单样本
    return x[0]**2 + x[1]**2           # 返回标量

problem = Problem(nInput=2, nObj=1, lb=-1, ub=1, objFunc=objFunc)
res = problem.evaluate([[0.2, 0.3], [0.5, 0.6]])   # 批量调用正常工作
```

等价于手动编写：

```python
def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)
```

---

## 参数速查

### Problem 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `nInput` | `int` | — | 输入变量个数 |
| `nObj` | `int` | — | 目标函数个数（必填） |
| `lb` | `int/float/list/array` | — | 输入下界 |
| `ub` | `int/float/list/array` | — | 输入上界 |
| `objFunc` | `callable` | `None` | 目标函数 |
| `conFunc` | `callable` | `None` | 约束函数 |
| `nCon` | `int` | `0` | 约束个数 |
| `optType` | `str/list` | `'min'` | 优化方向 |
| `conWgt` | `list` | `None` | 约束权重，形状 `(1, nCon)` |
| `varType` | `list` | 全连续 | 变量类型：`0`=连续, `1`=整数, `2`=离散 |
| `varSet` | `dict` | `None` | 离散变量取值，如 `{2: [0.1, 0.3, 0.5]}` |
| `xLabels` | `list` | 自动生成 | 输入变量标签 |
| `objLabels` | `list` | 自动生成 | 目标标签 |
| `conLabels` | `list` | 自动生成 | 约束标签 |
| `name` | `str` | 类名 | 问题名称 |
| `space` | `SpaceBase` | 自动生成 | 自定义输入空间 |
| `evaluator` | `EvaluatorBase` | 自动生成 | 自定义评估器 |

### ModelProblem 额外参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `simFunc` | `callable` | — | 仿真函数（必填） |
| `obs` | `np.ndarray` | `None` | 观测矩阵 `(nTime, nSeries)` |
| `mask` | `np.ndarray` | `None` | 缺测掩码，形状同 obs |
| `seriesLabels` | `list` | 自动生成 | 序列标签 |
| `evaluator` | `ModelEvaluatorBase` | 自动生成 | 自定义仿真评估器 |

### ProblemBase 实例属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `nInput` | `int` | 输入维度 |
| `nObj` | `int` | 目标维度 |
| `nCon` | `int` | 约束维度 |
| `lb` | `np.ndarray` | 下界 `(1, nInput)` |
| `ub` | `np.ndarray` | 上界 `(1, nInput)` |
| `optType` | `str` | 优化方向字符串 |
| `opt` | `int/array` | `1`=最小化, `-1`=最大化 |
| `varType` | `np.ndarray` | 变量类型编码 |
| `idxF` | `np.ndarray` | 连续变量索引 |
| `idxI` | `np.ndarray` | 整数变量索引 |
| `idxD` | `np.ndarray` | 离散变量索引 |
| `varSet` | `dict` | 离散变量取值 |
| `xLabels` | `list` | 输入变量标签 |
| `objLabels` | `list` | 目标标签 |
| `conLabels` | `list` | 约束标签（无约束时为 `None`） |
| `conWgt` | `np.ndarray` | 约束权重 |

### ModelProblem 额外属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `obs` | `np.ndarray` | 观测矩阵 |
| `mask` | `np.ndarray` | 缺测掩码 |
| `obsShape` | `tuple` | `obs.shape` |
| `nObs` | `int` | 展平后观测总长度 |
| `seriesLabels` | `list` | 序列标签 |
