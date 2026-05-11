# Problem API

## `UQPyL.problem`

`problem` 模块定义 UQPyL 的统一建模协议。采样、分析、优化、推断、校准和代理建模都会依赖 `Problem` 或 `ModelProblem`。

## 导入

```python
from UQPyL.problem import Problem, ModelProblem, Eval
```

## 公共对象

| 对象 | 作用 |
|---|---|
| `Problem` | 定义静态目标和可选约束问题。 |
| `ModelProblem` | 定义带观测和 mask 的仿真模型问题。 |
| `Eval` | `evaluate()` 的标准返回对象。 |
| `Space` | 定义连续、整数或离散变量的有界输入空间。 |
| `ProblemBase` / `ProblemABC` | 内置 benchmark problem 使用的基础接口。 |
| `singleFunc` | 将单样本目标函数适配为批量输入。 |
| `singleEval` | 将单样本 `Eval` 函数适配为批量输入。 |

## `Problem`

用于目标和约束能直接从输入变量计算出来的场景。

```text
Problem(
    nInput=None,
    nObj=None,
    ub=None,
    lb=None,
    objFunc=None,
    conFunc=None,
    evaluate=None,
    nCon=0,
    varType=None,
    varSet=None,
    optType="min",
    xLabels=None,
    name=None,
    space=None,
    objLabels=None,
    conLabels=None,
)
```

| 参数 | 含义 |
|---|---|
| `nInput` | 输入变量个数。未提供 `space` 时必填。 |
| `nObj` | 目标个数。 |
| `ub`, `lb` | 上下界，可以是标量、列表或数组。 |
| `objFunc` | 目标函数，接收批量 `X`，返回 `(n_samples, n_obj)`。 |
| `conFunc` | 约束函数，返回 `(n_samples, n_con)`。 |
| `evaluate` | 组合评估函数，必须返回 `Eval`。 |
| `nCon` | 约束个数，默认 `0`。 |
| `varType` | 变量类型：`0` 连续，`1` 整数，`2` 离散。 |
| `varSet` | 离散变量取值集合。 |
| `optType` | 优化方向，`"min"`、`"max"` 或每个目标一个方向。 |
| `xLabels`, `objLabels`, `conLabels` | 输入、目标、约束标签。 |
| `name` | 问题名称。 |
| `space` | 可选输入空间对象。 |

`Problem` 只能使用一种 callable 配置：

| 配置 | 使用场景 |
|---|---|
| `objFunc` | 只有目标。 |
| `objFunc` + `conFunc` | 目标和约束分开计算。 |
| `evaluate` | 目标和约束需要一起计算。 |

不要同时传 `evaluate` 和 `objFunc` / `conFunc`。

## 常用方法

| 方法 | 返回 | 含义 |
|---|---|---|
| `evaluate(X, target=None)` | `Eval` | 评估目标和约束。 |
| `objFunc(X)` | `np.ndarray` | 只评估目标。 |
| `conFunc(X)` | `np.ndarray` 或 `None` | 只评估约束。 |
| `validate(X)` | `np.ndarray` | 转成二维输入并检查维度。 |
| `unit_to_space(X)` | `np.ndarray` | 将 `[0, 1]` 空间样本映射到问题边界。 |
| `apply_var_type(X)` | `np.ndarray` | 应用整数和离散变量变换。 |

`target` 可选：

| 值 | 含义 |
|---|---|
| `None` | 返回所有可用输出。 |
| `"objs"` | 只返回目标。 |
| `"cons"` | 只返回约束。 |

## `Eval`

`Problem.evaluate()` 返回 `Eval`，不是字典。

| 字段 | 含义 |
|---|---|
| `objs` | 目标矩阵。 |
| `cons` | 约束矩阵，无约束时为 `None`。 |
| `sim` | 仿真输出，通常来自 `ModelProblem`。 |
| `extras` | 可选附加数据。 |

## `ModelProblem`

用于校准或仿真工作流。

| 参数 | 含义 |
|---|---|
| `nInput`, `ub`, `lb` | 参数维度和边界。 |
| `simFunc` | 批量仿真函数，返回 `(n_samples, n_time, n_series)`。 |
| `obs` | 观测矩阵，shape 为 `(n_time, n_series)`。 |
| `mask` | 与 `obs` 同形状，`True` 表示忽略该观测位置。 |
| `simLabels` | 仿真序列标签。 |
| `name` | 模型问题名称。 |

## 下一步

| 目标 | 阅读 |
|---|---|
| 用户指南 | [Problem](../problem.md) |
| 采样 | [DOE API](doe.md) |
| 校准 | [Calibration API](calibration.md) |
