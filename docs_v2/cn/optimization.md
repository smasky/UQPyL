# 优化

`optimization` 模块用于在一个 `Problem` 的输入空间中搜索更好的决策变量。

UQPyL 的优化算法可以先按三类理解：

| 类别 | 适用场景 | 导入路径 | 算法 |
|---|---|---|---|
| 单目标优化 | 只有一个目标，例如一个损失、成本、误差或评分。 | `UQPyL.optimization.soea` | `GA`, `PSO`, `DE`, `SCE_UA`, `ML_SCE_UA`, `CSA`, `ABC` |
| 多目标优化 | 有两个或更多目标，希望得到一组 Pareto 折中解。 | `UQPyL.optimization.moea` | `NSGAII`, `NSGAIII`, `MOEAD`, `RVEA` |
| 昂贵优化 | 每次真实模型评估都很贵，需要代理模型辅助搜索。 | `UQPyL.optimization.expensive` | `ASMO`, `EGO`, `MOASMO` |

核心流程始终是：

```text
define Problem -> choose algorithm -> algorithm.run(problem, seed=...) -> OptResult
```

## 先选对算法类别

先看问题类型，不要先纠结算法名字。

| 你的情况 | 建议先用 | 原因 |
|---|---|---|
| 一个目标，而且模型评估不算贵，可以跑几百到几千次。 | `GA` 或 `DE` | 通用的种群搜索算法，适合普通单目标问题。 |
| 一个目标，希望使用水文领域常见的 shuffled complex 优化器。 | `SCE_UA` | 常用于类似参数率定的连续参数搜索。 |
| 两个或多个目标，希望得到基础 Pareto 前沿。 | `NSGAII` | 多目标优化的常用默认选择。 |
| 目标很多，或需要参考方向/参考向量搜索。 | `NSGAIII` 或 `RVEA` | 更适合高维目标空间。 |
| 一个昂贵目标。 | `ASMO` 或 `EGO` | 用代理模型减少真实模型调用次数。 |
| 多个昂贵目标。 | `MOASMO` | 多目标代理辅助优化。 |

实用默认选择：第一次单目标优化用 `GA`，第一次多目标优化用 `NSGAII`，真实评估明显昂贵时再考虑 `ASMO`。

## 优化器需要什么

优化器需要一个定义完整的 `Problem`。

| `Problem` 字段 | 对优化的意义 |
|---|---|
| `nInput` | 决策变量个数。 |
| `lb`, `ub` | 算法搜索的下界和上界。 |
| `objFunc` | 目标函数。应接收批量 `X`，返回形状为 `(n_samples, n_obj)` 的二维数组。 |
| `optType` | `"min"`、`"max"`，或多目标时的列表，例如 `["min", "max"]`。 |
| `conFunc`, `nCon` | 可选约束。约束值 `<= 0` 表示可行。 |
| `varType`, `varSet` | 可选变量类型设置，用于整数变量或离散变量。 |

优化器内部按最小化处理。如果 `optType="max"`，UQPyL 会在搜索时转换目标方向，并在结果中用原始方向报告 `bestObjs`。

## 单目标优化

单目标算法返回一个当前最好的解。

这个例子最小化：

```text
f(x) = x1^2 + x2^2
```

```python
import numpy as np

from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="Sphere2D")
algorithm = GA(nPop=8, maxFEs=40, maxIters=5, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)

print(result.bestDecs)
print(result.bestObjs)
print(result.FEs, result.iters, result.bestFeasible)
```

Example output:

```text
[[0.003  0.1049]]
[[0.011]]
40 5 True
```

可以这样读结果：

| 输出 | 含义 |
|---|---|
| `bestDecs` | 找到的最佳决策变量。 |
| `bestObjs` | `bestDecs` 对应的目标函数值。 |
| `FEs` | 已使用的真实问题评估次数。 |
| `iters` | 已完成的优化迭代次数。 |
| `bestFeasible` | 最佳解是否满足约束。 |

这个测试问题的真实最优解是 `[0, 0]`，所以目标值接近 `0` 是合理的。

## 单目标算法怎么选

| 算法 | 适合的第一次使用场景 |
|---|---|
| `GA` | 通用单目标搜索，使用交叉和变异。 |
| `DE` | 连续变量搜索，使用差分变异。 |
| `PSO` | 连续变量的粒子群搜索。 |
| `SCE_UA` | shuffled complex evolution，常用于类似参数率定的连续优化问题。 |
| `ML_SCE_UA` | `SCE_UA` 的多层版本。 |
| `CSA` | Crow search algorithm。 |
| `ABC` | Artificial bee colony algorithm。 |

构造参数见 [Optimization API](api/optimization.md)。

## 多目标优化

多目标算法返回一组非支配解，而不是一个唯一最优解。

当目标之间存在冲突时使用多目标优化。例如，你可能希望成本低，同时可靠性高；这两个目标往往不能同时达到最好。

```python
import numpy as np

from UQPyL.optimization.moea import NSGAII
from UQPyL.problem import ZDT1

np.set_printoptions(precision=4, suppress=True)


problem = ZDT1(nInput=5)
algorithm = NSGAII(nPop=12, maxFEs=48, maxIters=3, verboseFlag=False, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)

print(result.bestDecs.shape)
print(result.bestObjs.shape)
print(result.bestObjs[:3])
print(round(result.bestMetric, 4), result.FEs, result.iters)
```

Example output:

```text
(8, 5)
(8, 2)
[[0.5078 1.8356]
 [0.5826 1.7025]
 [0.0076 2.3533]]
0.4891 48 4
```

可以这样读结果：

| 输出 | 含义 |
|---|---|
| `bestDecs.shape == (8, 5)` | 当前 Pareto 解集中有 8 个决策向量，每个向量有 5 个变量。 |
| `bestObjs.shape == (8, 2)` | 对应的 Pareto 目标矩阵有 8 行、2 个目标。 |
| `bestMetric` | 当前多目标进展指标，目前是 hypervolume。用于跟踪优化进展，通常越大越好。 |

不要直接把第一行当成“最优解”。Pareto 解集中的每一行都是一种折中方案，最终选择哪一行需要结合业务偏好、图形分析或额外决策规则。

## 多目标算法怎么选

| 算法 | 适合的第一次使用场景 |
|---|---|
| `NSGAII` | 默认的双目标或小规模多目标优化。 |
| `NSGAIII` | 多目标数量较多，或需要参考方向搜索。 |
| `MOEAD` | 基于分解的多目标搜索。 |
| `RVEA` | 参考向量引导的多目标搜索。 |

多目标的 `optType` 可以是列表：

```text
problem = Problem(nInput=3, nObj=2, lb=0.0, ub=1.0, objFunc=objFunc, optType=["min", "max"])
```

算法会在内部处理不同目标方向。

## 昂贵优化

昂贵优化适用于一次真实模型评估成本很高的问题。

| 昂贵评估例子 | 为什么昂贵优化有帮助 |
|---|---|
| 一次水文模拟需要几分钟。 | 避免直接调用真实模型几千次。 |
| CFD 或结构仿真。 | 用代理模型预测来提出更值得真实评估的点。 |
| 实验室实验或外部可执行程序。 | 减少真实实验或外部程序调用次数。 |

昂贵优化通常遵循这个模式：

```text
initial real samples -> fit surrogate -> optimize surrogate -> evaluate selected real point(s) -> repeat
```

只有当真实评估足够贵时，才值得使用这类算法。对于便宜的数学函数，普通的 `GA`、`DE` 或 `NSGAII` 通常更直接。

## 单目标昂贵优化

`ASMO` 和 `EGO` 是单目标昂贵优化器。

```python
import numpy as np

from UQPyL.optimization.expensive import ASMO
from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="Sphere2D")
algorithm = ASMO(nInit=6, optimizer=GA(nPop=6, maxFEs=18, maxIters=3, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False), maxFEs=12, maxIters=2, verboseFlag=False, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)

print(result.bestDecs)
print(result.bestObjs)
print(result.FEs, result.iters)
```

Example output:

```text
[[0.0981 0.1052]]
[[0.0207]]
9 4
```

`nInit=6` 表示算法先做 6 次真实评估作为初始数据。后续迭代会拟合代理模型，再选择新的真实评估点。

## 在昂贵优化中使用已有数据

如果你已经有评估过的样本，可以通过 `xInit` 和 `yInit` 传入。

```python
import numpy as np

from UQPyL.optimization.expensive import ASMO
from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="Sphere2D")
xInit = np.array([[0.8, 0.8], [0.2, 0.1], [-0.5, 0.3]])
yInit = objFunc(xInit)

algorithm = ASMO(nInit=6, optimizer=GA(nPop=6, maxFEs=18, maxIters=3, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False), maxFEs=8, maxIters=1, verboseFlag=False, logFlag=False, saveFlag=False)
result = algorithm.run(problem, xInit=xInit, yInit=yInit, seed=123)

print(yInit)
print(result.FEs)
print(result.bestObjs)
```

Example output:

```text
[[1.28]
 [0.05]
 [0.34]]
5
[[0.0207]]
```

如果 `xInit` 的行数少于 `nInit`，算法会自动补充初始样本。如果省略 `yInit`，UQPyL 会用真实 `Problem` 评估 `xInit`。

## 多目标昂贵优化

真实问题有多个昂贵目标时，使用 `MOASMO`。

```python
import numpy as np

from UQPyL.optimization.expensive import MOASMO
from UQPyL.optimization.moea import NSGAIII
from UQPyL.problem import ZDT1

np.set_printoptions(precision=4, suppress=True)


problem = ZDT1(nInput=6, ub=1.0, lb=0.0)
inner = NSGAIII(nPop=12, maxFEs=40, maxIters=2, verboseFlag=False, logFlag=False, saveFlag=False)
algorithm = MOASMO(optimizer=inner, pct=0.5, nInit=6, maxFEs=20, maxIters=2, verboseFlag=False, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)

print(result.bestDecs.shape)
print(result.bestObjs.shape)
print(result.bestObjs[:3])
```

Example output from a small run:

```text
(4, 6)
(4, 2)
[[0.     4.6318]
 [0.006  2.0188]
 [0.7385 1.1758]]
```

代理模型和样本预算会影响 Pareto 解集，示例中的小预算只用于展示流程。真正解释结果时，应使用更大的评估预算，并检查 Pareto 前沿是否稳定。

## 处理约束

带约束的 `Problem` 需要提供 `conFunc` 和 `nCon`。

约束符号约定是：

```text
constraint value <= 0 means feasible
constraint value > 0 means violation
```

```python
import numpy as np

from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


def conFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] + X[:, 1] - 0.5).reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, nCon=1, lb=-1.0, ub=1.0, objFunc=objFunc, conFunc=conFunc, optType="min", name="ConstrainedSphere2D")
algorithm = GA(nPop=8, maxFEs=40, maxIters=5, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)

print(result.bestDecs)
print(result.bestObjs)
print(result.bestCons)
print(result.bestFeasible)
```

Example output:

```text
[[0.044  0.0452]]
[[0.004]]
[[-0.4108]]
True
```

这里最佳约束值是负数，所以最佳解满足约束。

## 控制运行预算

最重要的预算参数有：

| 参数 | 含义 |
|---|---|
| `maxFEs` | 最大真实问题评估次数。通常是最重要的预算。 |
| `maxIters` | 最大算法迭代次数。 |
| `nPop` | 许多演化算法中的种群规模。 |
| `nInit` | 昂贵优化算法中的初始真实样本数。 |
| `maxTolerates` | 连续多次没有明显改进后停止。 |
| `tolerate` | 用于容忍停止的最小改进幅度。 |

普通优化算法中，`maxFEs` 统计直接调用 `Problem` 的真实评估次数。

昂贵优化算法中，`maxFEs` 仍然统计真实问题评估次数，不统计廉价的代理模型预测次数。

## 查看 verbose 过程

设置 `verboseFlag=True` 可以打印进度和最终摘要。

```python
import numpy as np

from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="Sphere2D")
algorithm = GA(nPop=8, maxFEs=24, maxIters=3, tolerate=None, verboseFlag=True, verboseFreq=1, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)
```

Example output:

```text
Algorithm: GA
Problem: Sphere2D
nInput: 2
nObj: 1
maxFEs: 24
maxIters: 3
GA | iter=0 eval=8 best=1.8884e-02 cv=0 time=0.0s
Optimization finished
  algorithm        : GA
  status           : finished
  iterations       : 3
  evaluations      : 24
  best value       : 1.1468e-02
  best X           : [3.0409e-03, 1.0704e-01]
  constraint viol. : 0
  elapsed          : 0.0s
```

调试和教学时可以打开 `verboseFlag=True`。批量实验或自动化脚本中，如果只需要 `OptResult`，通常设置为 `False`。

## 读取 `OptResult`

每次优化运行都会返回一个 `OptResult`。

| 字段 | 含义 |
|---|---|
| `bestDecs` | 最佳决策矩阵。多目标优化中，这是当前 Pareto 决策集。 |
| `bestObjs` | 最佳目标矩阵。多目标优化中，这是 Pareto 目标集。 |
| `bestCons` | `bestDecs` 对应的约束值，如果问题有约束。 |
| `bestFeasible` | 报告的最佳解或解集是否可行。 |
| `bestMetric` | 多目标进展指标，目前是 hypervolume。单目标中通常为 `None`。 |
| `FEs` | 最终真实问题评估次数。 |
| `iters` | 最终迭代次数。 |
| `appearFEs` | 最佳结果首次出现时的评估次数。 |
| `appearIters` | 最佳结果首次出现时的迭代次数。 |
| `history` | 优化过程历史快照。 |
| `summary()` | 紧凑的运行摘要字典。 |

保存到 sqlite 和 reader 方法见 [Optimization API](api/optimization.md)。

### 读取内存中的结果

在 `algorithm.run(...)` 之后，可以直接这样读：

```python
import numpy as np

from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="Sphere2D")
algorithm = GA(nPop=8, maxFEs=40, maxIters=5, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)

bestX = result.bestDecs[0]
bestY = float(result.bestObjs[0, 0])
recentHistory = result.history.bestObjHistory[-3:]

print(bestX)
print(bestY)
print(recentHistory)
print(result.summary()["fes"], result.summary()["iters"])
```

Example output:

```text
[0.003  0.1049]
0.011019868612514926
[0.01146775171564494, 0.01146775171564494, 0.011019868612514926]
40 5
```

单目标优化中，`result.bestDecs[0]` 是最佳决策向量，`result.bestObjs[0, 0]` 是对应目标值。多目标优化中，`result.bestDecs` 和 `result.bestObjs` 会包含多行 Pareto 解。

### 读取保存的 SQLite 结果

设置 `saveFlag=True` 后，优化结果会保存到 `Result/` 下的 sqlite 文件。

```python
from pathlib import Path

import numpy as np

from UQPyL.optimization.runtime import OptReader
from UQPyL.optimization.soea import GA
from UQPyL.problem import Sphere

np.set_printoptions(precision=4, suppress=True)


problem = Sphere(nInput=2, ub=1.0, lb=-1.0)

resultDir = Path("Result")
before = set(resultDir.glob("*.sqlite3")) if resultDir.exists() else set()

algorithm = GA(nPop=6, maxFEs=18, maxIters=3, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=True, saveFreq=2)
algorithm.run(problem, seed=123)

after = set(resultDir.glob("*.sqlite3"))
dbPath = sorted(after - before)[0]

reader = OptReader(dbPath)
summary = reader.get_run_summary()
params = reader.get_run_params()
snapshots = reader.list_snapshots()
best = reader.load_last_best()
reader.close()

print(dbPath.as_posix())
print(summary["method"], summary["problem_name"])
print(summary["final_fes"], summary["final_iters"])
print(params["seed"])
print(len(snapshots))
print(best.decs)
print(best.objs)
```

Example output:

```text
Result/ga_Sphere_20260510_1709_9d67.sqlite3
GA Sphere
18 3
123
3
[[ 0.1004 -0.1692]]
[[0.0387]]
```

如果你已经知道 sqlite 文件路径，只需要 reader 这部分：

```text
from UQPyL.optimization.runtime import OptReader

dbPath = "Result/ga_Sphere_20260510_1709_9d67.sqlite3"

reader = OptReader(dbPath)
summary = reader.get_run_summary()
best = reader.load_last_best()
reader.close()
```

`summary` 用于查看运行元信息；`best.decs`、`best.objs` 和 `best.cons` 用于查看最后一次保存的最佳种群或最佳解集。

保存的运行会包含序列化的问题对象。如果你在 notebook 单元格或临时脚本里临时定义 `objFunc`，Python 可能无法 pickle 它。需要长期保存 sqlite 结果时，建议使用可导入的问题类，或把目标函数定义在可导入的 Python 模块中。

## 常见错误

| 错误 | 会发生什么 | 修正方式 |
|---|---|---|
| 用单目标算法跑多目标问题 | 结果不是 Pareto 搜索结果。 | 使用 `NSGAII`、`NSGAIII`、`MOEAD`、`RVEA` 或 `MOASMO`。 |
| 把 Pareto 解集第一行当成最佳解 | 可能只是任意一个折中解。 | 画图或按领域偏好对 Pareto 解集排序。 |
| `maxFEs` 设置太小 | 算法还没充分搜索就停止。 | 小预算只用于流程检查，正式解释要增加预算。 |
| 忘记设置 `optType` | 目标方向可能错误。 | 设置 `"min"`、`"max"`，多目标时用列表。 |
| `conFunc` 符号写反 | 可行点和不可行点会被混淆。 | 让可行约束返回 `<= 0`。 |
| 对便宜函数使用昂贵优化 | 代理模型开销可能比真实评估还大。 | 便宜目标优先用普通 `GA`、`DE` 或 `NSGAII`。 |
| 传入 `xInit` 但 `yInit` 行数不匹配 | 已有评估数据无法正确复用。 | 保证 `xInit.shape[0] == yInit.shape[0]`，或省略 `yInit` 让 UQPyL 自动评估。 |

## 下一步

| 目标 | 阅读 |
|---|---|
| 定义目标和约束 | [Problem](problem.md) |
| 查看算法构造参数和结果字段 | [Optimization API](api/optimization.md) |
| 生成初始样本 | [试验设计](doe.md) |
| 训练代理模型 | [Surrogate Modeling](surrogate.md) |
| 查看完整工作流 | [Examples](examples.md) |
