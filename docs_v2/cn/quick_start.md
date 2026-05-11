# Quick Start

这一页展示 UQPyL 最短、最完整的上手路径。

大多数工作流都遵循同一个形状：

```text
Problem / ModelProblem -> Method -> Result
```

如果你想先确认 UQPyL 已正确安装，并快速理解主要模块是怎么串起来的，就从这里开始。

## 0. 安装 UQPyL

UQPyL 支持 Python 3.8 及以上版本。推荐先在你的项目环境里安装：

```bash
python -m pip install UQPyL
```

如果你需要运行带图形输出的示例，安装可视化依赖：

```bash
python -m pip install "UQPyL[viz]"
```

如果你正在本地开发 UQPyL，或者想直接使用仓库里的最新代码：

```bash
git clone https://github.com/smasky/UQPyL.git
cd UQPyL
python -m pip install -e .
```

源码安装时可能需要编译扩展模块。若编译失败，先确认当前环境有 `Cython`、`pybind11`、`numpy`、`scipy` 和可用的 C/C++ 编译工具链。

安装后可以先运行这段最小验证：

```python
import UQPyL
from UQPyL.problem import Problem
from UQPyL.doe import LHS

print("UQPyL is ready")
```

Example output:

```text
UQPyL is ready
```

常见安装问题：

| 现象 | 处理方式 |
|---|---|
| `ModuleNotFoundError: No module named 'UQPyL'` | 你运行脚本或 Notebook 的 Python 环境里还没有安装 UQPyL。用同一个环境执行 `python -m pip install UQPyL`。 |
| Notebook 里导入失败，但命令行可以导入 | Notebook kernel 和命令行不是同一个 Python。切换 kernel，或在 kernel 对应环境里安装。 |
| 源码安装时报 C/C++ 编译错误 | 优先安装发布包；若必须源码安装，先补齐编译器、`Cython`、`pybind11` 和构建依赖。 |

## 1. 定义一个 Problem

`Problem` 用来定义输入边界、目标函数和优化方向。其他模块都会复用这个对象。

这个 toy problem 有两个输入：

```text
y = x1^2 + 0.2*x2^2
```

```python
import numpy as np

from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.2 * X[:, 1] ** 2
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D", xLabels=["x1", "x2"])

res = problem.evaluate([[0.2, 0.3]])

print(res.objs)
print(res.cons)
```

Example output:

```text
[[0.058]]
None
```

`evaluate()` 返回的是 `Eval` 对象。目标值用 `res.objs` 读，约束值用 `res.cons` 读。这里没有约束，所以 `res.cons` 是 `None`。

## 2. 生成样本

DOE 采样器会直接从 `problem` 读取边界信息。

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.2 * X[:, 1] ** 2
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D", xLabels=["x1", "x2"])

X = LHS("classic").sample(problem, nSamples=10, seed=123)
Y = problem.evaluate(X).objs

print(X.shape)
print(Y.shape)
print(X[:3])
print(Y[:3])
```

Example output:

```text
(10, 2)
(10, 1)
[[ 0.9598  0.6464]
 [-0.2153 -0.9892]
 [ 0.3648  0.9036]]
[[1.0048]
 [0.2421]
 [0.2964]]
```

这里的行是严格对齐的：`Y[i]` 就是 `X[i, :]` 对应的输出。

## 3. 运行分析

分析模块用来解释输入如何影响输出。这个例子里，`x1` 的影响应该更大，因为它前面的系数更大。

```python
import numpy as np

from UQPyL.analysis import RBDFAST
from UQPyL.doe import LHS
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.2 * X[:, 1] ** 2
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D", xLabels=["x1", "x2"])

X = LHS("classic").sample(problem, nSamples=256, seed=123)
Y = problem.evaluate(X).objs
result = RBDFAST(verboseFlag=False).analyze(problem, X, Y=Y)

print(result.metricNames)
print(result.getMetric("S1").values)
print(result.getMetric("S1").colLabels)
```

Example output:

```text
['S1']
[[0.951  0.0475]]
['x1', 'x2']
```

第一列对应 `x1`，明显比第二列大很多。

## 4. 运行优化

优化模块用来搜索一个好的输入向量。这个问题在 `[0, 0]` 附近最小。

```python
import numpy as np

from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.2 * X[:, 1] ** 2
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D", xLabels=["x1", "x2"])

result = GA(nPop=8, maxFEs=40, maxIters=5, tolerate=None, verboseFlag=False, logFlag=False, saveFlag=False).run(problem, seed=123)

print(result.bestDecs)
print(result.bestObjs)
print(result.FEs, result.iters)
```

Example output:

```text
[[-0.0233 -0.0817]]
[[0.0019]]
40 5
```

`bestDecs` 是找到的最好输入行，`bestObjs` 是该行对应的目标值。

## 5. 运行推断

推断模块用于对标量目标或对数概率进行链式采样。

```python
import numpy as np

from UQPyL.inference import MH
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.2 * X[:, 1] ** 2
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D", xLabels=["x1", "x2"])

result = MH(nChains=3, warmUp=5, maxIters=30, verboseFlag=False, logFlag=False, saveFlag=False).run(problem, gamma=0.2, seed=123)

print(result.decs.shape)
print(result.acceptanceRate)
print(result.bestDecs)
print(result.bestObjs)
```

Example output:

```text
(3, 30, 2)
[0.8966 0.7586 0.9655]
[[-0.0296  0.1336]]
[[0.0044]]
```

`decs.shape` 的含义是 `(n_chains, draws, n_input)`，`acceptanceRate` 是每条链对应一个接受率。

## 6. 训练代理模型

代理模型会从已经评估好的 `X` 和 `Y` 里学习一个廉价预测器。

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.problem import Problem
from UQPyL.surrogate.rbf import RBF

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.2 * X[:, 1] ** 2
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D", xLabels=["x1", "x2"])

X = LHS("classic").sample(problem, nSamples=20, seed=123)
Y = problem.evaluate(X).objs

model = RBF()
model.fit(X, Y)

pred = model.predict([[0.0, 0.0], [0.5, 0.5]])

print(X.shape, Y.shape)
print(pred)
```

Example output:

```text
(20, 2) (20, 1)
[[0.0004]
 [0.3024]]
```

在 `[0.5, 0.5]` 处，预测值接近真实值 `0.5^2 + 0.2*0.5^2 = 0.3`。

## 7. 校准一个仿真模型

校准使用的是 `ModelProblem`，因为它需要把仿真结果和观测数据进行比较。

这个 toy model 有两个参数、两个观测时刻：

```text
obs = [[1.0], [2.0]]
sim(t1) = x1
sim(t2) = x2
```

```python
import numpy as np

from UQPyL.calibration import GLUE
from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)

obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, lb=0.0, ub=3.0, simFunc=simFunc, obs=obs, simLabels=["Q"], name="ToyModel")
X = np.array([[1.0, 2.0], [1.0, 2.4], [0.0, 0.0]])

result = GLUE(metric="rmse", verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X, threshold=0.3)

print(result.bestDecs)
print(result.bestSim)
print(result.behavioralDecs)
```

Example output:

```text
[[1. 2.]]
[[1. 2.]]
[[1.  2. ]
 [1.  2.4]]
```

`bestDecs` 是最优参数行，`behavioralDecs` 是被 GLUE 阈值接受的参数行。

## 运行控制项

大多数可运行方法都接受同一组运行控制参数：

| 选项 | 含义 |
|---|---|
| `verboseFlag` | 打印终端进度和最终摘要。 |
| `verboseFreq` | 进度输出间隔。 |
| `logFlag` | 在支持时写文本日志。 |
| `saveFlag` | 保存结构化运行结果，通常是 `Result/` 下的 sqlite 文件。 |
| `saveFreq` | 在支持快照的算法中，控制保存间隔。 |

学习和跑示例时，建议先把 `verboseFlag=False`、`logFlag=False`、`saveFlag=False`，这样终端更干净，也不会额外生成文件。

## 下一步

| 目标 | 阅读 |
|---|---|
| 理解统一建模协议 | [Problem](problem.md) |
| 生成输入样本 | [Design of Experiment](doe.md) |
| 分析输入影响 | [Analysis](analysis.md) |
| 优化参数 | [Optimization](optimization.md) |
| 推断参数分布 | [Inference](inference.md) |
| 校准仿真模型 | [Calibration](calibration.md) |
| 训练代理模型 | [Surrogate Modeling](surrogate.md) |
| 复制完整工作流 | [Examples](examples.md) |
