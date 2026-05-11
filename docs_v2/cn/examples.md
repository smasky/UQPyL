# Examples

这一页收集可以直接运行的完整工作流。每个例子都把 `Problem` 或 `ModelProblem` 和 UQPyL 的功能模块串起来。

示例预算都很小，目的是快速跑通。真实任务中，通常需要增加样本数、优化预算、链长度或校准候选规模。

## 工作流地图

| 工作流 | 示例问题 | 实际使用时替换 |
|---|---|---|
| 采样和敏感性分析 | 加权二维 sphere | `objFunc` |
| 单目标优化 | 最小化到原点的距离 | `objFunc`、边界、优化器设置 |
| 多目标优化 | ZDT1 benchmark Pareto 搜索 | benchmark 或自定义多目标 `Problem` |
| MCMC 推断 | 围绕低目标值区域采样 | `objFunc` 或 `logProbFunc` |
| 模型校准 | 两个仿真时刻匹配观测 | `simFunc`、`obs`、候选参数矩阵 |
| 代理模型训练和验证 | 学习非线性响应面 | `objFunc`、采样规模、代理模型 |
| 代理辅助优化 | 优化代理模型，再用原模型复核 | 昂贵目标函数、infill/update 策略 |

## 采样和敏感性分析

这个例子判断两个输入中谁对输出更重要：

```text
y = x1^2 + 0.2 * x2^2
```

```python
import numpy as np

from UQPyL.analysis import RBDFAST
from UQPyL.doe import LHS
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return (X[:, 0] ** 2 + 0.2 * X[:, 1] ** 2).reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, ub=1.0, lb=-1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D")

X = LHS("classic").sample(problem, nSamples=256, seed=123)
Y = problem.evaluate(X).objs
result = RBDFAST(verboseFlag=False).analyze(problem, X, Y=Y)

print(result.metricNames)
print(result.getMetric("S1").values)
```

替换点：

| 示例部分 | 实际替换 |
|---|---|
| `objFunc` | 你的模型、仿真包装器或目标计算。 |
| `nSamples=256` | 适合模型成本和输入维度的样本数。 |
| `RBDFAST` | 如果问题需要，换成其他分析方法。 |

## 单目标优化

这个例子最小化二维 sphere：

```python
import numpy as np

from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, ub=1.0, lb=-1.0, objFunc=objFunc, optType="min", name="Sphere2D")

algorithm = GA(nPop=8, maxFEs=40, maxIters=5, verboseFlag=False, logFlag=False, saveFlag=False)
result = algorithm.run(problem, seed=123)

print(result.bestDecs)
print(result.bestObjs)
```

`bestDecs` 是找到的最好决策行，`bestObjs` 是对应目标值。

## 多目标优化

这个例子使用内置 `ZDT1` benchmark，先确认多目标工作流。

```python
from UQPyL.optimization.moea import NSGAII
from UQPyL.problem import ZDT1


problem = ZDT1(nInput=5)
algorithm = NSGAII(nPop=12, maxFEs=48, maxIters=3, verboseFlag=False, logFlag=False, saveFlag=False)
result = algorithm.run(problem, seed=123)

print(result.bestDecs.shape)
print(result.bestObjs.shape)
print(result.bestMetric)
```

实际使用时，将 `ZDT1(nInput=5)` 替换为你的多目标 `Problem`，并根据成本调整 `NSGAII` 预算。

## MCMC 推断

这个例子把标量目标当作未归一化评分。sphere 值越低，默认 log probability 越高。

```python
import numpy as np

from UQPyL.inference import MH
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(nInput=2, nObj=1, ub=2.0, lb=-2.0, objFunc=objFunc, optType="min", name="SpherePosterior")
method = MH(nChains=3, warmUp=5, maxIters=30, verboseFlag=False, logFlag=False, saveFlag=False)
result = method.run(problem, gamma=0.2, seed=123)

print(result.decs.shape)
print(result.logProb.shape)
print(result.acceptanceRate)
```

`decs.shape` 是 `(n_chains, draws, n_input)`。

## 模型校准

这个例子用两个参数匹配两个观测时刻：

```text
obs = [[1.0], [2.0]]
sim(t1) = x1
sim(t2) = x2
```

```python
import numpy as np

from UQPyL.calibration import GLUE
from UQPyL.problem import ModelProblem


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, simLabels=["Q"], name="ToyModel")
X = np.array([[1.0, 2.0], [1.0, 2.4], [0.0, 0.0]])

result = GLUE(metric="rmse", verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X, threshold=0.3)

print(result.bestDecs)
print(result.bestSim)
print(result.behavioralDecs)
```

实际使用时替换 `simFunc`、`obs` 和候选参数矩阵 `X`。

## 代理模型训练和验证

这个例子采样一个非线性响应面，训练 `RBF`，并在测试集上验证。

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.problem import Problem
from UQPyL.surrogate import RandSelect, mse, r_square
from UQPyL.surrogate.rbf import RBF

np.random.seed(123)


def objFunc(X):
    X = np.atleast_2d(X)
    return (np.sin(np.pi * X[:, 0]) + X[:, 1] ** 2).reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, ub=1.0, lb=0.0, objFunc=objFunc, optType="min", name="SurrogateToy")

X = LHS("classic").sample(problem, nSamples=30, seed=123)
Y = problem.evaluate(X).objs
trainIdx, testIdx = RandSelect(pTest=25).split(X)

model = RBF()
model.fit(X[trainIdx], Y[trainIdx])
pred = model.predict(X[testIdx])

print(pred.shape)
print(r_square(Y[testIdx], pred))
print(mse(Y[testIdx], pred))
```

## 代理辅助优化模式

基本手动模式：

1. 在 DOE 点上评估昂贵模型。
2. 用这些评估训练代理模型。
3. 低成本优化代理模型。
4. 用原始昂贵模型复核代理最优点。

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem
from UQPyL.surrogate.rbf import RBF


def expensiveObjFunc(X):
    X = np.atleast_2d(X)
    return (np.sin(np.pi * X[:, 0]) + X[:, 1] ** 2).reshape(-1, 1)


expensiveProblem = Problem(nInput=2, nObj=1, ub=1.0, lb=0.0, objFunc=expensiveObjFunc, optType="min", name="ExpensiveToy")

X = LHS("classic").sample(expensiveProblem, nSamples=30, seed=123)
Y = expensiveProblem.evaluate(X).objs
surrogate = RBF()
surrogate.fit(X, Y)


def surrogateObjFunc(Xnew):
    return surrogate.predict(Xnew)


surrogateProblem = Problem(nInput=2, nObj=1, ub=1.0, lb=0.0, objFunc=surrogateObjFunc, optType="min", name="SurrogateToy")
result = GA(nPop=8, maxFEs=40, maxIters=5, verboseFlag=False, logFlag=False, saveFlag=False).run(surrogateProblem, seed=123)

checkedObj = expensiveProblem.evaluate(result.bestDecs).objs

print(result.bestDecs)
print(result.bestObjs)
print(checkedObj)
```

`result.bestObjs` 是代理预测值，`checkedObj` 是真实模型复核值。

## 下一步

| 目标 | 阅读 |
|---|---|
| 理解统一建模协议 | [Problem](problem.md) |
| 生成样本 | [Design of Experiment](doe.md) |
| 做敏感性分析 | [Analysis](analysis.md) |
| 优化目标 | [Optimization](optimization.md) |
| 运行推断 | [Inference](inference.md) |
| 校准仿真模型 | [Calibration](calibration.md) |
| 训练代理模型 | [Surrogate Modeling](surrogate.md) |
