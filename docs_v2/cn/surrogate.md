# Surrogate Modeling

`surrogate` 模块用已经评估过的输入输出数据训练廉价预测模型。

当原始模型很贵，而你已经有一张输入 `X` 和输出 `Y` 的数据表时，可以用代理模型做快速预测、验证、响应面绘图、筛选或代理辅助优化。

核心流程：

```text
采样或收集 X -> 评估 Y -> model.fit(X, Y) -> model.predict(Xnew)
```

## 代理模型需要什么

代理模型不会自己调用你的仿真。它只从已有数据里学习。

| 对象 | 形状 | 含义 |
|---|---|---|
| `X` | `(n_samples, n_input)` | 输入表。每行是一个已评估参数向量。 |
| `Y` | `(n_samples, n_output)` | 输出表。每行必须与 `X` 同一行对应。 |
| `Xnew` | `(n_new, n_input)` | 需要预测的新输入行。 |
| `pred` | `(n_new, n_output)` | 预测输出。 |

单输出时，推荐把 `Y` 写成 `(n_samples, 1)` 的列。

## 基本 Fit 和 Predict

```python
import numpy as np

from UQPyL.surrogate.rbf import RBF

np.set_printoptions(precision=4, suppress=True)

X = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
Y = np.sin(2 * np.pi * X)

model = RBF()
model.fit(X, Y)

pred = model.predict([[0.25], [0.75]])

print(X.shape, Y.shape)
print(pred.shape)
print(pred)
```

`[[0.25], [0.75]]` 表示两行、一个变量。对于二维输入，一行应该写成 `[[0.2, 0.8]]`。

## 保持行对齐

`X` 和 `Y` 必须按同一顺序描述同一批模型运行。

| 行 | 输入 | 输出 |
|---:|---|---|
| `0` | `X[0, :]` | `Y[0, :]` |
| `1` | `X[1, :]` | `Y[1, :]` |
| `2` | `X[2, :]` | `Y[2, :]` |

不要分别打乱 `X` 和 `Y`。划分训练集和测试集时，对两个数组使用同一组索引。

## 选择模型

| 模型 | 导入路径 | 首选用途 |
|---|---|---|
| `RBF` | `UQPyL.surrogate.rbf` | 平滑数据的默认响应面模型。 |
| `GPR` | `UQPyL.surrogate.gp` | 需要预测不确定性的 Gaussian process。 |
| `KRG` | `UQPyL.surrogate.kriging` | Kriging 风格模型，支持趋势项和不确定性。 |
| `LinearRegression` | `UQPyL.surrogate.regression` | 线性基线模型。 |
| `PolynomialRegression` | `UQPyL.surrogate.regression` | 低阶平滑多项式关系。 |
| `MARS` | `UQPyL.surrogate.mars` | 可选依赖可用时的分段回归样条。 |
| `SVR` | `UQPyL.surrogate.svr` | 可选依赖可用时的支持向量回归。 |

实践默认：先用 `RBF`；需要 uncertainty 时用 `GPR` 或 `KRG`；用回归模型做 baseline。

## 验证预测质量

训练后至少用 held-out data 检查一次。

```python
import numpy as np

from UQPyL.surrogate import RandSelect, mse, r_square
from UQPyL.surrogate.rbf import RBF

np.set_printoptions(precision=4, suppress=True)
np.random.seed(123)

X = np.linspace(0.0, 1.0, 24).reshape(-1, 1)
Y = np.sin(2 * np.pi * X) + 0.2 * X

trainIdx, testIdx = RandSelect(pTest=25).split(X)

model = RBF()
model.fit(X[trainIdx], Y[trainIdx])

pred = model.predict(X[testIdx])

print(trainIdx.shape, testIdx.shape)
print(np.round(r_square(Y[testIdx], pred), 4))
print(np.round(mse(Y[testIdx], pred), 6))
print(pred[:3])
```

| 指标 | 方向 | 含义 |
|---|---|---|
| `r_square` | 越大越好，`1.0` 表示完美预测。 | 预测解释输出变化的比例。 |
| `nse` | 越大越好。 | Nash-Sutcliffe efficiency。 |
| `mse` | 越小越好。 | 均方误差。 |
| `rank_score` | 越大越好。 | 预测是否保留样本排序。 |
| `sort_score` | 越小越好。 | 真实排序和预测排序的距离。 |

## 缩放数据

变量单位或量级差异很大时，建议使用 scaler。

```python
import numpy as np

from UQPyL.surrogate import StandardScaler
from UQPyL.surrogate.rbf import RBF

np.set_printoptions(precision=4, suppress=True)

X = np.linspace(0.0, 100.0, 12).reshape(-1, 1)
Y = np.cos(X / 20.0)

model = RBF(scalers=(StandardScaler(0, 1), StandardScaler(0, 1)))
model.fit(X, Y)

pred = model.predict([[50.0], [75.0]])

print(pred.shape)
print(pred)
print(model.xTrain.mean(axis=0), model.xTrain.std(axis=0, ddof=1))
```

`scalers=(xScaler, yScaler)` 中，第一个 scaler 用于输入，第二个用于输出。预测结果会自动回到原始输出尺度。

## 带不确定性的预测

只有支持 uncertainty 的模型才能返回标准差或方差。当前公共模型中，通常使用 `GPR` 或 `KRG`。

```python
import numpy as np

from UQPyL.surrogate import StandardScaler
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF as RBFKernel

np.set_printoptions(precision=4, suppress=True)

X = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
Y = X**2

model = GPR(scalers=(StandardScaler(0, 1), StandardScaler(0, 1)), kernel=RBFKernel(), nRestartTimes=1)
model.rng = np.random.default_rng(123)
model.fit(X, Y)

mean, std = model.predict([[0.3], [0.7]], returnStd=True)

print(mean.shape, std.shape)
print(mean)
print(std)
```

调用形式：

```text
model.predict(Xnew)                 -> mean only
model.predict(Xnew, returnStd=True) -> mean, standard deviation
model.predict(Xnew, returnVar=True) -> mean, variance
```

`RBF` 不提供 uncertainty 输出。

## 多输出代理模型

多输出时，可以用 `MultiSurrogate` 为每个输出列训练一个模型。

```python
import numpy as np

from UQPyL.surrogate import MultiSurrogate
from UQPyL.surrogate.rbf import RBF

np.set_printoptions(precision=4, suppress=True)

X = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
Y = np.hstack([np.sin(2 * np.pi * X), X**2])

model = MultiSurrogate(2, models_list=[RBF(), RBF()])
model.fit(X, Y)

pred = model.predict([[0.25], [0.75]])

print(Y.shape)
print(pred.shape)
print(pred)
```

`n_surrogates` 必须与 `Y.shape[1]` 一致，`models_list` 的长度也必须一致。

## 调参

`AutoTuner` 用验证集搜索参数。`gridTune()` 显式、容易检查，适合先用。

```python
import numpy as np

from UQPyL.surrogate import AutoTuner, StandardScaler
from UQPyL.surrogate.rbf import RBF

np.set_printoptions(precision=4, suppress=True)
np.random.seed(123)

X = np.linspace(0.0, 1.0, 16).reshape(-1, 1)
Y = X**2 + 0.1

model = RBF(
    scalers=(StandardScaler(0, 1), StandardScaler(0, 1)),
    C_smooth_attr={"ub": 0.1, "lb": 0.0, "type": "float", "log": False},
)
tuner = AutoTuner(model=model)

bestParams, bestScore = tuner.gridTune(X, Y, paraGrid={"C_smooth": [0.0, 1e-6, 1e-4]}, ratio=25)

print(bestParams)
print(round(float(bestScore), 4))
print(model.predict([[0.5]]))
print(model.getParameterValues("C_smooth"))
```

调参后，`AutoTuner` 会把最优参数应用到模型，并用全量数据重新拟合。

## 与 `Problem` 结合

代理模型的训练数据通常来自 DOE 样本和 `Problem.evaluate()`。

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.problem import Problem
from UQPyL.surrogate.rbf import RBF

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.5 * X[:, 1] ** 2
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D")

X = LHS("classic").sample(problem, nSamples=20, seed=123)
Y = problem.evaluate(X).objs

model = RBF()
model.fit(X, Y)

pred = model.predict([[0.0, 0.0], [0.5, 0.5]])

print(X.shape, Y.shape)
print(pred)
```

## 在优化中使用代理模型

拟合完成后，可以把 `model.predict()` 包装成新的 `Problem`，让优化器搜索代理模型。

```python
import numpy as np

from UQPyL.doe import LHS
from UQPyL.optimization.soea import GA
from UQPyL.problem import Problem
from UQPyL.surrogate.rbf import RBF

np.set_printoptions(precision=4, suppress=True)


def realObjFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.5 * X[:, 1] ** 2
    return y.reshape(-1, 1)


realProblem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=realObjFunc, optType="min", name="RealModel")
X = LHS("classic").sample(realProblem, nSamples=20, seed=123)
Y = realProblem.evaluate(X).objs

model = RBF()
model.fit(X, Y)


def surrogateObjFunc(X):
    return model.predict(X)


surrogateProblem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=surrogateObjFunc, optType="min", name="SurrogateModel")
result = GA(nPop=8, maxFEs=32, maxIters=4, verboseFlag=False, logFlag=False, saveFlag=False).run(surrogateProblem, seed=123)

print(result.bestDecs)
print(result.bestObjs)
print(realProblem.evaluate(result.bestDecs).objs)
```

最后一行很重要：代理模型是近似，关键候选点必须回到真实模型验证。

## 常见错误

| 错误 | 修正 |
|---|---|
| `X` 和 `Y` 行不匹配 | 从采样、评估到切分都保持同一行顺序。 |
| 一维输入预测写错形状 | 一个变量两行写 `[[0.2], [0.8]]`。 |
| 只看训练误差 | 使用 held-out validation 或交叉验证。 |
| 期待 `RBF` 返回 uncertainty | 使用 `GPR` 或 `KRG`。 |
| 忘记代理模型只是近似 | 重要候选点必须用原始模型复核。 |
| 手动缩放了 `X` 但忘记缩放 `Xnew` | 优先使用模型自带 `scalers`。 |

## 下一步

| 目标 | 阅读 |
|---|---|
| 查模型构造参数、核函数和 scaler | [Surrogate API](api/surrogate.md) |
| 定义被评估系统 | [Problem](problem.md) |
| 生成训练样本 | [Design of Experiment](doe.md) |
| 优化拟合后的代理模型 | [Optimization](optimization.md) |
| 查看端到端示例 | [Examples](examples.md) |


## 核对象作为构建模板

GPR、KRG 和 RBF 代理模型会复制传入的核对象，构造函数和 `setKernel()` 使用相同规则。外部核用于描述配置，训练只修改模型内部的独立核实例。

```python
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.gp.kernel import RBF as RBFKernel

kernel = RBFKernel(length_scale=2.0)
modelA = GPR(kernel=kernel)
modelB = GPR(kernel=kernel)

assert modelA.kernel is not kernel
assert modelB.kernel is not modelA.kernel
```

复制保留核的当前参数值、范围、类型和其他核配置，不携带模型专属参数。修改外部模板只影响后续创建或显式重新设置的模型；训练后的实际核参数通过 `model.kernel` 查看。

`kernel.clone()` 可显式获得独立配置副本。`setKernelChoices()` 在登记时复制候选模板，每次切换再创建独立运行核，因此训练不会改写候选配置。候选选择字段记录模板选择，运行核仍以 `model.kernel` 为准。

自定义核如果增加训练缓存或外部资源，应覆写 `clone()`，返回配置独立且不携带训练状态的新实例。

### Matern 的 nu 参数

`Matern(nu=0.7)` 可固定任意正的 nu；`nu=np.inf` 对应 Gaussian 极限。设置 `optimize_nu=True` 时，nu 只在 `0.5、1.5、2.5、np.inf` 中选择，构造时也应指定其中一个候选。

常用候选使用简化公式。其他 nu 使用通用 Bessel 公式；大 nu 或中间计算溢出时使用等价的稳定积分计算，因此可能比常用候选慢。零距离的相关值为 1。

### 核参数校验

内置 GP、Kriging 和 RBF 核在构造时检查参数，在初始化时检查参数和调优边界；每次整批核计算前重新检查当前参数，因此直接修改 Setting 或调优器更新出的非法值会在使用前报 `ValueError`。异常不会自动回滚 Setting 的修改。

`length_scale`、`alpha`、`epsilon` 必须为有限正数；`theta`、常数核幅度和 `sigma` 必须为有限非负数；`nu` 为正数或 `np.inf`。`length_scale`、`theta` 可为标量或与输入特征数一致的一维向量，其余为标量。连续调优边界必须合法、有序、有限，对数调优的边界还必须严格为正；初值无需落在调优区间内。GP 交叉核的两个输入必须具有相同特征数。

参数检查通常为 O(d)，其中 d 是特征数；形状检查不扫描整个距离矩阵，不在每个样本对上重复执行。

### Scaler 与预测不确定性的量纲

`scalers=(xScaler, yScaler)` 仍为可选配置，不强制改变各模型默认值。优化内部已经提供 0–1 输入时，通常不需要再次启用输入 Scaler；独立代理模型仍可对真实输入和输出作预处理。

内置 `StandardScaler` 和 `MinMaxScaler` 提供 `inverse_transform_std()` 与 `inverse_transform_var()`。对于逆变换 `y = b + a*z`，均值按该式还原，标准差乘 `abs(a)`，方差乘 `a**2`，不加偏移。GPR/KRG 在模型内部保留训练尺度的方差，预测输出时统一还原一次；多输出均值、标准差、方差形状均为 `(n_predictions, n_outputs)`。

`StandardScaler(muX, sitaX)` 允许自定义目标均值和正标准差；常规多样本数据保留 `ddof=1`。单样本及常数列用源尺度 1 回退；`MinMaxScaler(min_, max_)` 的常数列也用源跨度 1 回退。这样训练值映射到目标中心/下界，变换仍可逆，不会把常数观测误当成预测方差必须为零。

Scaler 接收二维 `(n_samples, n_features)`；一维输入沿用单行语义，单特征多样本请用 `reshape(-1, 1)`。拒绝空训练集、非有限值、维度不匹配、非法目标尺度和未拟合调用。自定义 yScaler 若用于不确定性预测，需要实现 `inverse_transform_var()`；仅均值预测仍只需普通逆变换。非线性变换不能套用上述仿射公式。


### AutoTuner 的预处理时机

`optTune()` 和 `gridTune()` 先划分原始训练/验证数据，仅在训练子集上拟合 Scaler 和准备训练特征。验证集通过模型的 `predict()` 复用这些变换，评分使用原始输出尺度。选定参数后，在全量原始数据上重新拟合 Scaler 和最终模型；`joint`、`separate` 两种模式均遵循此规则。返回的验证分数仍来自划分阶段，不是全量重训后的训练分数。


### 可复现的数据划分与调参

`RandSelect(...).split(X, seed=42)`、`KFold(...).split(X, seed=42)` 以及 AutoTuner 的 `gridTune(..., seed=42)` / `optTune(..., seed=42)` 支持关键字 `seed`；也可传 `rng=np.random.default_rng(42)`，两者不能同时指定。拆分器不再使用全局 NumPy 随机状态。未提供 seed/rng 时，AutoTuner 消费自身 `rng`，多次调用会推进该随机流。

AutoTuner 为拆分、模型拟合和优化器派生独立子种子；候选拟合及最终拟合使用相同模型子种子的初始随机状态。固定数据、配置和实现时可复现；用户自定义组件需使用模型提供的 rng 才能受这一控制。`tuner.lastSplit` 保存最近调用的 `train_indices/test_indices` 以及 `seed/split_seed/model_seed/optimizer_seed`，返回的参数和分数二元组不变。该信息保留在内存，按需由调用者导出。

KFold 将余数均匀分配到各折，每个样本在 full 模式中恰好参与一次验证，折大小最多差 1；single 模式在相同 seed 下返回 full 的第一折。折数须为 2 到样本数之间的整数。RandSelect 的 pTest 表示验证集百分比，须在 0 和 100 之间；普通数据取 floor(n*pTest/100)，并至少各保留一个训练/验证样本。单样本仍返回空验证集，它不能产生有效的验证评分。
