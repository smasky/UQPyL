# Calibration

`calibration` 模块通过比较仿真结果和观测数据来估计模型参数。

当你有观测数据、仿真模型、参数边界，并且希望找到能让仿真接近观测的参数时，使用校准模块。

校准方法使用 `ModelProblem`，不是普通 `Problem`。

在 UQPyL 里，校准问题的标准主链是：

```text
X -> simFunc(X) -> sim -> calibration metric / score -> parameter update or selection
```

对应到建模对象上就是：

```text
obs + simFunc + 参数边界 -> ModelProblem -> calibration.run(...) -> CalResult
```

也就是说，校准不是把普通 `Problem` 换个模块继续用，而是明确建立在 `ModelProblem` 这条仿真型问题主线上。

## 选择校准方法

| 方法 | 适合场景 | 主要输出 |
|---|---|---|
| `GLUE` | 已有候选参数，希望按阈值筛出 behavioral samples。 | `behavioralDecs`, `behavioralSims` |
| `SUFI2` | 需要 elite samples 和更新后的不确定性边界。 | `eliteDecs`, `updatedLb`, `updatedUb`, `pfactor`, `rfactor` |
| `ES` | 需要一次 ensemble smoother 更新。 | `posteriorDecs`, `posteriorSims` |
| `IES` | 需要多轮 ensemble smoother 更新。 | `posteriorDecs`, `posteriorSims`, 迭代历史 |

实践默认：已有候选样本时先用 `GLUE`；关心参数不确定性边界时用 `SUFI2`；做 ensemble smoothing 时用 `ES` 或 `IES`。

## 校准工作流

```text
obs + simFunc + 参数边界 -> ModelProblem -> calibration.run(...) -> CalResult
```

| 步骤 | 动作 |
|---|---|
| 准备观测 | `obs` 使用二维数组，shape 为 `(n_time, n_series)`。 |
| 定义仿真 | 写 `simFunc(X)`，支持批量参数行。 |
| 构建 `ModelProblem` | 提供 `nInput`、`lb`、`ub`、`simFunc`、`obs`，可选 `mask`。 |
| 选择方法 | 使用 `GLUE`、`SUFI2`、`ES` 或 `IES`。 |
| 读取结果 | 查看 `bestDecs`、`bestSim`、posterior、elite 或 behavioral samples。 |

## 构建 `ModelProblem`

推荐把 `ModelProblem` 理解成校准工作的标准建模容器：

- `simFunc(X)` 负责生成原始仿真输出
- `obs` 提供观测参照
- `mask` 控制哪些观测位置参与评分
- 校准方法再基于这些信息计算 metric、筛选样本或更新参数

对校准来说，`ModelProblem` 不要求必须定义 `objFunc`。只要有 `simFunc + obs`，一个 simulation-only `ModelProblem` 就已经是合法的校准容器。

这个 toy model 中，两个参数直接对应两个观测时刻：

```text
params [p1, p2] -> simulation [p1, p2]
obs = [1.0, 2.0]
```

```python
import numpy as np

from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, seriesLabels=["Q"], name="ToyModel")

sim = problem.simFunc([[1.0, 2.0]])

print(sim.shape)
print(problem.flattenObs())
print(problem.flattenMask())
```

| 对象 | 形状 | 含义 |
|---|---|---|
| `X` | `(n_samples, n_input)` | 候选参数行。 |
| `obs` | `(n_time, n_series)` | 观测数据。 |
| `simFunc(X)` | `(n_samples, n_time, n_series)` | 每个候选参数对应的仿真结果。 |
| flattened simulation | `(n_samples, n_time * n_series)` | 内部评分布局。 |

## 运行 GLUE

`GLUE` 会对每个候选参数评分，并保留通过阈值的 behavioral samples。

对于 `rmse` 这类越小越好的指标：

```text
score <= threshold
```

对于 `nse` 这类越大越好的指标：

```text
score >= threshold
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


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, seriesLabels=["Q"], name="ToyModel")
X = np.array([[1.0, 2.0], [1.0, 2.4], [0.0, 0.0]])

result = GLUE(metric="rmse", verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X, threshold=0.3)

print(result.bestDecs)
print(result.bestSim)
print(result.behavioralDecs)
print(result.diagnostics["scores"])
print(result.diagnostics["behavioralMask"])
```

| 输出 | 含义 |
|---|---|
| `bestDecs` | 最优参数行。 |
| `bestSim` | 最优参数对应的仿真输出。 |
| `behavioralDecs` | 通过阈值的候选参数行。 |
| `scores` | 每个候选参数的指标值。 |
| `behavioralMask` | 每个候选参数是否通过阈值。 |

## 指标方向

| 指标 | 更好方向 |
|---|---|
| `mse` | 越小越好 |
| `mae` | 越小越好 |
| `rmse` | 越小越好 |
| `nse` | 越大越好 |
| `r2` | 越大越好 |
| `pbias` | 越小越好 |
| `pearson_r` | 越大越好 |
| `kge` | 越大越好 |

阈值方向由指标决定。不要把 `rmse` 的阈值逻辑直接套到 `nse` 上。

## 使用 Mask

`mask` 用来在评分时忽略部分观测值，形状必须与 `obs` 一致。

```python
import numpy as np

from UQPyL.problem import ModelProblem


obs = np.array([[1.0, 10.0], [2.0, 20.0]])
mask = np.array([[False, True], [False, True]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 2))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    sim[:, :, 1] = 999.0
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, mask=mask, seriesLabels=["Q", "Ignored"], name="MaskedToyModel")

print(problem.obs.shape)
print(problem.mask.shape)
print(problem.flattenMask())
```

被 mask 的位置不会参与校准评分。

## 运行 SUFI2

`SUFI2` 会选择 elite samples，并根据 elite samples 更新参数不确定性边界。

```python
import numpy as np

from UQPyL.calibration import SUFI2
from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, seriesLabels=["Q"], name="ToyModel")
X = np.array([[1.0, 2.0], [1.0, 2.4], [0.0, 0.0]])

result = SUFI2(verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X, eliteSize=2)

print(result.bestDecs)
print(result.eliteDecs)
print(result.diagnostics["updatedLb"])
print(result.diagnostics["updatedUb"])
print(result.diagnostics["pfactor"], result.diagnostics["rfactor"])
```

| 输出 | 含义 |
|---|---|
| `eliteDecs` | 排名前 `eliteSize` 的参数行。 |
| `updatedLb`, `updatedUb` | 从 elite samples 得到的新边界。 |
| `pfactor` | 观测被不确定性带包住的比例。 |
| `rfactor` | 不确定性带宽度相对观测变化的大小。 |

## 运行 ES / IES

`ES` 执行一次 ensemble smoother 更新；`IES` 执行多轮更新。

```python
import numpy as np

from UQPyL.calibration import ES, IES
from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)


obs = np.array([[1.0], [2.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2, 1))
    sim[:, 0, 0] = X[:, 0]
    sim[:, 1, 0] = X[:, 1] ** 2
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, seriesLabels=["Q"], name="NonlinearToyModel")
X = np.array([[0.0, 0.5], [2.0, 1.0], [1.5, 2.0]])

esResult = ES(verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X)
iesResult = IES(maxIters=4, lam=1e-6, verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X)

print(iesResult.bestDecs)
print(iesResult.posteriorDecs.shape)
print(iesResult.posteriorSims.shape)
print(len(iesResult.history.metricsHistory))
print(np.mean(esResult.diagnostics["scores"]), np.mean(iesResult.diagnostics["scores"]))
```

用 `history.metricsHistory` 查看迭代级摘要。

## 读取 `CalResult`

| 字段 | 含义 |
|---|---|
| `bestDecs` | 当前指标下最优参数行。 |
| `bestSim` | `bestDecs` 对应仿真，按观测向量布局展平。 |
| `behavioralDecs`, `behavioralSims` | GLUE 通过阈值的样本。 |
| `eliteDecs`, `eliteSims` | SUFI2 elite samples。 |
| `posteriorDecs`, `posteriorSims` | ES 或 IES 的 posterior ensemble。 |
| `diagnostics` | 方法相关的分数、mask、边界和摘要。 |
| `history.metricsHistory` | 迭代方法的每轮摘要。 |
| `summary()` | 适合报告的紧凑字典。 |

## 常见错误

| 错误 | 修正 |
|---|---|
| 用普通 `Problem` 做校准 | 使用带 `simFunc` 和 `obs` 的 `ModelProblem`。 |
| `simFunc` 返回形状错误 | 返回 `(n_samples, n_time, n_series)`。 |
| `obs` 是一维数组 | 使用二维数组，例如 `obs.reshape(-1, 1)`。 |
| 忘记指标方向 | 对越小越好的指标用 `<= threshold`，对越大越好的指标用 `>= threshold`。 |
| GLUE 阈值过严 | 查看 `diagnostics["scores"]` 后调整阈值。 |
| `mask` 形状不对 | 保证 `mask.shape == obs.shape`。 |
| 期待 `bestSim` 保持三维 | 结果中仿真通常是评分布局；需要时间序列时按 `obs.shape` reshape。 |

## 下一步

| 目标 | 阅读 |
|---|---|
| 构建仿真问题 | [Problem](problem.md) |
| 生成候选参数集 | [Design of Experiment](doe.md) |
| 查构造参数和结果字段 | [Calibration API](api/calibration.md) |
| 对比推断工作流 | [Inference](inference.md) |
| 查看完整工作流 | [Examples](examples.md) |

### ES / IES 最优样本的指标方向

最终后验集合按指标本身的方向选出 best：RMSE/MSE/MAE 越小越好，NSE/KGE/R² 等越大越好。内部 `normalizedScore()` 对越大越好的指标取负号、对 `pbias` 取绝对值，统一供最小化比较；它不把分数缩放到 0–1。诊断、打印和保存仍使用原始指标值，指标选择不改变 ES/IES 的集合更新公式。


### PBIAS 的距零判据

使用 `metric="pbias"` 标签时，选优比较 `abs(PBIAS)`；GLUE 的阈值是非负、有限的百分数容差，例如 5 表示 `-5% <= PBIAS <= 5%`，包含边界。ES、IES、SUFI2 同样沿用距零选优。

原始公式保持 `100 * sum(sim - obs) / sum(obs)`，打印、保存及 diagnostics 保留正负号。绝对值只取在最终指标外，不对逐时误差取绝对值后求和；正负误差相抵时 PBIAS 可以为零，这不表示每个时刻都准确。上述特殊规则由字符串标签启用，直接传入自定义指标函数仍沿用默认最小化约定。


### ES / IES 的协方差退化处理

观测维数超过集合规模减一、重复观测或集合没有差异时，样本协方差可能秩不足。ES/IES 共用以下求解规则：满数值秩时使用线性求解；否则使用对称特征分解构成截断伪逆。特征值不大于 `n_valid_obs * eps * max(abs(eigenvalues))` 的方向被舍弃；零秩时增益为零，集合保持不变。伪逆不能补充集合没有表达的信息，也不保证所有观测都能拟合。

默认 R 仍为零，IES 的 lam 仍默认 0；不自动添加观测噪声或岭项。R 必须形状正确、有限、对称且半正定；浮点舍入范围内的不对称被对称化，极小负特征值截为零。IES lam 必须是有限非负标量。

ES 的 `diagnostics['covarianceSolve']` 和 IES 的逐轮 `diagnostics['covarianceSolves']` 记录 solver、rank、dimension、cutoff。检测使用观测空间特征分解，会增加计算开销；本轮不包含针对超长观测序列的集合空间加速。
