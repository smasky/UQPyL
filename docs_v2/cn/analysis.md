# 分析

`analysis` 模块用于估计输入变量如何影响目标输出或约束输出。

当你想回答下面这些问题时，就会用到分析模块：

| 问题 | 适合的方法 |
|---|---|
| 哪些输入变量最重要？ | `RBDFAST`, `RSA`, `DeltaTest`, `Morris` |
| 每个变量解释了多少输出方差？ | `Sobol`, `FAST` |
| 变量交互是否重要？ | 使用 `secondOrder=True` 的 `Sobol`，或比较 `S1` 和 `ST` |
| 哪些变量可以早期筛掉？ | `Morris`, `DeltaTest` |
| 我已经有普通样本和输出了，还能做什么？ | `RBDFAST`, `RSA`, `DeltaTest` |

分析模块不负责优化模型。它解释的是“已经采样的输入”和“模型输出”之间的关系。

## 分析前需要准备什么

大多数分析流程需要四个对象：

| 对象 | 含义 |
|---|---|
| `problem` | 一个 `Problem` 或 benchmark problem，定义输入、边界、标签和输出。 |
| `X` | 输入样本矩阵，形状为 `(n_samples, n_input)`。 |
| `Y` | 输出矩阵，形状为 `(n_samples, n_outputs)`。每一行必须对应 `X` 的同一行。 |
| `meta` | 来自 `sampleWithMeta()` 的采样元信息。`Sobol`、`FAST` 和 `Morris` 必须使用。 |

常见流程是：

```text
define problem -> generate X -> evaluate Y -> run analysis -> read AnaResult
```

如果模型很贵，建议先计算并保存 `Y`，再把 `Y` 传给 `analyze()`。如果省略 `Y`，UQPyL 会在分析过程中内部评估 `problem`。

## 基本流程

这个例子回答一个问题：在 `f(x1, x2) = x1^2 + 0.2*x2^2` 中，哪个输入变量影响更大？

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

print(X.shape, Y.shape)
print(result.metricNames)
print(result.getMetric("S1").values)
print(result.getMetric("S1").rowLabels)
print(result.getMetric("S1").colLabels)
```

Example output:

```text
(256, 2) (256, 1)
['S1']
[[0.951  0.0475]]
['obj1']
['x1', 'x2']
```

解释：`x1` 明显比 `x2` 更重要。指标矩阵的列顺序跟 `problem.xLabels` 一致，所以第一个值对应 `x1`，第二个值对应 `x2`。

## 理解 `X`、`Y` 和 `meta`

`X` 是输入表。

```text
X[row, column] = one sampled value
row = one model run
column = one input variable
```

`Y` 是输出表。

```text
Y[row, column] = one output value
row = the same model run as X[row]
column = one objective or constraint output
```

行顺序必须保持一致：

| 行 | 输入行 | 输出行 |
|---:|---|---|
| `0` | `X[0, :]` | `Y[0, :]` |
| `1` | `X[1, :]` | `Y[1, :]` |
| `2` | `X[2, :]` | `Y[2, :]` |

模型评估之后，不要单独打乱 `X` 或 `Y`。只要行对应关系错了，分析结果就没有意义。

`meta` 是描述 `X` 如何生成的字典，不是模型输出。有些分析方法需要它，是因为这些方法要求 `X` 的行排列具有特定结构。

## 指定设计和自由设计

分析方法可以分成两组。

| 分组 | 方法 | 含义 |
|---|---|---|
| 指定设计方法 | `Sobol`, `FAST`, `Morris` | `X` 的行必须来自匹配的 DOE 专用设计。需要使用 `sampleWithMeta()`，并传入 `meta`。 |
| 自由设计方法 | `RBDFAST`, `RSA`, `DeltaTest`, `MARS` | 可以分析普通样本矩阵。可以使用 `LHS`、`Random`、低差异 `Sobol`，也可以使用已有仿真样本。 |

指定设计方法是严格的：

| 分析方法 | 必须使用的 DOE 设计 | 为什么必须这样 |
|---|---|---|
| `Sobol` | `SaltelliDesign(secondOrder=...).sampleWithMeta(...)` | `Sobol` 需要基础样本和混合样本按 Saltelli 行结构排列。 |
| `FAST` | `FASTDesign(M=...).sampleWithMeta(...)` | `FAST` 需要按 Fourier 频率构造的样本。 |
| `Morris` | `MorrisDesign(numLevels=...).sampleWithMeta(...)` | `Morris` 需要轨迹样本，每一步只改变一个变量。 |

自由设计方法更灵活：

| 分析方法 | 常见样本来源 | 说明 |
|---|---|---|
| `RBDFAST` | `LHS("classic")`、`Random`、普通 DOE `Sobol`、已有 `X` | 已有普通样本时的好起点。 |
| `RSA` | `LHS("classic")`、`Random`、普通 DOE `Sobol`、已有 `X` | 输出区域内需要足够样本。 |
| `DeltaTest` | `LHS("classic")`、`Random`、普通 DOE `Sobol`、已有 `X` | 使用近邻估计，样本密度会影响结果。 |
| `MARS` | 普通训练样本 | 只有安装了可选依赖时可用。 |

这里的“自由”只表示方法不绑定某个专用 DOE 设计，并不表示样本质量无所谓。要解释结果，仍然建议使用空间填充较好的样本，并保证样本量足够覆盖输入范围。

特别注意 `Sobol` 这个名字：`UQPyL.doe.Sobol()` 是低差异采样器，而 `UQPyL.analysis.Sobol` 是敏感性分析方法。分析方法需要 `SaltelliDesign`，不是普通 `Sobol()` 样本。

## 选择分析方法

从你想回答的问题开始选。

| 场景 | 建议先用 | 说明 |
|---|---|---|
| 你有普通样本，想快速得到变量排序。 | `RBDFAST` | 给出一阶敏感性 `S1`。 |
| 你需要标准的基于方差的敏感性指标。 | `Sobol` | 需要 `SaltelliDesign`。 |
| 你需要 Fourier-based 方差方法。 | `FAST` | 需要 `FASTDesign`。 |
| 输入变量很多，需要筛选。 | `Morris` | 需要 `MorrisDesign`。 |
| 你已经有普通 DOE 的仿真存档。 | `RBDFAST`, `RSA`, `DeltaTest` | 不需要方法专用的 `meta`。 |
| 你想做近邻贡献估计。 | `DeltaTest` | 对样本密度和近邻结构敏感。 |

实用默认选择：如果你已经有 `LHS` 或 `Random` 样本，先用 `RBDFAST`。如果可以重新设计样本，并且需要正式的方差分解指标，用 `Sobol`。

## Sobol 分析

`Sobol` 估计一阶和总阶方差贡献。

当你可以在运行模型之前生成 `SaltelliDesign` 时，使用它。

```python
import numpy as np

from UQPyL.analysis import Sobol
from UQPyL.doe import SaltelliDesign
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] + 0.1 * X[:, 1]
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc, optType="min", name="Linear2D", xLabels=["flow", "roughness"])

X, meta = SaltelliDesign(secondOrder=False).sampleWithMeta(problem, N=64, seed=123)
Y = problem.evaluate(X).objs

result = Sobol(verboseFlag=False).analyze(problem, X, Y=Y, meta=meta)

print(X.shape)
print(meta)
print(result.metricNames)
print(result.getMetric("S1").values)
print(result.getMetric("ST").values)
print(result.getMetric("S1_norm").values)
```

Example output:

```text
(256, 2)
{'designType': 'saltelli', 'N': 64, 'secondOrder': False, 'skipValue': 0, 'scramble': True, 'blockSize': 4, 'seed': 123}
['S1', 'S1_norm', 'ST', 'ST_norm']
[[0.9878 0.0104]]
[[0.9908 0.0098]]
[[0.9896 0.0104]]
```

解释：`flow` 主导输出方差。`roughness` 的贡献较小，因为它在模型中的系数是 `0.1`。

当 `nInput=2` 且 `secondOrder=False` 时，总模型运行次数是：

```text
N * (nInput + 2) = 64 * 4 = 256
```

如果 `secondOrder=True`，`Sobol` 还会返回 `S2`，样本数量变为：

```text
N * (2*nInput + 2)
```

## FAST 分析

`FAST` 也估计一阶和总阶敏感性，但它使用 Fourier 设计。

使用 `FASTDesign.sampleWithMeta()`。基础样本量 `N` 必须满足 `N > 4*M^2`。

```python
import numpy as np

from UQPyL.analysis import FAST
from UQPyL.doe import FASTDesign
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] + 0.1 * X[:, 1]
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc, optType="min", name="Linear2D", xLabels=["flow", "roughness"])

X, meta = FASTDesign(M=4).sampleWithMeta(problem, N=256, seed=123)
Y = problem.evaluate(X).objs

result = FAST(verboseFlag=False).analyze(problem, X, Y=Y, meta=meta)

print(X.shape)
print(meta)
print(result.getMetric("S1").values)
print(result.getMetric("ST").values)
```

Example output:

```text
(512, 2)
{'designType': 'fast', 'N': 256, 'M': 4, 'blockSize': 256, 'seed': 123}
[[0.9879 0.01  ]]
[[0.9901 0.0101]]
```

当 `nInput=2` 时，`FASTDesign` 返回 `N * nInput = 512` 行。

## Morris 筛选

`Morris` 是筛选方法。它不追求完整的方差分解，而是沿轨迹估计 elementary effects。

| 模式 | 含义 |
|---|---|
| `mu_star` 高 | 变量整体影响强。 |
| `sigma` 高 | 变量可能存在非线性影响，或参与交互。 |
| `mu_star` 低 | 对当前输出而言，变量可能不活跃。 |

```python
import numpy as np

from UQPyL.analysis import Morris
from UQPyL.doe import MorrisDesign
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] + 0.5 * X[:, 1] + 0.0 * X[:, 2]
    return y.reshape(-1, 1)


problem = Problem(nInput=3, nObj=1, lb=0.0, ub=1.0, objFunc=objFunc, optType="min", name="Screening3D", xLabels=["rain", "soil", "unused"])

X, meta = MorrisDesign(numLevels=4).sampleWithMeta(problem, numTrajectory=8, seed=123)
Y = problem.evaluate(X).objs

result = Morris(verboseFlag=False).analyze(problem, X, Y=Y, meta=meta)

print(X.shape)
print(meta)
print(result.metricNames)
print(result.getMetric("mu_star").values)
print(result.getMetric("sigma").values)
print(result.getMetric("S1_norm").values)
```

Example output:

```text
(32, 3)
{'designType': 'morris', 'numTrajectory': 8, 'numLevels': 4, 'trajectorySize': 4, 'seed': 123}
['mu', 'mu_star', 'sigma', 'S1_norm']
[[1.  0.5 0. ]]
[[0. 0. 0.]]
[[0.6667 0.3333 0.    ]]
```

解释：`rain` 最强，`soil` 次之，`unused` 不活跃。`sigma` 为零，因为这个示例模型是线性的，没有交互。

当 `nInput=3` 时，每条 Morris 轨迹有 `nInput + 1 = 4` 行。`numTrajectory=8` 时，样本矩阵有 `32` 行。

## 分析已有输出

当模型评估很贵时，保存 `Y` 并重复使用。

```python
import numpy as np

from UQPyL.analysis import DeltaTest, RSA
from UQPyL.doe import LHS
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.2 * X[:, 1] ** 2
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D", xLabels=["x1", "x2"])

X = LHS("classic").sample(problem, nSamples=128, seed=123)
Y = problem.evaluate(X).objs

deltaResult = DeltaTest(nNeighbors=2, verboseFlag=False).analyze(problem, X, Y=Y)
rsaResult = RSA(nRegion=4, verboseFlag=False).analyze(problem, X, Y=Y)

print(deltaResult.metricNames)
print(deltaResult.getMetric("S1_norm").values)
print(rsaResult.metricNames)
print(rsaResult.getMetric("S1_norm").values)
```

Example output:

```text
['S1', 'S1_norm']
[[ 1.0331 -0.0331]]
['S1', 'S1_norm']
[[0.7945 0.2055]]
```

有限样本和近邻估计会带来噪声，所以 `DeltaTest` 的归一化值可能出现负数。可以把它作为筛选信号；如果某些变量很重要，再用更大样本或方差型方法确认。

## 选择目标和输出列

用 `target` 选择要分析的输出块。

| `target` | 含义 |
|---|---|
| `"objs"` | 分析目标输出。 |
| `"cons"` | 分析约束输出。 |

用 `index` 选择输出列。

| `index` | 含义 |
|---|---|
| `"all"` | 分析所选输出块中的全部列。 |
| `0` | 只分析第一列输出。 |
| `[0, 2]` | 分析指定输出列。 |

下面这个例子有两个目标列，只分析第二个目标。

```python
import numpy as np

from UQPyL.analysis import RBDFAST
from UQPyL.doe import LHS
from UQPyL.problem import Problem

np.set_printoptions(precision=4, suppress=True)


def objFunc(X):
    X = np.atleast_2d(X)
    first = X[:, 0] ** 2
    second = X[:, 1] ** 2
    return np.column_stack([first, second])


problem = Problem(nInput=2, nObj=2, lb=-1.0, ub=1.0, objFunc=objFunc, optType=["min", "min"], xLabels=["x1", "x2"])

X = LHS("classic").sample(problem, nSamples=128, seed=123)
Y = problem.evaluate(X).objs

result = RBDFAST(verboseFlag=False).analyze(problem, X, Y=Y, target="objs", index=1)

print(result.getMetric("S1").rowLabels)
print(result.getMetric("S1").values)
```

Example output:

```text
['obj1']
[[0.0092 0.9801]]
```

行标签显示为 `obj1`，是因为分析结果中只包含一个被选中的输出列。这个被选中的列对应原始问题中的第二个目标。

## 查看 verbose 过程

需要简洁运行过程时，设置 `verboseFlag=True`。

```python
import numpy as np

from UQPyL.analysis import RBDFAST
from UQPyL.doe import LHS
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    y = X[:, 0] ** 2 + 0.2 * X[:, 1] ** 2
    return y.reshape(-1, 1)


problem = Problem(nInput=2, nObj=1, lb=-1.0, ub=1.0, objFunc=objFunc, optType="min", name="WeightedSphere2D", xLabels=["x1", "x2"])

X = LHS("classic").sample(problem, nSamples=64, seed=123)
result = RBDFAST(verboseFlag=True).analyze(problem, X)
```

Example output:

```text
Analysis: RBDFAST
Problem: WeightedSphere2D
nInput: 2
nOutput: 1
params: {'M': 4}
Analysis finished
runtime: 0.001s
[S1]
obj1: x1=9.3350e-01  x2=1.0117e-01
```

如果脚本中只需要结果对象，通常设置 `verboseFlag=False`。

## 读取 `AnaResult`

每个分析方法都会返回一个 `AnaResult`。

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

metric = result.getMetric("S1")

print(result.summary()["metric_names"])
print(metric.name)
print(metric.values)
print(metric.rowLabels)
print(metric.colLabels)
```

Example output:

```text
['S1']
S1
[[0.951  0.0475]]
['obj1']
['x1', 'x2']
```

`AnaResult` 汇总一次分析运行产生的全部指标。

| 字段或方法 | 含义 |
|---|---|
| `result.method` | 分析方法名，例如 `"RBDFAST"` 或 `"Sobol"`。 |
| `result.target` | 被分析的输出块，例如 `"objs"` 或 `"cons"`。 |
| `result.metricNames` | 可用指标名称。 |
| `result.getMetric("S1")` | 返回一个 `AnaMetric`。 |
| `result["S1"]` | `result.getMetric("S1")` 的简写。 |
| `result.X` | 记录的输入矩阵。 |
| `result.Y` | 记录的输出矩阵。 |
| `result.summary()` | 紧凑的运行摘要。 |

`AnaMetric` 存储一个指标矩阵。

| 字段 | 含义 |
|---|---|
| `metric.name` | 指标名称，例如 `S1`、`ST` 或 `mu_star`。 |
| `metric.values` | 数值矩阵。行是输出，列是变量或变量对。 |
| `metric.rowLabels` | 输出标签，例如 `obj1` 或 `con1`。 |
| `metric.colLabels` | 输入标签，例如 `x1`、`rain` 或 `soil`。 |
| `metric.colDim` | 列类型。`decsDim1` 表示每列一个输入变量；`decsDim2` 表示每列一个变量对。 |

## 读取保存的 SQLite 结果

设置 `saveFlag=True` 后，会在 `Result/` 下保存 sqlite 文件。

```python
from pathlib import Path

import numpy as np

from UQPyL.analysis import RBDFAST
from UQPyL.analysis.runtime import AnaReader
from UQPyL.doe import LHS
from UQPyL.problem import Sphere

np.set_printoptions(precision=4, suppress=True)


problem = Sphere(nInput=3, ub=1.0, lb=-1.0)
X = LHS("classic").sample(problem, nSamples=64, seed=123)
Y = problem.evaluate(X).objs

resultDir = Path("Result")
before = set(resultDir.glob("*.sqlite3")) if resultDir.exists() else set()

RBDFAST(verboseFlag=False, logFlag=False, saveFlag=True).analyze(problem, X, Y=Y)

after = set(resultDir.glob("*.sqlite3"))
dbPath = sorted(after - before)[0]

with AnaReader(dbPath) as reader:
    summary = reader.get_run_summary()
    params = reader.get_run_params()
    metric = reader.get_metric("S1")
    loaded = reader.load_result()

print(dbPath.as_posix())
print(summary["method"], summary["problem_name"], summary["target"])
print(summary["metric_names"])
print(params["M"])
print(metric.values)
print(loaded.X.shape, loaded.Y.shape)
```

Example output:

```text
Result/rbdfast_Sphere_YYYYMMDD_HHMM_xxxx.sqlite3
RBDFAST Sphere objs
['S1']
4
[[0.1853 0.2958 0.3061]]
(64, 3) (64, 1)
```

如果你已经知道 sqlite 路径，只需要 reader 这部分：

```text
from UQPyL.analysis.runtime import AnaReader

dbPath = "Result/rbdfast_Sphere_YYYYMMDD_HHMM_xxxx.sqlite3"

with AnaReader(dbPath) as reader:
    summary = reader.get_run_summary()
    result = reader.load_result()
```

保存的运行会包含序列化的问题对象。如果你在 notebook 单元格或临时脚本里临时定义 `objFunc`，Python 可能无法 pickle 它。需要长期保存 sqlite 结果时，建议使用可导入的问题类，或把目标函数定义在可导入的 Python 模块中。

## 常见指标怎么解释

不同方法会产生不同指标。

| 指标 | 由哪些方法产生 | 含义 |
|---|---|---|
| `S1` | `Sobol`, `FAST`, `RBDFAST`, `RSA`, `DeltaTest`, `MARS` | 一阶或方法特定的单变量贡献。通常越大越重要。 |
| `S1_norm` | `Sobol`, `FAST`, `RSA`, `DeltaTest`, `Morris`, `MARS` | 按行归一化的贡献，适合在同一个输出内给变量排序。 |
| `ST` | `Sobol`, `FAST` | 总阶贡献，包括与其他变量的交互影响。 |
| `ST_norm` | `Sobol`, `FAST` | 按行归一化的总阶贡献。 |
| `S2` | `secondOrder=True` 的 `Sobol` | 两两变量的二阶贡献。列是输入变量对。 |
| `mu` | `Morris` | elementary effect 的均值，符号可提示影响方向。 |
| `mu_star` | `Morris` | elementary effect 绝对值的均值，常用于 Morris 变量排序。 |
| `sigma` | `Morris` | elementary effects 的离散程度。越大越可能存在非线性或交互。 |

这些指标更适合用作变量重要性排序，而不是精确的物理常数。敏感性估计依赖 `problem.lb` 和 `problem.ub` 定义的输入不确定性范围。

## 常见错误

| 错误 | 会发生什么 | 修正方式 |
|---|---|---|
| 给 `Sobol`、`FAST` 或 `Morris` 传样本但不传 `meta` | 方法无法解释样本行结构。 | 使用 `X, meta = sampler.sampleWithMeta(...)`，并传入 `meta=meta`。 |
| `X` 来自一个采样器，`meta` 来自另一个采样器 | 结果无效，或验证失败。 | 保持同一次调用返回的 `X` 和 `meta` 配套使用。 |
| 评估后单独打乱 `Y` | 输入和输出不再逐行对应。 | 保持 `X[i]` 与 `Y[i]` 对齐。 |
| 传入形状不明确的一维 `Y` | 方法可能会 reshape，但意图不清楚。 | 优先使用形状为 `(n_samples, n_outputs)` 的 `Y`。 |
| 样本量太少 | 变量排序可能很噪声。 | 小样本只用于流程检查，正式解释要增加样本量。 |
| 比较不同边界下的敏感性 | 结果会变，因为输入不确定性范围变了。 | 每个分析结果都记录 `lb` 和 `ub`。 |
| 把 `S1_norm` 当成绝对物理贡献 | 它是为了同一输出内排序而归一化的值。 | 用它比较同一结果行内的变量。 |
| 混淆 `S1` 和 `ST` | 可能漏掉交互影响。 | 如果 `ST` 明显大于 `S1`，说明交互或非线性可能重要。 |
| 保存交互式定义的 `objFunc` | `saveFlag=True` 时 pickle 可能失败。 | 保存 sqlite 运行时，使用可导入的问题类或函数。 |

## 下一步

| 目标 | 阅读 |
|---|---|
| 生成兼容的样本 | [试验设计](doe.md) |
| 定义被评估的系统 | [Problem](problem.md) |
| 查看构造参数和结果字段 | [Analysis API](api/analysis.md) |
| 查看完整工作流 | [Examples](examples.md) |
