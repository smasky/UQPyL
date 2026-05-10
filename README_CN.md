# UQPyL：Uncertainty Quantification Python Lab

<p align="center"><img src="./docs_v2/assets/logo.png" width="400"/></p>

[![PyPI version](https://badge.fury.io/py/UQPyL.svg?icon=si%3Apython&icon_color=%2331aadd)](https://badge.fury.io/py/UQPyL) [![CI](https://github.com/smasky/UQPyL/actions/workflows/ci.yml/badge.svg)](https://github.com/smasky/UQPyL/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/smasky/UQPyL/branch/dev/graph/badge.svg)](https://codecov.io/gh/smasky/UQPyL) ![PyPI - Downloads](https://img.shields.io/pypi/dm/UQPyL) ![PyPI - License](https://img.shields.io/pypi/l/UQPyL) ![GitHub last commit](https://img.shields.io/github/last-commit/smasky/UQPyL) ![Static Badge](https://img.shields.io/badge/Author-wmtSky-orange) ![Static Badge](https://img.shields.io/badge/Contact-wmtsmasky%40gmail.com-blue)

[English](README.md) | [中文](README_CN.md) | [Documentation](https://uqpyl.readthedocs.io)

UQPyL 是一个面向不确定性量化、优化、推断、率定和代理建模的 Python 库。
它强调问题定义一次，然后在不同 UQ 工作流中复用。

## UQPyL 解决什么问题

UQPyL 为常见的不确定性量化工作流提供统一的问题接口。

先定义模型或决策问题，然后将同一个问题对象用于：

- 试验设计
- 敏感性与不确定性分析
- 优化
- Bayesian 风格推断
- 模型率定
- 代理建模

问题定义完成后，UQPyL 的各个模块可以围绕同一个对象互通使用。

<p align="center">
  <img src="./docs_v2/assets/architecture.png" alt="UQPyL 架构概览" width="1000"/>
</p>

## Problem 抽象

`Problem` 是 UQPyL 的主要入口。它把模型、基准函数或决策任务组织成一个可复用对象，供其他模块使用。

构建一个 `Problem` 时，需要定义：

| 部分 | 作用 |
|---|---|
| 输入空间 | 变量个数、边界、标签和变量类型。 |
| 评估规则 | 输入样本如何转换成目标值和可选约束。 |
| 附加信息 | 优化方向、问题名称，以及服务于算法、输出、日志和保存结果的元数据。 |

例如：

```python
import numpy as np

from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(
    # 输入空间
    nInput=2, lb=-1.0, ub=1.0,

    # 评估规则
    nObj=1, objFunc=objFunc,

    # 附加信息
    optType="min", name="Sphere2D",
)
```

对于只需要目标值或约束值的工作流，这已经足够，包括 DOE、分析、优化、推断和代理建模。

对于很多水文模拟问题，工作流需要的不只是最终目标值。率定和不确定性分析可能需要在方法执行过程中保留模拟序列、观测序列和有效观测位置。

`ModelProblem` 就是在这种情况下对 `Problem` 的扩展。它增加：

| 额外部分 | 作用 |
|---|---|
| `simFunc` | 运行模型并返回模拟序列或模拟场。 |
| `obs` / `mask` | 保存观测数据，并标记用于模拟-观测对比的有效位置。 |

默认使用 `Problem`。当方法需要模拟过程语义时，例如需要比较 `sim` 和 `obs` 的率定方法，再使用 `ModelProblem`。具体用法请参考 [文档](https://uqpyl.readthedocs.io)。

<p align="center">
  <img src="./docs_v2/assets/Problem.webp" alt="Problem 与 ModelProblem 对比" width="1000"/>
</p>

对于水文应用来说，难点通常不在 UQ 算法本身，而在连接外部模型、准备输入、运行模拟和收集输出。[hydroPilot](https://github.com/smasky/hydroPilot) 面向这一层模型运行管理。需要让 hydroPilot 管理水文模型运行、让 UQPyL 负责采样、分析、率定、优化、推断或代理建模时，可以把两者配合使用。

## 架构概览

UQPyL 围绕统一的 `problem` 抽象组织，并在其上构建一套功能模块。

| 类型 | 模块 | 作用 |
|---|---|---|
| Core | `problem` | 定义参数空间、评估规则、目标、约束、模拟及相关元数据。 |
| Function | `doe` | 为实验、分析、初始化和建模生成设计样本。 |
| Function | `analysis` | 分析输入变量如何影响模型或目标输出。 |
| Function | `optimization` | 搜索单目标、多目标或昂贵模型最优解。 |
| Function | `inference` | 执行 MCMC 风格参数推断。 |
| Function | `calibration` | 基于观测率定模拟模型。 |
| Function | `surrogate` | 为高代价评估训练与评估代理模型。 |
| Support | `runtime`、`viz` | 保存结构化运行结果、日志、中间状态，并提供可视化工具。 |

## 快速开始示例

### 使用 `Problem` 做优化

```python
import numpy as np

from UQPyL.problem import Problem
from UQPyL.optimization.soea import SCE_UA


def objFunc(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1, keepdims=True)


problem = Problem(
    nInput=2, nObj=1,
    ub=1.0, lb=-1.0,
    objFunc=objFunc, optType="min",
    name="Sphere2D",
)

algorithm = SCE_UA(maxFEs=200, verboseFlag=False, logFlag=False, saveFlag=False)

result = algorithm.run(problem, seed=123)

print(result.bestDecs)
print(result.bestObjs)
```

### 使用 `ModelProblem` 做率定

```python
import numpy as np

from UQPyL.calibration import GLUE
from UQPyL.problem import ModelProblem

obs = np.array([[1.0], [2.0], [3.0]])


def simFunc(X):
    X = np.atleast_2d(X)
    # 这里假设每组参数都会返回 1 条模拟序列，
    # 一共有 3 个时刻，因此返回尺寸是 (n_samples, 3, 1)。
    sim = np.zeros((X.shape[0], 3, 1))
    sim[:, 0, 0] = X[:, 0] * 1.0
    sim[:, 1, 0] = X[:, 0] * 2.0
    sim[:, 2, 0] = X[:, 0] * 3.0
    return sim


problem = ModelProblem(
    nInput=1, nObj=1,
    lb=0.0, ub=2.0,
    simFunc=simFunc, obs=obs,
    name="LinearScaleModel",
)

X = np.linspace(0.5, 1.5, 32).reshape(-1, 1)
result = GLUE(metric="rmse", verboseFlag=False).run(problem, X, threshold=0.2)
```

方法会返回结构化结果对象，例如 `OptResult`、`AnaResult`、`InfResult` 或 `CalResult`。

## 模块速览

下表只列出代表性方法，不是完整 API。完整的模块用法和 API 细节请参考 [文档](https://uqpyl.readthedocs.io)。

| 模块 | 代表性方法 |
|---|---|
| `doe` | `LHS`、`FFD`、`Random`、`Sobol`、`SaltelliDesign`、`FASTDesign`、`MorrisDesign` |
| `analysis` | `Sobol`、`FAST`、`RBDFAST`、`Morris`、`RSA`、`DeltaTest`、`MARS` |
| `optimization` | `GA`、`PSO`、`DE`、`SCE_UA`、`NSGAII`、`NSGAIII`、`MOEAD`、`RVEA`、`EGO` |
| `inference` | `MH`、`AMH`、`MH_Gibbs`、`DEMC`、`DREAM_ZS` |
| `calibration` | `GLUE`、`SUFI2`、`ES`、`IES` |
| `surrogate` | `RBF`、`GPR`、`KRG`、`LinearRegression`、`PolynomialRegression`、`AutoTuner` |

## 运行输出与保存

大多数可运行方法都共享三个常见运行期控制选项。

| 选项 | 作用 |
|---|---|
| `verboseFlag` | 在终端打印进度和摘要。 |
| `logFlag` | 在支持时写出更完整的运行日志。 |
| `saveFlag` | 保存结构化结果，通常是 sqlite。 |

```python
algorithm = SCE_UA(maxFEs=200, verboseFlag=True, logFlag=True, saveFlag=True)
result = algorithm.run(problem, seed=123)
```

终端输出示例如下：

```text
Algorithm: SCE-UA
Problem: Sphere2D
nInput: 2
nObj: 1
maxFEs: 200
maxIters: 1000
SCE-UA | iter=10 eval=84 best=4.3210e-03 cv=0 time=0.0s
SCE-UA | iter=20 eval=154 best=2.1500e-04 cv=0 time=0.0s
Optimization finished
  algorithm        : SCE-UA
  status           : finished
  iterations       : 27
  evaluations      : 203
  best value       : 1.0000e-04
  best X           : [1.0000e-02, -0.0000e+00]
  constraint viol. : 0
  elapsed          : 0.0s
```

启用 `saveFlag=True` 后，运行可以产出供对应 reader 后续读取的持久化结果。

| 保存选项 | 规则 |
|---|---|
| `saveFlag=True` | 为当前运行启用结构化结果保存。 |
| `saveFreq` | 对优化方法，会每隔 `saveFreq` 次迭代保存一次中间快照，并在结束时始终保存最终结果。 |
| Reader 读取 | 保存下来的 sqlite 结果后续可以通过 `OptReader`、`AnaReader`、`CalReader` 等模块 reader 读取。 |

## 更多示例

使用 `Sobol` 做敏感性分析：

```python
import numpy as np

from UQPyL.analysis import Sobol
from UQPyL.doe import SaltelliDesign
from UQPyL.problem import Problem


def objFunc(X):
    X = np.atleast_2d(X)
    y = np.sin(X[:, 0]) + 7 * np.sin(X[:, 1])**2 + 0.1 * X[:, 2]**4 * np.sin(X[:, 0])
    return y[:, None]


problem = Problem(
    nInput=3, nObj=1,
    lb=-np.pi, ub=np.pi,
    objFunc=objFunc, name="Ishigami",
)

X, meta = SaltelliDesign(secondOrder=True).sampleWithMeta(problem, 512)
Y = problem.evaluate(X, target="objs").objs
result = Sobol(verboseFlag=False).analyze(problem, X, Y, meta=meta, target="objs")
```

## 安装

UQPyL 需要 Python 3.8 或更高版本。

```bash
pip install -U UQPyL
```

如果需要绘图工具：

```bash
pip install -U "UQPyL[viz]"
```

从源码安装：

```bash
git clone https://github.com/smasky/UQPyL.git
cd UQPyL
pip install .
```

## 引用

引用 UQPyL 2.0 时，请使用以下预印版：

Wu, M., Sun, R., Xu, P., Yang, X., Hu, P., & Duan, Q. UQPyL 2.0: An Open-Source Python Package for Uncertainty Quantification and Optimization. Available at SSRN: https://ssrn.com/abstract=5393295 or http://dx.doi.org/10.2139/ssrn.5393295

```bibtex
@misc{wu2025uqpyl2,
  title = {UQPyL 2.0: An Open-Source Python Package for Uncertainty Quantification and Optimization},
  author = {Wu, Mengtian and Sun, Ruochen and Xu, Pengcheng and Yang, Xu and Hu, Pengjie and Duan, Qingyun},
  year = {2025},
  note = {SSRN preprint},
  doi = {10.2139/ssrn.5393295},
  url = {https://ssrn.com/abstract=5393295}
}
```

引用 UQPyL 1.0 时，请使用：

Wang, C., Duan, Q., Tong, C. H., Di, Z., & Gong, W. (2016). A GUI platform for uncertainty quantification of complex dynamical models. *Environmental Modelling & Software*, 76, 1-12. https://doi.org/10.1016/j.envsoft.2015.11.004

```bibtex
@article{wang2016uqpyl,
  title = {A GUI platform for uncertainty quantification of complex dynamical models},
  author = {Wang, Chen and Duan, Qingyun and Tong, Charles H. and Di, Zhenhua and Gong, Wei},
  journal = {Environmental Modelling & Software},
  volume = {76},
  pages = {1--12},
  year = {2016},
  doi = {10.1016/j.envsoft.2015.11.004}
}
```

## 贡献

欢迎贡献。适合补充的方向包括新算法、模型接口、基准问题、示例、测试和文档改进。

## 许可证

UQPyL 基于 MIT License 发布。详见 [LICENSE.md](LICENSE.md)。
