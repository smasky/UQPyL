# UQPyL: 参数不确定性分析及优化工具包

<p align="center"><img src="./docs/UQ.svg" width="400"/></p>

[![PyPI version](https://badge.fury.io/py/UQPyL.svg?icon=si%3Apython&icon_color=%2331aadd)](https://badge.fury.io/py/UQPyL) ![PyPI - Downloads](https://img.shields.io/pypi/dm/UQPyL) ![PyPI - License](https://img.shields.io/pypi/l/UQPyL) ![GitHub last commit](https://img.shields.io/github/last-commit/smasky/UQPyL) ![Static Badge](https://img.shields.io/badge/Author-wmtSky-orange) ![Static Badge](https://img.shields.io/badge/Contact-wmtsmasky%40gmail.com-blue)

UQPyL是一个功能全面的Python工具包，专注参数不确定性分析与优化，广泛适用数值模型校准、资源优化调度、产品设计等各类工程问题。该工具包目前已形成完整的方法体系，包括实验设计 (Design of Experiments)、敏感性分析 (Sensitivity Analysis) 以及支持单目标与多目标的参数优化。此外，UQPyL还内置了替代模型 (Surrogate Models) 模块，可用于计算代价昂贵问题 (Computational Expensive Problem)的高效求解。

👉[English Doc](./README.md)

## Contents

- [功能特点](#-功能特点)
- [安装指南](#-安装指南)
- [实用链接](#-实用链接)
- [方法预览](#-方法预览)
   - [敏感性分析方法](#-敏感性分析方法)
   - [优化算法](#-优化算法)
   - [替代模型](#-替代模型)
   - [单目标优化基准问题](#-单目标优化基准问题)
   - [多目标优化基准问题](#-多目标优化基准问题)
- [快速开始](#-快速开始)
- [欢迎合作](#-欢迎合作)
- [联系方式](#-联系方式)

## ✨ 功能特点
1. **集成主流敏感性分析与优化方法**: 实现了当前广泛使用的敏感性分析方法和优化算法，满足多样化求解需求。
2. **支持运行过程可视化与结果存储**: 可记录执行历史，并自动保存分析结果，便于用户回溯分析流程、管理输出数据。
3. **内置先进替代模型与自动调优工具**: 集成多种替代模型，并支持自动化参数优化，以提升模型效果。
4. **应用资源全面覆盖**: 提供丰富的基准测试问题与实际应用案例，便于用户快速入门与方法验证。(👉 近期规划：针对水科学领域，我们计划开发定制化程序接口，将水利模型与UQPyL平台深度融合，以增强其在实际工程中的适用性(例如：[SWAT-UQ](https://github.com/smasky/SWAT-UQ))。欢迎有兴趣的研究者与我们开展合作。)
5. **模块化与可扩展架构**: 构建了统一的敏感性分析与优化架构，支持用户灵活扩展与自定义新方法(我们诚挚欢迎各类贡献，共同推动 UQPyL 的发展)。

## ⚙️ 安装指南
![Static Badge](https://img.shields.io/badge/Python-3.6%2C%203.7%2C%203.8%2C%203.9%2C%203.10%2C%203.11%2C%203.12-blue) ![Static Badge](https://img.shields.io/badge/OS-Windows%2C%20Linux-orange)

推荐使用PyPi或者Conda安装:

```bash
pip install -U UQPyL
```

```bash
conda install UQPyL --upgrade
```

或者:

```bash
git clone https://github.com/smasky/UQPyL.git 
cd UQPyL
pip install .
```

## 🔗 实用链接

- **官网网站**: [参数敏感性分析及优化实验室](http://www.uq-pyl.com) (**TODO**: 需要更新)
- **开源代码**: [GitHub 仓库](https://github.com/smasky/UQPyL/)
- **官方文档**: [查看文档](https://uqpyl.readthedocs.io/en/latest/) (**TODO**: 正在更新中... )
- **文章引用**: UQPyL 2.0(**TODO**: 需要更新), [UQPyL 1.0](https://www.sciencedirect.com/science/article/pii/S1364815215300955)

---

## 🎉 方法预览

### 敏感性分析方法

| 简称 | 全称 | 引用 |
| -------|------------|----------|
| Sobol' | \ |[Sobol(2010)](https://www.sciencedirect.com/science/article/pii/S0378475400002706), [Saltelli (2002)](https://www.sciencedirect.com/science/article/pii/S0010465502002801)|
| DT| Delta Test| [Eirola et al. (2008)](https://www.semanticscholar.org/paper/Using-the-Delta-Test-for-Variable-Selection-Eirola-Liiti%C3%A4inen/fa131898bbd99e848e706837f4072a310e1109e5?p2df)|
| FAST | Fourier Amplitude Sensitivity Test | [Cukier et al. (1973)](https://pubs.aip.org/aip/jcp/article-abstract/59/8/3873/533535/Study-of-the-sensitivity-of-coupled-reaction), [Saltelli et al. (1999)](https://amstat.tandfonline.com/doi/abs/10.1080/00401706.1999.10485594)|
| RBD-FAST| Random Balance Designs Fourier Amplitude Sensitivity Test | [Tarantola et al. (2006)](https://www.sciencedirect.com/science/article/pii/S0951832005001444), [Tissot, Prieur (2012)](https://www.sciencedirect.com/science/article/pii/S0951832012001159)
|MARS-SA|  Multivariate Adaptive Regression Splines for Sensibility Analysis |[Friedman, (1991)](https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-1/Multivariate-Adaptive-Regression-Splines/10.1214/aos/1176347963.full)|
|Morris| \ |[Morris, (2012)](https://www.tandfonline.com/doi/abs/10.1080/00401706.1991.10484804)|
|RSA| Regional Sensitivity Analysis | [Hornberger, Spear, (1981)](https://www.osti.gov/biblio/6396608), [Pianosi (2016)](https://www.sciencedirect.com/science/article/pii/S1364815216300287) |


💡 **提示:** 当前上述方法均支持使用替代模型，缓解高计算成本带来的挑战。

🚀 **致谢:**  在开发敏感性分析模块过程中，部分方法参考了[SALib](https://github.com/SALib/SALib)项目，特此致谢其贡献。

### 优化算法

| 简称 | 全称 |   标签   |  引用  |
|--------------|-----------| ----------|---------------|
| SCE-UA | Shuffled Complex Evolution| Single | [Duan et al. (1992)](https://link.springer.com/article/10.1007/BF00939380)|
| ML-SCE-UA| M&L Shuffled Complex Evolution| Single | [Muttil, Liong (2006)](https://www.worldscientific.com/doi/abs/10.1142/9789812707208_0036) |
| GA | Genetic Algorithm| Single | [Holland (1992)](https://direct.mit.edu/books/monograph/2574/Adaptation-in-Natural-and-Artificial-SystemsAn)|
| CSA | Cooperation Search Algorithm | Single | [Feng et al. (2021)](https://www.sciencedirect.com/science/article/pii/S1568494620306724) |
| PSO | Particle Swarm Optimization | Single | [Kennedy and Eberhart (1995)](https://ieeexplore.ieee.org/abstract/document/488968/) |
| DE | Differential Evolution | Single | [Storn and Price (1997)](https://link.springer.com/article/10.1023/a:1008202821328) |
| ABC |Artificial Bee Colony | Single | [Karaboga (2005)](https://abc.erciyes.edu.tr/pub/tr06_2005.pdf) |
| ASMO | Adaptive Surrogate Modelling based Optimization | Single, Surrogate | [Wang et al.(2014)](https://www.sciencedirect.com/science/article/pii/S1364815214001698) |
| EGO | Efficient Global Optimization | Single, Surrogate | [Jones (1998)](https://link.springer.com/article/10.1023/A:1008306431147)
| MOEA/D | Multi-objective Evolutionary Algorithm based on Decomposition | Multiple | [Zhang, Li (2007)](https://ieeexplore.ieee.org/document/4358754)|
| NSGA-II| Nondominated Sorting Genetic Algorithm II | Multiple | [Deb et al. (2002)](https://ieeexplore.ieee.org/document/996017)|
| NSGA-III| Nondominated Sorting Genetic Algorithm III| Multiple | [Deb, Jain (2014)](https://ieeexplore.ieee.org/document/6600851)|
| RVEA | Reference Vector guided Evolutionary Algorithm | Multiple | [Cheng et al. (2016)](https://ieeexplore.ieee.org/document/7386636)|
|MO-ASMO|Multi-Objective Adaptive Surrogate Modelling-based Optimization| Multiple, Surrogate | [Gong et al. (2015)](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2015WR018230)|  

(标签 `Surrogate` 表示该算法可用于解决计算代价昂贵问题)

💡 **提示:** 我们正在不断更新和完善模块中的算法。如果您有特定需求，欢迎随时与我们联系！

### 替代模型

| 简称 | 全称 | 特点 |
|--------------|-----------|----------|
| KRG | Kriging | 支持 `guass`, `cubic`, `exp` 等核函数 |
| GP | Gaussian Process | 支持 `const`, `rbf`, `dot`, `matern`, `rq` 等核函数 |
| LR | Linear Regression | 支持 `origin`, `ridge`, `lasso` 等损失函数|
| PR | Polynomial Regression | 支持 `origin`, `ridge`, `lasso` 等损失函数|
| RBF | Radial Basis Function |支持 `cubic`, `guass`, `linear`, `mq`, `tps` 等核函数以及它们对应的超参数|
| SVM | Support Vector Machine | 使用 [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) 作为核心库 |
| MARS | Multivariate Adaptive Regression Splines | 使用 [Earth](http://www.milbo.users.sonic.net/earth/) 作为核心库 |

❤️ 为进一步优化使用体验，UQPyL提供了替代模型的**自动校准工具**，无需手动设定超参数，即可实现模型的高效构建。

### 单目标优化基准问题

| 名称 | 公式 | 最优解 | 最优值 | 
|------|---------|------------------|--------|
|Sphere| <img src="./docs/pic/Sphere.svg" /> | ( 0, 0, 0 ... 0 ) | 0.0 |
|Schwefel_2_22| <img src="./docs/pic/Schwefel_2_22.svg" /> | ( 0, 0, 0 ... 0 ) | 0.0 |
|Schwefel_1_22| <img src="./docs/pic/Schwefel_1_22.svg" /> | ( 0, 0, 0 ... 0 ) | 0.0 |
|Schwefel_2_21| <img src="./docs/pic/Schwefel_2_21.svg" /> | ( 0, 0, 0 ... 0 ) | 0.0 |
|Schwefel_2_26 | <img src="./docs/pic/Schwefel_2_26.svg" /> | (420.9687 ... 420.9687) | -12569.5 |
| Rosenbrock | <img src="./docs/pic/Rosenbrock.svg" /> | ( 0, 0, 0 ... 0 ) | 0.0 |
| Step | <img src="./docs/pic/Step.svg" /> | ( 1, 1, 1 ... 1) | 0.0 |
| Quartic | <img src="./docs/pic/Quartic.svg" /> | ( 1, 1, 1 ... 1) | 0.0 |
| Rastrigin | <img src="./docs/pic/Rastrigin.svg" /> | ( 0, 0, 0 ... 0 ) | 0.0 |
| Ackley | <img src="./docs/pic/Ackley.svg" /> | ( 0, 0, 0 ... 0 ) | 0.0 |
| Griewank | <img src="./docs/pic/Griewank.svg" /> | ( 0, 0, 0 ... 0) | 0.0 |
| Trid | <img src="./docs/pic/Trid.svg" /> | <img src="./docs/pic/Trid_solution.svg">| `-D(D+4)(D-1)/6` |
| Bent_Cigar | <img src="./docs/pic/Bent_Cigar.svg" /> |(0, 0, 0 ... 0) | 0.0 |
| Discus | <img src="./docs/pic/Discus.svg" /> | (0, 0, 0 ... 0) | 0.0 |
| Weierstrass | <img src="./docs/pic/Weierstrass.svg" /> | (0, 0, 0 ... 0) | 0.0 |

### 多目标优化基准问题

| 名称 | 目标数量 | 帕累托前沿特征 | 问题特性 |
|------|-------------------|---------------------------|---------|
| ZDT1 |         2         |           Line            | Convex  |
| ZDT2 |         2         |           Line            | Concave |
| ZDT3 |         2         |           Line            | Disconnected |
| ZDT4 |         2         |           Line            | Convex |
| ZDT6 |         2         |           Line            | Concave |
| DTLZ1 | >=3 (user define) |         Surface          | Multimodal |
| DTLZ2 | >=3 (user define) |         Surface          | Single-peaked |
| DTLZ3 | >=3 (user define) |         Surface          | Multimodal|
| DTLZ4 | >=3 (user define) |         Surface          | Multimodal|
| DTLZ5 | >=3 (user define) |         Line         | Multimodal|
| DTLZ6 | >=3 (user define) |         Line         | Multimodal|
| DTLZ7 | >=3 (user define) | Discrete Surface        | Multimodal|

### 实际问题

**TODO:** 正在计划将一些常见的水文模型校准案例（如 SWAT、SAC 等）或相关水资源优化调度问题纳入UQPyL。

---

## 🍭 快速开始

### 问题定义
为了高效使用UQPyL, 需要定义拟解决的问题，包括：

1. **提供问题基本信息**，例如决策变量的维度，优化目标的个数，每个变量取值范围以及变量类型，参数名称等。
2. **定义目标函数**，即说明如何根据决策变量`x`获得输出目标`obj`。在UQPyL，目标函数被命名为`objFunc`。这个函数可以是解析函数，也可以包含计算模型，或外部黑盒过程。如有约束条件，也需要定义相应的约束函数，命名为`conFunc`。

以下问题是一个Rosenbrock函数的变体，它在原始函数的基础上增加了一个约束条件($x_1^2+x_2^2+x_3^2 \ge 4$)，并更改了变量的类型：原本都是连续型，即float类型，现在将 $x_2$ 设置为整数(int)， $x_3$ 设置为离散变量(discrete)。以此为例，具体说明问题定义的步骤。

<p align="center"><img src="./docs/pic/Problem1.svg" width=500/></p>

UQPyL提供了一个名为`Problem`的Python类，用于简化问题定义的工作流程。


<a href="https://nbviewer.org/github/smasky/UQPyL/blob/dev/notebooks/defefine_problem.ipynb" target="_blank">📘 在线查看 Jupyter Notebook 示例 </a>


```python
# 第1步: 从UQPyL的problems模块导入Problem类
from UQPyL.problems import Problem

# 第2步: 定义objFunc函数
# objFunc是一个接收numpy格式的二维矩阵X并返回numpy格式的二维矩阵objs的函数, 其中:
# 二维矩阵X的行向量代表一组决策变量，列向量则对应同一变量的不同取值
# 返回的矩阵objs的行数应与X相同，列数则与该问题的目标数相等。
# 具体来说:
# 对于单目标问题，二维矩阵objs的形状应为 (N, 1)
# 对于多目标问题，二维矩阵objs的形状应为 (N, M)
# 其中，N 表示输入的决策变量组合数，M 表示目标函数的个数。
# 用户应自行保证返回的矩阵objs的形状满足上述要求

def objFunc(X):
    
    # 如果条件允许，建议对矩阵X进行向量化操作，提升计算效率
    objs =100 * (X[:, 2] - X[:, 1]**2)**2+ 100 * (X[:, 1] - X[:, 0]**2)**2  + \
            (1 - X[:, 1])**2 + (1 - X[:, 0])**2 

    return objs[:, None] # 尽管UQPyL会做进一步检查，请自行确保返回的objs是二维矩阵。

# UQPyL 还支持另一种定义 objFunc 函数的方式。
# 对于涉及数值计算模型的问题，通常无法对矩阵 X 进行向量化操作。
# 为此，UQPyL 提供了装饰器函数 @singleFunc，用于启用“单例模式”。
# 在该模式下，objFunc 仅接收 Python list 或一维 numpy array 作为输入，
# 每次仅处理一组决策变量，适用于结构复杂或难以向量化的目标函数定义。

# 首先，从UQPyL的problems模块导入启用单例模式的装饰器
from UQPyL.problems import singleFunc

@singleFunc
def objFunc_(X): # 输入 X 应为 numpy 一维 array 或 Python list
    # 对 X 中的每个元素执行计算
    obj = 100 * (X[2] - X[1]**2)**2 + 100 * (X[1] - X[0]**2)**2 + \
            (1 - X[1])**2 + (1 - X[0])**2 
    return obj # 返回目标函数值：单目标优化时返回数值，多目标优化时返回一维 array 或 list

# 第3步：定义约束函数 conFunc
# 与 objFunc 类似，conFunc 也支持两种定义方式。
# 注意：conFunc 的返回值表示约束的满足情况：
# - 小于 0 表示约束被违反，且值越小，违反程度越严重；
# - 大于 0 表示满足约束，即为可行解。
# 因此，用户在建模时可能需要调整原始约束函数，以符合上述约定。

# 矢量模式
def conFunc(X):
    cons = X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2 - 4 
    return cons[:, None]

# 单例模式
@singleFunc
def conFunc(X):
    con = X[0]**2 + X[1]**2 + X[2]**2 - 4 
    return con

# 第4步：设置问题的基础信息

nInput = 3 # 决策变量的维度（输入维数），此处为3
nOutput = 1  # 目标函数数量（输出维数），此处为1

# 设置决策变量的取值范围
ub = [10, 10, 10] # 上界，可为 float、int、list 或 numpy 格式；此处所有变量上界均为10，也可设为 ub = 10

lb = [0, 0, 0] # 下界，同理可简写为 lb = 0


# 定义变量类型：0=连续型(float)、1=整数型(int)、2=离散型(discrete)
varType = [0, 1, 2] # 若未指定，默认所有变量为连续型（0）

# 指定离散型变量的可行取值集合
varSet = {2: [2, 3.4, 5.1, 7]} 
# 键为变量索引（从0开始），值为该变量允许的取值列表
# 例如：第3个变量 x3 只能取 2、3.4、5.1 或 7

# 设置优化方向：'min' 表示最小化，'max' 表示最大化
optType = 'min'

# 可选：为决策变量指定名称
xLabels = ['x1', 'x2', 'x3'] # 若未指定，将默认命名为 'x1', 'x2', ...

# 可选：为目标函数指定名称
yLabels = ['obj1']   # 若未指定，将默认命名为 'obj1', 'obj2', ...

# 可选：设置问题名称，用于标识、记录或保存结果
name = 'Rosenbrock'

# 第5步：实例化问题对象
problem = Problem(
    nInput=nInput,
    nOutput=nOutput,
    objFunc=objFunc,
    conFunc=conFunc,
    ub=ub,
    lb=lb,
    varType=varType,
    varSet=varSet,
    xLabels=xLabels,
    yLabels=yLabels,
    name=name
)
# Problem 类将收集以上信息，用于后续优化过程

# 第6步：使用优化算法求解问题
# UQPyL 提供多种优化算法，可通过读取 problem 实例获取所需信息
# 本例以遗传算法（GA）为例

from UQPyL.optimization.single_objective import GA

# 创建遗传算法实例（可选传参，此处使用默认参数）
ga = GA()

# 调用遗传算法的run方法进行求解
ga.run(problem = problem)
# 所有优化算法均提供统一的 .run(problem) 方法接口，用于执行优化。

# 输出:
# Time:  0.0 day | 0.0 hour | 0.0 minute |  1.17 second
# Used FEs:    50000  |  Iters:  999
# Best Objs and Best Decision with the FEs
# +-------------------+-------------------+-------------------+-------------------+
# |        FEs        |       Iters       |      OptType      |      Feasible     |
# +-------------------+-------------------+-------------------+-------------------+
# |         50        |         0         |        min        |        True       |
# +-------------------+-------------------+-------------------+-------------------+
# +-------------------+-------------------+-------------------+-------------------+
# |        obj1       |         x1        |         x2        |         x3        |
# +-------------------+-------------------+-------------------+-------------------+
# |      4.0e+02      |       0.000       |       0.000       |       2.000       |
# +-------------------+-------------------+-------------------+-------------------+
```

### 基准测试问题

UQPyL 提供多种内置的基准测试问题（均继承自 Problem 类），用于评估优化算法性能。
用户可灵活设定变量维度、取值范围等参数，以满足不同测试需求。

```python
from UQPyL.problems.single_objective import Sphere, Ackley
from UQPyL.problems.multi_objective import ZDT1, DTLZ1

# 单目标基准测试问题
problem1 = Sphere(nInput=10, ub=100, lb=-100)   # 10维 Sphere 问题，变量取值范围 [-100, 100]
problem2 = Ackley(nInput=10, ub=np.ones(10)*100, lb=np.ones(10)*-100)   # 10维 Ackley 问题，支持向量化边界设置

# 多目标基准测试问题
problem3 = ZDT1(nInput=5)   # 5维 ZDT1问题
problem4 = DTLZ1(nInput=15) # 15维 DTLZ1问题

# 这些基准问题适用于单目标和多目标优化算法的验证与对比。
```

### 敏感性分析

此处以 Ishigami 函数为例，演示敏感性分析模块方法的使用流程。

<p align="center"><img src="./docs/pic/Problem2.svg" width=400 /></p>

Ishigami 函数的理论敏感性指数如下：
一阶敏感性：x1 = 0.314，x2 = 0.442，x3 = 0.000
总敏感性：  x1 = 0.558，x2 = 0.442，x3 = 0.244

<a href="https://nbviewer.org/github/smasky/UQPyL/blob/dev/notebooks/sensitivity_analysis.ipynb" target="_blank">📘 在线查看 Jupyter Notebook 示例 </a>


```python
import numpy as np
from UQPyL.problems import Problem

# 定义 Ishigami 函数
def objFunc(X):
    objs = np.sin(X[:, 0]) + 7 * np.sin(X[:, 1])**2 + \
                 0.1 * X[:, 2]**4 * np.sin(X[:, 0])
    return objs[:, None]

# 构建 Problem 实例
Ishigami = Problem(nInput = 3, nOutput = 1, objFunc = objFunc,
                    ub = np.pi, lb = -1*np.pi, varType = [0, 0, 0],
                    name = "Ishigami")
                    
from UQPyL.sensibility import Sobol

# 初始化 Sobol 方法
sobol = Sobol()

# 定义基础样本数量 N
# 注意：由于 Sobol 方法需要构造多个样本组合，实际调用目标函数的次数将远大于 N。
X = sobol.sample(problem = Ishigami, N = 512)
# 所有敏感性分析方法均提供统一的 .sample(problem) 方法接口，用于对参数空间采样。

# 计算采样点对应的目标值
Obj = problem.objFunc(X)

# 执行敏感性分析
# 参数说明：
#   - problem: 问题实例
#   - X: 采样样本
#   - Obj: 目标函数输出
sobol.analyze(problem, X, Obj)
# 所有敏感性分析方法均提供统一的 .analyze(problem, X, Obj) 方法接口，用于执行敏感性分析。

# 输出：
# =======================Attribute=======================
# First Order Sensitivity: True
# Second Order Sensitivity: False
# Total Order Sensitivity: True
# ======================Conclusion=============================
# --------------------------S1---------------------------------
# +-------------------+-------------------+-------------------+
# |        x_1        |        x_2        |        x_3        |
# +-------------------+-------------------+-------------------+
# |       0.3222      |       0.4531      |       0.0175      |
# +-------------------+-------------------+-------------------+
# --------------------------ST---------------------------------
# +-------------------+-------------------+-------------------+
# |        x_1        |        x_2        |        x_3        |
# +-------------------+-------------------+-------------------+
# |       0.5436      |       0.4306      |       0.2416      |
# +-------------------+-------------------+-------------------+

```

### 参数优化

本示例使用 SCE-UA 算法对 Sphere 问题进行优化。

<a href="https://nbviewer.org/github/smasky/UQPyL/blob/dev/notebooks/optimization.ipynb" target="_blank">📘 在线查看 Jupyter Notebook 示例 </a>

```python

# 导入 Sphere 测试问题
from UQPyL.problems.single_objective import Sphere

# 实例化 Sphere 问题，设定维数为 10，其他参数使用默认值
sphere = Sphere(nInput = 10) #其余设置采用默认

# 导入 SCE-UA 优化算法
from UQPyL.optimization.single_objective import SCE_UA

# 实例化 SCE-UA 算法，使用默认配置
sce = SCE_UA()

# 执行优化
# 输入为优化问题实例，输出为 Result 类对象
res = sce.run(sphere)

# 提取最优解及其对应的目标函数值
bestDecs = res.bestDecs   # 最优决策变量
bestObjs = res.bestObjs   # 最优目标值

# 优化过程将自动在终端打印历史信息与最终结果
# 输出如下: 
# =========Conclusion================================= 
# Time:  0.0 day | 0.0 hour | 0.0 minute |  5.32 second
# Used FEs:    24356  |  Iters:  1000
# Best Objs and Best Decision with the FEs
# +-----------------+-----------------+-----------------+-----------------+-----------------+
# |       FEs       |      Iters      |     OptType     |     Feasible    |       y_1       |
# +-----------------+-----------------+-----------------+-----------------+-----------------+
# |      24122      |       990       |       min       |       True      |     4.6e-12     |
# +-----------------+-----------------+-----------------+-----------------+-----------------+
# +-----------------+-----------------+-----------------+-----------------+-----------------+
# |       x_1       |       x_2       |       x_3       |       x_4       |       x_5       |
# +-----------------+-----------------+-----------------+-----------------+-----------------+
# |      -0.000     |      -0.000     |      0.000      |      0.000      |      -0.000     |
# +-----------------+-----------------+-----------------+-----------------+-----------------+
# +-----------------+-----------------+-----------------+-----------------+-----------------+
# |       x_6       |       x_7       |       x_8       |       x_9       |       x_10      |
# +-----------------+-----------------+-----------------+-----------------+-----------------+
# |      -0.000     |      -0.000     |      0.000      |      -0.000     |      -0.000     |
# +-----------------+-----------------+-----------------+-----------------+-----------------+
```

### 替代模型

以使用 RBF 模型拟合并预测 Sphere 问题的目标函数为例

<a href="https://nbviewer.org/github/smasky/UQPyL/blob/dev/notebooks/surrogate_modelling.ipynb" target="_blank">📘 在线查看 Jupyter Notebook 示例 </a>

```python
from UQPyL.problems import Sphere

# 实例化 Sphere 问题（维数为 10）
sphere = Sphere(nInput = 10)

# 从DoE模块导入超立方拉丁采样（LHS）方法，用于生成训练集和测试集
from UQPyL.DoE import LHS

# 使用 LHS 方法生成 200 个训练样本
lhs = LHS()
xTrain = lhs.sample(200, problem.nInput)

# 计算训练样本的目标函数值
yTrain = sphere.objFunc(xTrain)

# 使用相同方法生成 50 个测试样本
xTest = lhs.sample(50, problem.nInput)

# 计算测试样本的真实目标值
yTest = sphere.objFunc(xTest)

# 从surrogate模块导入 RBF 替代模型
from UQPyL.surrogate.rbf import RBF

# 实例化 RBF 模型（默认参数）
rbf = RBF()

# 用训练数据拟合 RBF 模型
rbf.fit(xTrain, yTrain)

# 对测试样本进行预测
yPred = rbf.predict(xTest)

# 导入 R² 评估指标
from UQPyL.utility.metric import r_square
# 计算预测结果的 R² 分数，衡量模型拟合效果

r2 = r_square(yTest, yPred)
# 输出 R² 分数
print(r2)
```

💡 **提示:** 更多高级功能与示例即将上线——请查看我们的[官方文档](https://uqpyl.readthedocs.io/en/latest/)(正在更新中，感谢您的耐心等待！)

---

## 🔥 欢迎合作

欢迎大家参与贡献，共同扩展UQPyL，加入更多先进的敏感性方法、优化算法以及实际工程问题的示例。

## 📧 联系方式

如有任何问题，请联系：

**wmtSky**  
Email: [wmtsmasky@gmail.com](mailto:wmtsmasky@gmail.com)(优先), [wmtsky@hhu.edu.cn](mailto:wmtsky@hhu.edu.cn)

---


**本项目遵循 MIT 许可协议 - 具体内容详见 [LICENSE](https://github.com/smasky/UQPyL/LICENSE)**

















