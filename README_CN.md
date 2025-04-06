# UQPyL: 参数不确定性分析及优化工具包

<p align="center"><img src="./docs/UQ.svg" width="400"/></p>

[![PyPI version](https://badge.fury.io/py/UQPyL.svg?icon=si%3Apython&icon_color=%2331aadd)](https://badge.fury.io/py/UQPyL) ![PyPI - Downloads](https://img.shields.io/pypi/dm/UQPyL) ![PyPI - License](https://img.shields.io/pypi/l/UQPyL) ![GitHub last commit](https://img.shields.io/github/last-commit/smasky/UQPyL) ![Static Badge](https://img.shields.io/badge/Author-wmtSky-orange) ![Static Badge](https://img.shields.io/badge/Contact-wmtsmasky%40gmail.com-blue)

UQPyL 是一个功能全面的 Python 工具包，专注于参数不确定性分析与优化，适用于数值模型校准、资源调度、产品设计等工程优化问题。该工具包同时集成了多种常用方法，包括实验设计 (Design of Experiments)、敏感性分析 (Sensitivity Analysis) 以及参数优化 (支持单目标与多目标) 。此外，内置的替代模型 (Surrogate Models) 模块可用于计算代价昂贵问题 (Computational Expensive Problem) 的求解。

👉[English Doc](./README.md)

## Contents

- [功能特点](#-功能特点)
- [安装指南](#-安装指南)
- [实用链接](#-实用链接)
- [方法预览](#-方法预览)
   - [敏感性分析方法](#敏感性分析方法)
   - [优化算法](#优化算法)
   - [替代模型](#替代模型)
   - [单目标优化基准问题](#单目标优化基准问题)
   - [多目标优化基准问题](#多目标优化基准问题)
- [快速开始](#快速开始)
- [Call for Contributions](#-call-for-contributions)
- [Contact](#-contact)

## ✨ 功能特点
1. **全面支持敏感性分析与优化**: 实现了当前广泛使用的敏感性分析方法和优化算法。
2. **运行显示与结果保存**: 允许用户跟踪并保存运行历史和结果。
3. **先进的替代模型**: 集成了多种替代模型及自动调优工具，以提升模型性能。
4. **丰富的应用资源**: 提供了全面的基准问题和实际案例，帮助用户快速上手。(👉 近期规划： 针对水科学研究，我们计划定制特定模型专用的程序接口，将水利相关模型与 UQPyL 集成，提升可用性和功能性，类似于我们已开发的[SWAT-UQ](https://github.com/smasky/SWAT-UQ)。如果您感兴趣，欢迎联系我们进行合作。)
5. **模块化与可扩展的架构**: 设计了统一的敏感性分析与优化架构，支持用户快速开发新方法或算法(我们非常欢迎并感谢您对UQPyL的贡献)。

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
- **引用信息**: [UQPyL 2.0](**TODO**: 需要更新), [UQPyL 1.0](https://www.sciencedirect.com/science/article/pii/S1364815215300955)

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


💡 **提示:** 以上方法现在均支持使用替代模型。

🚀 **致谢:** UQPyL敏感性分析模块的部分想法参考[SALib](https://github.com/SALib/SALib)。

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

(标签 `Surrogate` 表示该算法可用于解决计算代价昂贵wenti)

💡 **提示:** 该模块正在持续更新先进算法中，如果您有需要其它算法，请联系我们。

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

❤️ 在这里，我们还提供了替代模型的**自动校准工具**，因此不再需要担心选取替代模型的超参数，保证模型的最优构建。

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

| 名称 | 目标数量 | 帕累托前沿形状 | 特性 |
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

**TODO:** 我们计划将一些常见的水文模型校准方法（如 SWAT、SAC 等）或相关的水资源优化案例纳入 UQPyL。

---

## 🍭 快速开始

为了高效使用UQPyL, 首先是描述要解决的问题，包括以下两个方面：

1. **给出决策变量的信息**，例如决策变量的维度，每个变量取值范围以及变量类型 (支持float, int 以及 discrete)。
2. **定义目标函数**，即说明如何根据决策变量`x`获得输出目标`obj`。在UQPyL中，目标函数被命名为`objFunc`。这个函数可以是解析函数、计算模型，或外部黑盒过程。如有约束条件，也需要定义相应的约束函数，命名为 conFunc。

以下问题是一个Rosenbrock函数的变体，它在原始函数的基础上增加了一个约束条件($x_1^2+x_2^2+x_3^2 \ge 4$)，并更改了变量的类型：原本都是连续型，即float类型，现在将$x_2$设置为整数(int)，$x_3$设置为离散变量(discrete)。

<p align="center"><img src="./docs/pic/Problem1.svg" width=500/></p>

UQPyL提供了一个名为`Problem`的Python类，用于简化问题定义的工作流程。

```Python
# 第1步: 从UQPyL的problems模块导入Problem类
from UQPyL.problems import Problem

# 第2步: 定义objFunc函数
# objFunc是一个接收numpy的二维矩阵X并返回numpy的二维矩阵objs的函数, 其中:
# 矩阵X的每一行代表一组决策变量，每一列对应同一变量的不同取值
# 返回的矩阵objs的行数应与X相同，列数则等于该问题的目标数
# 具体来说:
# 对于单目标问题，二维矩阵objs的形状应为 (N, 1)
# 对于多目标问题，二维矩阵objs的形状应为 (N, M)
# 其中，N 表示输入的决策变量组合数，M 表示目标函数的个数。

def objFunc(X):
    
    # 如果条件允许，建议对矩阵 X 进行向量化操作，以提升计算效率
    objs =100 * (X[:, 2] - X[:, 1]**2)**2+ 100 * (X[:, 1] - X[:, 0]**2)**2  + \
            (1 - X[:, 1])**2 + (1 - X[:, 0])**2 

    return objs[:, None] #需要确保返回的矩阵objs是二维的，即使UQPyL后续会帮你进行检查

# UQPyL还提供另外一种定义objFunc函数的方式。
# 对于涉及数值计算模型的问题，通常不能对矩阵X进行向量化操作
# UQPyL提供一种装饰器函数`@singleFunc`，来启用`单例运行模式`
# objFunc函数将只接收numpy的一维array或者python的list格式的变量
# 因此，该函数一次只能处理一组决策，这在每次评估计算开销较大或模型设计为一次处理一个解的情况下特别有用

# 首先，导入开启单例模式的装饰器
from UQPyL.problems import singleFunc

@singleFunc
def objFunc_(X): # 变量X应为numpy的一维array或者python的list格式
    #对变量X进行逐元素操作
    obj = 100 * (X[2] - X[1]**2)**2 + 100 * (X[1] - X[0]**2)**2 + \
            (1 - X[1])**2 + (1 - X[0])**2 
    return obj #此处应返回数值、一维array、list形式的函数值obj

# 第3步: 定义conFunc函数
# 与 objFunc 函数类似，约束函数 concFunc 也有两种定义方式可选。
# 需要注意的是，concFunc 的返回值表示约束的违反程度：
# - 返回值小于 0 表示约束被违反，且值越小，违反程度越严重；
# - 返回值大于 0 表示满足约束，即为正常可行解。
# 因此，用户有时需要对问题的实际约束函数进行修改或重新建模，以满足上述约定。

# 矩阵模式
def conFunc(X):
    cons = X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2 - 4 
    return cons[:, None]

# 单例模式
@singleFunc
def conFunc(X):
    con = X[0]**2 + X[1]**2 + X[2]**2 - 4 
    return con

# 第4步: 给出问题的其它信息

nInput = 3 #决策变量的输入维数，这里，它等于3
nOutput = 1 #输出目标的个数，这里，它等于1

# 决策变量的取值上界
ub = [10, 10, 10] # ub 可以是python的float，int，list格式或者是numpy格式
# 在本例，所有变量的上界均为10，因此也可使用 `ub = 10`

# 决策变量的取值下界
lb = [0, 0, 0]
# 在本例，所有变量的下界均为0，因此也可使用`lb = 0`

# 定义变量的类型
# 0 表示 连续型即float，1表示整数(int)型，2表示离散(discrete)型
varType = [0, 1, 2] #不给定的情况下，默认所有变量均为连续型

# 指定变量类型为离散型之后，需要指定该变量的可行解
varSet = {2: [2, 3.4, 5.1, 7]} 
# varSet 是一个字典，其中键表示变量的索引（2 表示第三个变量x3）。它遵循 Python 的零基索引规则。
# 与键2相关联的值指定了x3的可能取值集合 [2, 3.4, 5.1, 7]。
# 这意味着x3只能取这四个值之一：2、3.4、5.1 或 7。

# 指定优化类型， 'min' 表示最小化， 'max' 表示极大化
optType = 'min'

# 如果决策变量具有对应的名称，可以为其指定名称。
xLabels = ['x1', 'x2', 'x3'] 
# 要不然, UQPyL为其指定默认名字'x1', 'x2', 'x3', 等.

# 如果目标值有名称，也可为其指定
yLabels = ['obj1']
# 要不然，UQPyL为其指定默认名字'obj1','obj2'等

# 问题名称
name = 'Rosenbrock'
# 可用于标识问题实例、整理结果、保存文件等用途。

# 第5步: 实例化当前问题

problem = Problem(nInput = nInput, nOutput = nOutput, objFunc = objFunc, concFunc = concFunc,
                    ub = ub, lb = lb, varType = varType, varSet = varSet,
                        xLabels = xLabels, yLabels = yLabels, name = name)
# nInput, nOutput, objFunc, conFunc, ub, lb, varType, varSet, xLabels, yLabels, name等均为Problem类的参数名称

# 第6步: 使用优化算法求解
# UQPyL中的所有方法或算法都可以读取'problem'类来获取足够问题信息
# 这里，我们使用遗传算法作为例子

from UQPyL.optimization.single_objective import GA

# 创建遗传算法的实例

ga = GA() # 对于GA类存在可选参数，此处采用默认参数

# 导入problem来运行遗传算法

ga.run(problem = problem)

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















