# UQPyL: 参数不确定性分析及优化工具包

<p align="center"><img src="./docs/UQ.svg" width="400"/></p>

[![PyPI version](https://badge.fury.io/py/UQPyL.svg?icon=si%3Apython&icon_color=%2331aadd)](https://badge.fury.io/py/UQPyL) ![PyPI - Downloads](https://img.shields.io/pypi/dm/UQPyL) ![PyPI - License](https://img.shields.io/pypi/l/UQPyL) ![GitHub last commit](https://img.shields.io/github/last-commit/smasky/UQPyL) ![Static Badge](https://img.shields.io/badge/Author-wmtSky-orange) ![Static Badge](https://img.shields.io/badge/Contact-wmtsmasky%40gmail.com-blue)

UQPyL 是一个功能全面的 Python 工具包，专注于参数不确定性分析与优化，适用于数值模型校准、资源调度、产品设计等工程优化问题。该工具包同时集成了多种常用方法，包括实验设计 (Design of Experiments)、敏感性分析 (Sensitivity Analysis) 以及参数优化 (支持单目标与多目标) 。此外，内置的替代模型 (Surrogate Models) 模块可用于计算代价昂贵问题 (Computational Expensive Problem) 的求解。

👉[English Doc](./README.md)

## Contents

- [功能特点](#-功能特点)
- [安装指南](#-安装指南)
- [实用链接](#-实用链接)
- [方法与基准问题预览](#-方法与基准问题预览)
   - [Sensitivity Analysis](#sensitivity-analysis)
   - [Optimization Algorithms](#optimization-algorithms)
   - [Surrogate Models](#surrogate-models)
   - [Single-objective Problems](#single-objective-problems)
   - [Multi-objective Problems](#multi-objective-problems)
- [Quick Start](#-quick-start)
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

## 🎉 方法与基准问题预览

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













