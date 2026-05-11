# UQPyL 文档

UQPyL 提供统一的问题建模协议，以及一组用于采样、敏感性分析、优化、推断、校准和代理建模的不确定性量化工作流。

大多数工作流都从 `Problem` 或 `ModelProblem` 开始，然后将该对象传给一个或多个功能模块。

```text
Problem / ModelProblem -> Method -> Result
```

## 从这里开始

| 目标 | 阅读 |
|---|---|
| 跑通最短完整工作流 | [Quick Start](quick_start.md) |
| 理解统一的建模协议 | [Problem](problem.md) |
| 查看完整工作流示例 | [Examples](examples.md) |
| 查类、参数和结果对象 | [API Reference](api_reference.md) |

## 选择工作流

| 任务 | 模块指南 | API |
|---|---|---|
| 定义输入空间、目标、约束、仿真和评估输出 | [Problem](problem.md) | [Problem API](api/problem.md) |
| 为实验、分析、初始化或建模生成设计样本 | [Design of Experiment](doe.md) | [DOE API](api/doe.md) |
| 分析输入如何影响模型或目标输出 | [Analysis](analysis.md) | [Analysis API](api/analysis.md) |
| 搜索单目标、多目标或昂贵模型的最优解 | [Optimization](optimization.md) | [Optimization API](api/optimization.md) |
| 运行 MCMC 风格的参数推断 | [Inference](inference.md) | [Inference API](api/inference.md) |
| 让仿真模型拟合观测数据 | [Calibration](calibration.md) | [Calibration API](api/calibration.md) |
| 训练预测型代理模型 | [Surrogate Modeling](surrogate.md) | [Surrogate API](api/surrogate.md) |

## 推荐阅读路径

1. 先看 [Quick Start](quick_start.md)，理解整个工作流的基本形状。
2. 在使用任何功能模块前，先看 [Problem](problem.md)。
3. 再从上面的工作流表里选择一个你需要的模块页。
4. 当你需要查构造参数、返回字段或 reader 类时，再看 [API Reference](api_reference.md)。
5. 如果你想复制完整模式，继续看 [Examples](examples.md)。

## 核心概念

| 概念 | 在哪里看 |
|---|---|
| `Problem` | 静态目标与约束问题。见 [Problem](problem.md)。 |
| `ModelProblem` | 带观测、掩码和仿真上下文的仿真模型。见 [Problem](problem.md)。 |
| `Eval` | `problem.evaluate()` 的标准输出对象。见 [Problem API](api/problem.md)。 |
| 结果对象 | 如 `AnaResult`、`OptResult`、`InfResult`、`CalResult` 等模块结果对象。见 [API Reference](api_reference.md)。 |
| 保存结果 | 如 `AnaReader`、`OptReader`、`InfReader`、`CalReader` 等 sqlite reader。见各模块 API 页面。 |

## API 参考

API 参考按模块拆分。

| 模块 | API 页面 |
|---|---|
| `UQPyL.problem` | [Problem API](api/problem.md) |
| `UQPyL.doe` | [DOE API](api/doe.md) |
| `UQPyL.analysis` | [Analysis API](api/analysis.md) |
| `UQPyL.optimization` | [Optimization API](api/optimization.md) |
| `UQPyL.inference` | [Inference API](api/inference.md) |
| `UQPyL.calibration` | [Calibration API](api/calibration.md) |
| `UQPyL.surrogate` | [Surrogate API](api/surrogate.md) |

## 项目说明

| 页面 | 用途 |
|---|---|
| [Changelog](changelog.md) | 面向用户的版本变更记录。 |
