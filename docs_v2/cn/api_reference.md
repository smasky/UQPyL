# API Reference

这是 UQPyL 的 API 入口页。

当你知道自己要查某个对象，但不确定它在哪个模块页面里时，从这里进入。教程和工作流优先看 [Quick Start](quick_start.md) 或 [Examples](examples.md)。

## 模块页面

| 模块 | API 页面 | 主要用途 |
|---|---|---|
| `UQPyL.problem` | [Problem API](api/problem.md) | 建模协议、输入空间、评估结果、装饰器和 benchmark problems。 |
| `UQPyL.doe` | [Design of Experiment API](api/doe.md) | 采样类、设计元数据、分析专用设计。 |
| `UQPyL.analysis` | [Analysis API](api/analysis.md) | 敏感性分析方法、指标、结果对象、保存结果 reader。 |
| `UQPyL.optimization` | [Optimization API](api/optimization.md) | 单目标、多目标和昂贵优化 API。 |
| `UQPyL.inference` | [Inference API](api/inference.md) | MCMC 风格推断方法、链结果和保存结果 reader。 |
| `UQPyL.calibration` | [Calibration API](api/calibration.md) | 校准方法、指标处理、结果对象和保存结果 reader。 |
| `UQPyL.surrogate` | [Surrogate API](api/surrogate.md) | 代理模型、缩放、切分、指标、uncertainty 输出和调参。 |

## 常见调用协议

| 工作流 | 调用形式 | 返回 |
|---|---|---|
| 问题评估 | `problem.evaluate(X)` | `Eval` |
| 采样 | `sampler.sample(problem, nSamples=..., seed=...)` | `np.ndarray` |
| 带元数据采样 | `sampler.sampleWithMeta(problem, ...)` | `(X, meta)` |
| 分析 | `method.analyze(problem, X, Y=..., meta=...)` | `AnaResult` |
| 优化 | `algorithm.run(problem, seed=...)` | `OptResult` |
| 推断 | `method.run(problem, gamma=..., seed=...)` | `InfResult` |
| 校准 | `method.run(modelProblem, ...)` | `CalResult` |
| 代理模型训练 | `model.fit(X, Y)` | fitted model |
| 代理模型预测 | `model.predict(Xnew)` | `np.ndarray` 或 `(mean, std/var)` |

## 核心数据对象

| 对象 | 位置 | 作用 |
|---|---|---|
| `Problem` | [Problem API](api/problem.md) | 定义目标、约束、边界、标签和变量类型。 |
| `ModelProblem` | [Problem API](api/problem.md) | 定义带观测、mask 和仿真上下文的仿真模型。 |
| `Eval` | [Problem API](api/problem.md) | `evaluate()` 的标准返回对象。 |
| `AnaResult` | [Analysis API](api/analysis.md) | 敏感性分析结果。 |
| `OptResult` | [Optimization API](api/optimization.md) | 优化结果。 |
| `InfResult` | [Inference API](api/inference.md) | 推断链结果。 |
| `CalResult` | [Calibration API](api/calibration.md) | 校准结果。 |

`Problem.evaluate()` 返回 `Eval`，不是字典：

```text
res = problem.evaluate(X)
objs = res.objs
cons = res.cons
sims = res.sims
```

## 保存结果 Reader

运行方法在 `saveFlag=True` 时可以保存 sqlite 结果。

| 模块 | Reader | 常用导入 |
|---|---|---|
| Analysis | `AnaReader` | `from UQPyL.analysis.runtime import AnaReader` |
| Optimization | `OptReader` | `from UQPyL.optimization.runtime import OptReader` |
| Inference | `InfReader` | `from UQPyL.inference import InfReader` |
| Calibration | `CalReader` | `from UQPyL.calibration import CalReader` |

常见 reader 模式：

```text
with Reader("Result/example.sqlite3") as reader:
    summary = reader.get_run_summary()
    result = reader.load_result()
```

具体 reader 方法见各模块 API 页面。

## 运行控制项

分析、优化、推断和校准方法通常接受：

| 选项 | 含义 |
|---|---|
| `verboseFlag` | 打印运行进度或最终摘要。 |
| `verboseFreq` | 支持时控制进度打印间隔。 |
| `logFlag` | 支持时写文本日志。 |
| `saveFlag` | 支持时保存 sqlite 结构化结果。 |
| `saveFreq` | 支持时控制快照保存间隔。 |

示例和学习阶段建议使用 `verboseFlag=False`、`logFlag=False`、`saveFlag=False`。

## 导入速查

| 需求 | 导入 |
|---|---|
| 定义普通问题 | `from UQPyL.problem import Problem` |
| 定义仿真问题 | `from UQPyL.problem import ModelProblem` |
| 生成 LHS 样本 | `from UQPyL.doe import LHS` |
| 运行自由设计敏感性分析 | `from UQPyL.analysis import RBDFAST` |
| 运行单目标优化 | `from UQPyL.optimization.soea import GA` |
| 运行多目标优化 | `from UQPyL.optimization.moea import NSGAII` |
| 运行 MCMC 推断 | `from UQPyL.inference import MH` |
| 运行 GLUE 校准 | `from UQPyL.calibration import GLUE` |
| 训练 RBF 代理模型 | `from UQPyL.surrogate.rbf import RBF` |

## 命名说明

| 名称 | 说明 |
|---|---|
| `objs` | 目标矩阵，通常 shape 为 `(n_samples, n_obj)`。 |
| `cons` | 约束矩阵，值 `<= 0` 表示可行。 |
| `decs` | 决策/输入矩阵，或推断结果中的链样本。 |
| `sims` | `ModelProblem` 的仿真输出，通常 shape 为 `(n_samples, n_time, n_series)`。 |
| `bestDecs` | 最优决策行，或多目标方法中的 Pareto 决策矩阵。 |
| `bestObjs` | `bestDecs` 对应的目标值。 |

## 运行失败与保存结果

优化、分析、推断和校准的公开 `run` / `analyze` 入口会管理本次运行的数据库会话。初始化、计算或结果保存发生异常时，原始异常继续抛出；未提交的写入会回滚，已提交的数据保留。开启 `saveFlag` 且运行记录已创建时，数据库可写则将状态更新为 `failed` 并记录结束时间，随后关闭连接。正常返回的运行保持 `finished` 状态。

通过对应 Reader 的 `list_runs()` 或 `get_run_summary()["status"]` 查看运行状态；失败记录可能只有元数据或部分已提交快照，不保证存在完整结果对象。失败运行不会自动重试或返回部分成功结果。正常清理后，同一个方法实例可以重新运行，新运行使用新的会话。

如果数据库自身无法回滚或更新状态，异常会附带收尾错误说明，仍继续尝试关闭连接；不能保证这时数据库中的状态已改为 `failed`。若关闭本身也失败，保留 `method.session` 供排查和清理，禁止新运行覆盖该会话。这些保障适用于 Python 能够处理的异常（包括 `KeyboardInterrupt`），不涵盖强制终止进程或断电。直接手动组合 `setup` / `finalize` 时，会话生命周期仍由调用方负责。
