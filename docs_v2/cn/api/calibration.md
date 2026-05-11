# Calibration API

## `UQPyL.calibration`

`calibration` 模块通过比较仿真和观测来估计模型参数。所有校准方法都使用 `ModelProblem`。

## 导入

```python
from UQPyL.calibration import GLUE, SUFI2, ES, IES, CalReader
```

## 公共对象

| 对象 | 作用 |
|---|---|
| `GLUE` | Generalized Likelihood Uncertainty Estimation。 |
| `SUFI2` | Sequential Uncertainty Fitting。 |
| `ES` | Ensemble Smoother。 |
| `IES` | Iterative Ensemble Smoother。 |
| `CalResult` | 校准结果对象。 |
| `CalHistory` | 校准历史。 |
| `CalReader` | 读取保存的校准 sqlite。 |

## 通用调用

```text
result = method.run(modelProblem, X=None, seed=123)
```

| 参数 | 含义 |
|---|---|
| `modelProblem` | `ModelProblem`。 |
| `X` | 候选参数矩阵；部分方法可内部生成。 |
| `seed` | 随机种子。 |

运行控制参数：

| 参数 | 含义 |
|---|---|
| `verboseFlag` | 打印运行摘要。 |
| `logFlag` | 写日志。 |
| `saveFlag` | 保存 sqlite 结果。 |

## 方法

| 方法 | 关键参数 | 主要输出 |
|---|---|---|
| `GLUE` | `metric`, `threshold` | `behavioralDecs`, `behavioralSims` |
| `SUFI2` | `eliteSize`, `nSamples`, `maxIters` | `eliteDecs`, `updatedLb`, `updatedUb`, `pfactor`, `rfactor` |
| `ES` | ensemble 参数矩阵 `X` | `posteriorDecs`, `posteriorSims` |
| `IES` | `maxIters`, `lam` | `posteriorDecs`, `posteriorSims`, history |

## `CalResult`

| 字段 | 含义 |
|---|---|
| `method` | 校准方法名。 |
| `bestDecs` | 最优参数行。 |
| `bestSim` | 最优参数对应的仿真。 |
| `behavioralDecs`, `behavioralSims` | GLUE 接受的样本。 |
| `eliteDecs`, `eliteSims` | SUFI2 elite samples。 |
| `posteriorDecs`, `posteriorSims` | ES/IES posterior ensemble。 |
| `diagnostics` | 分数、mask、边界、pfactor/rfactor 等方法特定信息。 |
| `history` | 迭代历史。 |
| `summary()` | 摘要字典。 |

## 指标方向

| 指标 | 更好方向 |
|---|---|
| `mse`, `mae`, `rmse`, `pbias` | 越小越好 |
| `nse`, `r2`, `pearson_r`, `kge` | 越大越好 |

GLUE 的阈值判断会依据指标方向执行。

## `CalReader`

| 方法 | 含义 |
|---|---|
| `list_runs(result_dir)` | 列出校准结果。 |
| `get_run_summary()` | 读取摘要。 |
| `get_run_params()` | 读取参数。 |
| `load_problem()` | 读取保存的 `ModelProblem`。 |
| `load_result()` | 重建 `CalResult`。 |

## 下一步

| 目标 | 阅读 |
|---|---|
| 用户指南 | [Calibration](../calibration.md) |
| 仿真问题 | [Problem API](problem.md) |
| 候选参数采样 | [DOE API](doe.md) |
