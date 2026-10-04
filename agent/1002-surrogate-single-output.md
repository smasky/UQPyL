# 2026-10-02 单模型单输出与 MultiSurrogate 统一约定

用户明确选择：**每个替代模型只拟合一个输出，多输出统一由 MultiSurrogate 控制。** 当前代码已按这个约定实现；本记录取代 10 月 1 日 OM03 修复中“LR/PR Origin/Ridge 直接支持多输出”的接口决定。此前形状错误与离散映射修复仍保留追溯，不再以原生多输出作为单模型的验收标准。

最终 conda `py312` 全量 **2490 passed，58.58 秒，`-W error` 零未捕获警告**。相比前一阶段增加 89 项，原有数值断言迁移到容器或单输出逐列验证，没有删减测试；独立数值复核 **169 组零差异**。

## 当前接口

| 层级 | 训练目标 | 预测结果 |
|---|---|---|
| 单个模型 | 原始 `(nTrain,)` 或 `(nTrain,1)`；预处理后为一列矩阵 | 均值及支持的不确定性为 `(nPred,1)` |
| MultiSurrogate | `(nTrain,m)`，m 个不同的模型实例 | 均值/不确定性 `(nPred,m)`；导数 `(nPred,nVariables,m)` |

这个约定覆盖 RBF、GPR、KRG、MARS、SVR、LinearRegression 和 PolynomialRegression；三种回归损失 Origin/Ridge/Lasso 一致。Scaler、特征展开和独立评分函数仍可处理多列数组，输入特征数不受单输出限制。
Problem 的多个目标、敏感性分析的多个输出等上层接口保持；单输出约定针对每个具体替代模型。

## 实现边界

- [SurrogateABC](../UQPyL/surrogate/base.py) 共用单输出检查：原始数据在拟合 Scaler 前校验，预处理入口通过 `storeTrainingData` 校验；所有内置模型的 `fitModel` 都经过该入口。
- [GPR](../UQPyL/surrogate/gp/gaussian_process.py) / [KRG](../UQPyL/surrogate/kriging/kriging.py) 的 `fitHyper` 在优化前清理旧拟合状态并验证目标列数，防止直接调用绕过预处理检查。
- [AutoTuner](../UQPyL/surrogate/auto_tuner.py) 在划分和候选搜索前检查单输出；调参仍逐个子模型进行，未新增跨输出联合调参。
- [MARS](../UQPyL/surrogate/mars/mars.py) 的数据清洗也遵循单输出，移除多输出权重扩展分支，单模型导数末轴固定为 1。底层矩阵求解及原生扩展无需重建。
- 回归模型共享上述约定；Lasso 原生求解前保留同一检查，不再单独定义另一套输出规则。

多列 Y 在上述公开训练入口抛出明确的 ValueError，提示使用 MultiSurrogate。不会静默选第一列或把输出合并成一个值。原始 `fit`/`prepareTrainingData` 接收向量；直接 `fitModel`/`fitHyper` 使用标准预处理后的一列矩阵。

## MultiSurrogate

容器保持逐输出独立模型，按 `(nTrain,1)` 拆列训练并分配独立随机流，`fit` 成功返回自身。整体矩阵维度、样本数、输出列数和模型数在调用子模型前校验。失败/中断使全部子模型失效，避免将新旧拟合状态混用；重新成功拟合可恢复。

`predict` 继续返回按列拼接的均值，新增与单模型一致的 `returnStd`/`returnVar`；所有子模型须支持不确定性，两个标志不能同时启用。`supportsUncertainty` 根据子模型能力计算。每列使用各自的输出 Scaler，不将单个方差广播成所有输出，也不提供跨输出协方差。

`predict_deriv` 汇总支持导数的子模型，保留变量选择、原始单位和可选 missing mask。模型能力、预测样本数及单列形状、导数样本/变量/输出轴都有检查；不支持的能力明确拒绝。

过去 GPR/KRG 原生多输出会共享超参数，MARS 会共享基函数；统一后各输出独立拟合/调参，结果和耗时可能相应变化。这是用户选择的模型职责划分，不保留原生多输出兼容入口。

## 验证与迁移

新增 [单输出约定测试](../tests/test_surrogate_single_output_contract.py) **89 项**。首批 86 项在修改前 **69 failed / 17 passed**，实现后 **86 passed，0.86 秒**；随后补充两项容器失败/恢复和一项导数能力检查。

覆盖七类模型与回归损失的 11 种配置、四个训练入口、预处理/搜索之前的拒绝时机、向量/单列一致性、失败重拟合失效、两种调参模式/入口、容器整体数据校验、逐输出独立高斯矩阵均值/方差参照、能力与预测轴检查及失败恢复。新增测试仅把无效多列输入交给调用哨兵或提前检查，不依赖错误原生路径的行为作为验收结果。

既有数学验证保留并调整到新约定：

- [回归 38 项](../tests/test_surrogate_regression_multioutput.py)：独立最小二乘/岭矩阵、三种缩放配置、截距、常数输出、空/单/多点、预处理入口、重拟合与解析 Lasso 软阈值。
- [GPR/KRG 不确定性 10 项](../tests/test_surrogate_scaler_regressions.py)：容器与独立固定单输出模型对照，维持输出尺度/方差换算断言。
- [MARS 导数 9 项](../tests/test_surrogate_mars_derivative_contracts.py) 迁到容器，解析梯度及有限差分仍逐值核对；[评分 3 项](../tests/test_surrogate_public_fit_contracts.py) 逐列核对原单位加权公式。
- [GPR 独立似然/后验 12 项](../tests/test_gpr_likelihood_crosschecks.py) 和 [joint density 1 项](../tests/test_gpr_likelihood_direction.py)：容器各输出负对数似然之和仍与独立密度/行列式参照一致，三种核、一/三维、缩放、均值/方差/标准差的断言保留。
- [RBF 平滑/缩放 10 项](../tests/test_rbf_smoothing.py)：五种核、两种平滑强度，容器预测仍与 SciPy RBFInterpolator 比较。

首次全量发现最后一批 23 个原生多输出用例尚未迁移，**23 failed / 2467 passed**；这些用例迁移后相关三个文件 **133 passed，1.22 秒**。不是跳过或删除失败用例，也没有放宽数值容差。

| 验证 | 结果 | 日志 |
|---|---|---|
| 首批契约修复前 | 69 failed / 17 passed，1.50 秒 | [before](verification/1002-surrogate-single-output-before.txt) |
| 首批契约修复后 | 86 passed，0.86 秒 | [after](verification/1002-surrogate-single-output-after.txt) |
| 代理/调参/可视化相关 | 751 passed，4.04 秒 | [targeted](verification/1002-surrogate-single-output-targeted.txt) |
| 首次全量（发现待迁移用例） | 23 failed / 2467 passed，76.92 秒 | [initial full](verification/1002-surrogate-single-output-full-initial.txt) |
| 补迁三个文件 | 133 passed，1.22 秒 | [remaining](verification/1002-surrogate-single-output-remaining.txt) |
| 最终全量 | 2490 passed，58.58 秒 | [full](verification/1002-surrogate-single-output-full.txt) |

环境为 conda `py312`，pytest 均使用 `-W error`，数值命令限制 BLAS/OMP 为一个线程。相关 Ruff 静态、格式和差异检查通过。

独立 [其他模块审查脚本](verification/review_other_modules.py) 也按新约定调整：OM03 同时验证单模型拒绝多输出及容器的独立矩阵预测；GPR 的两输出均值/方差与各单输出负对数似然之和仍与独立矩阵计算比较。[新 JSON](verification/1002-other-modules-single-output.json) / [摘要](verification/1002-other-modules-single-output.txt) 保存 **169 组、零差异、106 组正常控制**，10 月 1 日原始及修复后证据不覆盖。

中英文 API、代理模型指南、测试导航及交接记录同步。OM01/OM02/OM04 的完成状态保持；极小尺度 RMSE、SA09/SA11、MARS 高阶贡献局限及 C15/C17/C20 暂缓项保持。未提交/推送/发布，未重建 Python 3.14 wheel；验证结果不能解读为所有模型在所有问题上已获数学正确性或统计收敛证明。
