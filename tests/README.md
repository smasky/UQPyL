# 测试导航

当前完整回归：**3122 项通过**。形状协议修复：[test_model_mask_layouts.py](test_model_mask_layouts.py) 覆盖展平/网格模拟的masked NaN、四校准方法等价和预测维度提示；[文档工作流](test_documented_workflows.py)直接执行中英文掩码示例并核对数值及全屏蔽处理。详见[修复报告](../agent/1004-shape-contract-fixes.md)。

本轮整理入口：[测试职责、数值依据和选跑命令](TEST_MATRIX.md)。测试整理时为 **3094 项**，其中 149 项已标记 `numerical`、8 项标记 `statistical`；默认全部运行。原 3084 个节点全部保留，新增 10 项 DoE 构造验证。详见[整理报告](../agent/1004-test-organization.md)。下方旧迁移表保留其历史数量，不作为当前总量。

公共辅助模块：`optimization_test_support.py` 提供算法工厂和独立 Sphere/ZDT1 结果断言；`analysis_test_support.py` 提供无截止并列情形的稠密 Delta 参照。并列邻居的加权参照独立保留。

收尾回归：[test_closeout_runtime_and_plot_data.py](test_closeout_runtime_and_plot_data.py) 的11项覆盖四领域真实运行/SQLite往返/关闭与复用、创建时间一致性、敏感性原值与标签对齐、非有限值warning、优化公共坐标与负区间、二维/三维参考点和推断变量标题。六类图形的数据核对及导出见 `agent/verification/check_closeout_figures.py`。

Problem转换回归：[test_problem_conversion_regressions.py](test_problem_conversion_regressions.py) 的31项覆盖两类Problem的行列边界/样本轴、边界副本、歧义布局拒绝、单点评价复用缓冲区，以及Decimal参照的极端/微小连续范围转换和混合变量往返。

敏感性极端范围：[test_analysis_extreme_input_and_statistics.py](test_analysis_extreme_input_and_statistics.py) 的13项覆盖Delta输入Decimal参照、四种子分析/穷举/真实GA跨单位一致、离散/固定轴，以及Morris恢复范围warning、字段状态、零值与SQLite往返。

DoE 边界修复：[test_doe_edge_regressions.py](test_doe_edge_regressions.py) 覆盖单样本LHS恢复与有效模式、优化次数预检/正常候选选优、Sobol/Saltelli跳点块对齐及独立序列对照、单次质量warning与FFD元数据隔离。

推断剩余边界：[test_inference_probability_and_covariance.py](test_inference_probability_and_covariance.py) 的39项覆盖无效概率拒绝、合法零概率区域初始化与转移、失败SQLite状态、纯snooker全维刷新/单位/固定维、独立高精度协方差与不重扫历史。长链分布和性能实测见 `agent/verification/check_inference_remaining_fixes.py`。

推断公共结构：[test_inference_common_structure.py](test_inference_common_structure.py) 的28项覆盖五方法公共参数/修改后预检、gamma格式等价与无效输入零模型调用、统一策略诊断、SQLite往返和嵌套副本隔离。176组重构前后逐值对照见 `agent/verification/check_inference_structure_parity.py`，包括轨迹、FEs、回调顺序和后续随机数。

推断转移正确性：[test_inference_transition_correctness.py](test_inference_transition_correctness.py) 核对越界拒绝及真实FEs、DEMC逐链条件更新/对称全维扰动、snooker投影/正反接受比/固定维/对数范围、DREAM预热适应统计、输入单位不变性及独立高斯/截断高斯矩；[档案测试](test_dream_archive_isolation.py) 验证占据状态副本、拒绝后的重复点、预热结束冻结和随机复现。长期分布复测与原始数据放在 `agent/verification/`。

校准误差范围与 ESS：[test_calibration_range_and_ess.py](test_calibration_range_and_ess.py) 使用 Decimal 验证 MSE/MAE 尺度、相减/求和溢出、次正规数及真正超范围 warning，覆盖实际筛选、GLUE 权重不变与告警保存，以及 ESS=20 舍入边界。

校准统一输出：[test_calibration_unified_output.py](test_calibration_unified_output.py) 覆盖四方法行对应、最佳行与原始评分、mask 区间映射、独立先验权重来源、空区间、数组隔离及 SQLite/reader 摘要往返。

校准精度修正：[test_calibration_accuracy_repairs.py](test_calibration_accuracy_repairs.py) 使用解析高斯/截断正态后验、独立 SciPy 随机 MAP、SQLite、低 ESS warning、历史最佳保留、有限差分边界/mask/成本对照。

校准控制补齐：[test_calibration_followup_controls.py](test_calibration_followup_controls.py) 覆盖 SUFI2 参数预检/范围保护、GLUE 独立加权分位数与 SQLite、IES 固定 RML 目标回溯/失败保留、边界步长缩短。

校准后验矩：[test_calibration_posterior_moments.py](test_calibration_posterior_moments.py) 用解析 Kalman 矩、独立随机 MAP 解和阻尼正规方程核对 ES/IES，覆盖重复迭代、奇异/零噪声、单位转换、随机复现、mask 与边界。

校准 RMSE 范围：[test_calibration_rmse_range.py](test_calibration_rmse_range.py) 新增 26 项，以 Decimal 独立参照验证量纲、mask、相减溢出与相消、次正规数及真正超范围告警，并验证 GLUE/SUFI2 的实际筛选不受单位缩放影响。

SUFI2 混合域：[test_sufi2_mixed_domains.py](test_sufi2_mixed_domains.py) 新增 9 项，覆盖多轮连续/整数/离散混合采样、非整数边界、无序且超出编码边界的离散选项、单值收缩、原问题保护，以及外部非法值模拟前拒绝。

SVR 收敛诊断：[test_svr_convergence_status.py](test_svr_convergence_status.py) 新增 5 项，覆盖两类 SVR 达上限后的 Python warning/原生静默/可预测、重新拟合状态更新、两种调参模式的候选与最终诊断，以及 warning 升异常后的失效。

GPR 噪声范围：[test_gpr_noise_accuracy.py](test_gpr_noise_accuracy.py) 新增 21 项，覆盖三档噪声、三个种子、有/无目标标准化的独立预测与方差估计，以及固定 C 的独立矩阵参照和显式范围优先。

替代模型预测精度：[test_surrogate_prediction_accuracy.py](test_surrogate_prediction_accuracy.py) 新增 16 项，
用独立测试点验证三种 GPR 核的短尺度函数精度、MARS 默认二阶交互、SVR 留出调参后的泛化，以及已复现 GPR 失配案例的区间覆盖。
最后一项是固定案例回归，不是普遍不确定性校准保证。390 组独立审查见 `agent/verification/1002-surrogate-accuracy-after-fixes.json`。

替代模型数值边界专项：[test_surrogate_solver_boundaries.py](test_surrogate_solver_boundaries.py) 新增 27 项，
覆盖 Cubic RBF 单位换算与独立自然样条、奇异重拟合 warning/残差诊断、整数/混合 dtype Lasso 原数组保护，以及三个种子的真实 nu-SVR 搜索和非法 nu 的后端前校验。
[test_surrogate_numeric_ranges.py](test_surrogate_numeric_ranges.py) 新增 62 项，覆盖 R²/NSE 手算、混合量纲输出权重、极端目标调参、StandardScaler 样本矩与常数列往返、GPR/KRG/容器标准差尺度一致性、方差范围告警，以及 GPR 非法目标/噪声在三个拟合入口的拒绝时机。
独立后验矩阵与 SciPy 插值对照另见 `agent/verification/review_surrogate_module.py`，修复前 215 组证据保留，修复后输出另存 after-fixes 文件。

RBF 可恢复奇异情况：[test_rbf_singular_warning.py](test_rbf_singular_warning.py) 新增 15 项。
五种核的重复/冲突观测与独立聚合后 SciPy 插值对照，验证约束残差、不同单位下的线性趋势、不可辨识趋势标记和近似求解失败后旧状态失效。warning 显式捕获，未屏蔽正常路径的告警。

完整测试套件要求已构建 MARS、Lasso、SVR 原生扩展并安装可视化依赖。必需组件导入失败会报错，不再通过 `importorskip` 隐藏缺失；这不改变产品中可选组件的公开导入策略。仅检查局部功能时可指定对应测试文件。

CI 在仓库外安装 wheel，先做依赖检查和 10 个原生扩展的来源/ABI 检查，再运行带 `-W error` 的全量测试。覆盖率启用 `--cov-branch`，每个 OS/Python 组合保存 XML、`coverage-summary.json` 和 `coverage-summary.md`，并写入 GitHub Job Summary。行与分支覆盖分别展示，暂不设置百分比阈值；可从同一组合的历史产物或 Codecov 查看趋势。覆盖 XML 中的代码路径会映射回仓库，JSON 保留 Python、平台与 CI 提交 SHA。

运行环境为 conda `py312`。全量回归：

```bash
conda run --no-capture-output -n py312 pytest -q -W error
```

校准无量纲指标专项：[test_calibration_metric_scaling.py](test_calibration_metric_scaling.py) 的 65 项覆盖 NSE/R²/PBIAS/Pearson/KGE/R-factor 独立参照与单位换算、mask/多模拟行、真实常数及均值舍入、零和/零均值与非零抵消、大偏移下可表示变化、独立模拟尺度及 GLUE/SUFI2 的实际评分与筛选。与 `test_calibration_metric_boundaries.py` 的形状/空观测/区间边界测试互补，不更改已有真退化错误协议。

推断提议尺度专项：[test_inference_proposal_scales.py](test_inference_proposal_scales.py) 的 43 项核对三个 MH 家族的高斯标准差/均匀半宽、独立随机生成器与实际分布方差、warm-up 和自适应后的单位换算、AMH 协方差下限与历史不足副本、固定参数及完整高斯相关协方差。固定随机数用于识别参数使用错误，不以精确达到最优点或短链后验收敛作为断言。

单输出约定专项：[test_surrogate_single_output_contract.py](test_surrogate_single_output_contract.py) 的 89 项覆盖七类模型与三种回归损失的原始/预处理拟合入口，拒绝多列 Y 的时机、向量/单列结果一致、失败重拟合失效、调参预检查、容器整体形状/样本轴、独立矩阵不确定性参照、子模型预测形状及容器失败恢复。所有多输出由 MultiSurrogate 管理，Scaler/评分函数仍可处理多列。

回归多输出专项：[test_surrogate_regression_multioutput.py](test_surrogate_regression_multioutput.py) 的 38 项已按上述约定迁到 MultiSurrogate，保留独立最小二乘/岭矩阵参照，覆盖两个模型、三种缩放配置、是否拟合截距、三个输出（含常数列）、空/单点/多点预测、子模型预处理入口、重拟合及单输出向量、Lasso 多输出提前拒绝和容器解析软阈值解。既有 GPR/KRG 多输出不确定性 10 项与 MARS 多输出导数 9 项改为容器对照，MARS 原始量纲评分 3 项逐列核对，未删除数学断言。

另有 GPR 独立似然/后验 12 项、joint log-density 1 项及 RBF 五种核的平滑/缩放 10 项迁到容器；仍用 NumPy 独立求解/对数行列式、SciPy 概率密度及 RBFInterpolator 核对数值，容器的独立输出负对数似然求和与原联合参照比较。

离散辅助映射专项：[test_problem_discrete_mapping.py](test_problem_discrete_mapping.py) 的 25 项覆盖整数/uint8/布尔/float32 输入、小数与负数选项、只读输入副本、混合变量及两个变换标志、Problem 委托入口；同时核对正式单位解码、真实值往返和模型实际接收值。

敏感性数值专项包括 `test_analysis_scientific_references.py`（独立数值参照）、`test_analysis_mars_reliability.py`（MARS 交互/留出告警）、`test_analysis_delta_scaling.py`（DeltaTest 单位缩放、带符号归一化及子集搜索）、`test_analysis_delta_ties.py`（等距权重/行排列/重复组）和 `test_analysis_output_range.py`（DeltaTest/MARS 极端输出单位、恒定性与训练隔离）。

2026-10-01 补充：`test_analysis_delta_subset_range.py` 验证极端输出下的实际 GA/穷举选择、多输出相对权重和目标尺度持久化；`test_analysis_morris_unit_effects.py` 通过解析效应核对标准单位区间计算、输入单位变换、非线性/交互标准差及持久化；`test_analysis_mars_stability.py` 复现高 R² 下的 GCV 搜索异常，并核对多次留出权重统计、原模型调用次数和诊断持久化。Morris 已统一单位区间效应，删除物理模式；这些测试不将 MARS 稳定性诊断当作全面准确性证明。

同日边界修复新增 50 项：`test_analysis_morris_unit_effects.py` 增加 uint8/bool 差分方向、整数输入及单轨迹检查；`test_analysis_rsa_statistics.py` 增加极端输出的手算 CvM 参照与非有限值拒绝；`test_analysis_rbd_fast_validation.py` 覆盖固定列、重复取值适用性、谐波预算和连续样本排列；`test_analysis_design_validation.py` 覆盖 FAST 完整块/元数据检查及 Sobol 稀有事件的基础方差诊断与总体参照。新用例包含正常对照和此前失败案例，不将无效配置的错误分数写成应保留行为。

RSA 样本不足告警补充 15 项：同一 `test_analysis_rsa_statistics.py` 覆盖小样本/稀有事件的 warning 与不可用状态、多输出选择及正常输出不受影响、手算 CvM=1.675 对照、方法复用、SQLite 诊断持久化和 nRegion 校验。既有恒定输出用例同时检查 `constant_output` 状态。预期 warning 显式捕获，全量仍使用 `-W error`。

Sobol 元数据复核共补充 28 项到 `test_analysis_design_validation.py`：先前 23 项按用户最新选择改为 warning / 可确认布局恢复 / 无法确认时未估计；额外 5 项验证歧义布局、可整除但无混合结构的输入、所选输出及缺失 Y 持久化、修正阶数和原元数据共同保存。整块缺失/多出继续计算，不完整块不丢行、不评价模型；正常 NumPy 元数据与解析效应对照保留。`test_analysis_sobol_more.py` 的原行数检查也采用 warning 和未估计标记。预期 warning 显式捕获；单位坐标、多输出及极端输出测试继续保留。

本目录沿用平铺结构，以模块和行为命名。下表列出 2026-09-29 从历史审查文件迁移的 488 项回归，不是整个测试套件；其他已有专项、工作流和文档测试仍保留原位置。

运行单个文件时把下表路径追加到上述命令即可。新增测试应放入对应模块/行为文件，避免继续使用审查日期或问题批次作为主文件名。

| 模块 | 测试文件 | 参数化后用例数 | 历史来源 |
|---|---|---:|---|
| analysis | [test_analysis_coordinates_and_effects.py](test_analysis_coordinates_and_effects.py) | 30 | `test_review_c01_c05.py` |
| analysis | [test_analysis_delta_validation.py](test_analysis_delta_validation.py) | 6 | `test_review_a07_a15.py` |
| analysis | [test_analysis_rsa_statistics.py](test_analysis_rsa_statistics.py) | 2 | `test_review_a01_a06.py` |
| analysis | [test_analysis_shape_contracts.py](test_analysis_shape_contracts.py) | 6 | `test_remaining_review.py` |
| autotuner | [test_autotuner_split_and_report_contracts.py](test_autotuner_split_and_report_contracts.py) | 37 | `test_review_c16_c18_c19_c21.py` |
| calibration | [test_calibration_ensemble_preflight.py](test_calibration_ensemble_preflight.py) | 23 | `test_review_c06_c12.py` |
| calibration | [test_calibration_evaluation_and_storage.py](test_calibration_evaluation_and_storage.py) | 4 | `test_remaining_review.py` |
| calibration | [test_calibration_gain_work.py](test_calibration_gain_work.py) | 5 | `test_review_remaining.py` |
| calibration | [test_calibration_projection_contracts.py](test_calibration_projection_contracts.py) | 22 | `test_review_c01_c05.py` |
| doe | [test_doe_documentation_examples.py](test_doe_documentation_examples.py) | 7 | `test_review_a07_a15.py` |
| inference | [test_inference_demc_configuration.py](test_inference_demc_configuration.py) | 5 | `test_review_a07_a15.py` |
| inference | [test_inference_diagnostics_and_partial_results.py](test_inference_diagnostics_and_partial_results.py) | 20 | `test_review_remaining.py` |
| inference | [test_inference_incremental_history.py](test_inference_incremental_history.py) | 15 | `test_review_c13_c14.py` |
| inference | [test_inference_result_isolation_and_directions.py](test_inference_result_isolation_and_directions.py) | 25 | `test_review_c01_c05.py` |
| inference | [test_inference_stop_reason_persistence.py](test_inference_stop_reason_persistence.py) | 5 | `test_review_c16_c18_c19_c21.py` |
| optimization | [test_optimization_asmo_reproducibility.py](test_optimization_asmo_reproducibility.py) | 1 | `test_review_a07_a15.py` |
| optimization | [test_optimization_capabilities_and_stop_reasons.py](test_optimization_capabilities_and_stop_reasons.py) | 29 | `test_review_c16_c18_c19_c21.py` |
| optimization | [test_optimization_configuration_lifecycle.py](test_optimization_configuration_lifecycle.py) | 26 | `test_review_c06_c12.py` |
| optimization | [test_optimization_ego_component_contracts.py](test_optimization_ego_component_contracts.py) | 1 | `test_review_remaining.py` |
| optimization | [test_optimization_evaluation_and_persistence.py](test_optimization_evaluation_and_persistence.py) | 22 | `test_remaining_review.py` |
| optimization | [test_optimization_hv_scheduling.py](test_optimization_hv_scheduling.py) | 33 | `test_review_c13_c14.py` |
| optimization | [test_optimization_reference_point_validation.py](test_optimization_reference_point_validation.py) | 18 | `test_review_a01_a06.py` |
| problem | [test_problem_public_contracts.py](test_problem_public_contracts.py) | 19 | `test_remaining_review.py` |
| runtime | [test_runtime_domain_and_identity.py](test_runtime_domain_and_identity.py) | 3 | `test_remaining_review.py` |
| runtime | [test_runtime_reader_field_names.py](test_runtime_reader_field_names.py) | 1 | `test_review_a07_a15.py` |
| surrogate | [test_surrogate_fit_and_validation.py](test_surrogate_fit_and_validation.py) | 30 | `test_review_a01_a06.py` |
| surrogate | [test_surrogate_mars_derivative_contracts.py](test_surrogate_mars_derivative_contracts.py) | 14 | `test_review_b01_b03.py` |
| surrogate | [test_surrogate_mars_refit_lifecycle.py](test_surrogate_mars_refit_lifecycle.py) | 3 | `test_review_c06_c12.py` |
| surrogate | [test_surrogate_prediction_work.py](test_surrogate_prediction_work.py) | 7 | `test_remaining_review.py` |
| surrogate | [test_surrogate_public_fit_contracts.py](test_surrogate_public_fit_contracts.py) | 26 | `test_review_b04_b06.py` |
| surrogate | [test_surrogate_randomness_and_failures.py](test_surrogate_randomness_and_failures.py) | 12 | `test_review_a07_a15.py` |
| surrogate | [test_surrogate_tuning_lifecycle.py](test_surrogate_tuning_lifecycle.py) | 17 | `test_review_b01_b03.py` |
| viz | [test_viz_smoothing_contracts.py](test_viz_smoothing_contracts.py) | 7 | `test_review_a07_a15.py` |
| viz | [test_viz_surrogate_limits.py](test_viz_surrogate_limits.py) | 7 | `test_review_c06_c12.py` |

辅助代码：

- [optimization_test_support.py](optimization_test_support.py)：停止语义与配置恢复共用的算法列表和工厂；每次调用创建新的算法实例。
- [algorithm_capability_test_support.py](algorithm_capability_test_support.py)：能力检查与停止原因持久化共用的问题工厂。

测试文件之间不互相导入。辅助模块不以 `test_` 开头，避免被当成测试文件收集。通用 pytest 配置仍在 `conftest.py`。

每个迁移函数上方保留旧文件及函数名；A/B/C 等审查编号保留在来源文件名中。完整参数化用例对应关系见 [迁移映射](../agent/verification/0929-test-migration-map.json)，验证过程见 [迁移记录](../agent/0929-test-migration.md)。

- `test_optimization_logic_and_range.py`：优化复形排序/重心/配置、ABC 计数、全评价最优记录、奇数交叉、方向几何与指标数值范围、非法结果和单目标惩罚协议。
