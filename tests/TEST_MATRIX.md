# 测试职责与数值依据

按用户要求已删除 obsLabels；观测协议专项现为 50 项（4 项标签校验改为 1 项接口移除检查），四方法存储往返补充无标签断言。见[删除记录](../agent/1004-remove-observation-labels.md)。下方 53 项为迁移当轮记录。

观测接口已迁移至 `obs/mask (nObs,)`、`sims (nSamples,nObs)`。新增 [53 项协议/排列/存储验证](test_observation_vector_contract.py) 和 3 项真实校准文档执行测试，详见 [迁移验收](../agent/1004-observation-vector-migration.md)。下方早期机器清单为历史快照。

本表是阅读入口，不代表所有方法/输入组合已穷尽。完整收集清单、分组数量和耗时见 [本轮机器记录](../agent/verification/1004-test-suite-after.json)。按模块统计以测试文件主要职责归类，跨模块流程归入 integration；它不是生产代码行覆盖率。

## 按问题选择测试

| 模块 | 主要检查 | 独立数值依据与入口 | 仍需专项判断 |
|---|---|---|---|
| 敏感性 | 指标量级/符号、交互、单位、并列邻居、输入设计 | [科学参照](test_analysis_scientific_references.py)：解析线性/乘积效应、RSA 手算；[Ishigami](test_analysis_benchmarks.py)；[Delta 并列距离](test_analysis_delta_ties.py) | MARS 近似精度依赖函数结构和样本；高 R² 不等于敏感性必然准确 |
| 校准 | ES/IES 更新、似然权重/区间、评分数值范围 | [后验矩](test_calibration_posterior_moments.py)：Kalman、固定随机 MAP、独立正规方程；[精度](test_calibration_accuracy_repairs.py)：解析后验、SciPy MAP；[极端评分](test_calibration_range_and_ess.py)：Decimal | 非线性/多峰、边界和小有效样本下的后验质量 |
| 推断 | 接受规则、更新次序、边界、适应期、采样分布 | [转移及分布](test_inference_transition_correctness.py)：Hastings 比、高斯矩、截断目标数值积分；[诊断](test_inference_diagnostics.py)：外部参照数据 | 短链找到高概率点不等于分布正确；多峰慢混合仍需长链和多个种子 |
| 优化 | 已评价结果、方向、预算、边界、选择与历史 | [逻辑及范围](test_optimization_logic_and_range.py)、[指标](test_optimization_utils_and_metrics.py)；三组 smoke 测试用[独立 Sphere/ZDT1 公式](optimization_test_support.py)核对返回点和目标值 | 返回值正确不保证有限预算达到全局最优；带约束的进一步增强暂缓 |
| 替代模型 | 训练/预测协议、独立测试集精度、均值/方差/似然 | [GPR 稠密求解参照](test_gpr_likelihood_crosschecks.py)、[独立点预测](test_surrogate_prediction_accuracy.py)、[求解边界](test_surrogate_solver_boundaries.py) | 有限函数集合上的精度不能保证新任务精度或区间校准 |
| DoE | 分层、网格、Saltelli 行结构、Morris 轨迹、FAST 构造 | [采样](test_doe_sampling.py)、[设计结构](test_doe_design_structure.py)、[边界](test_doe_edge_regressions.py) | 结构正确不代表任意小样本都足以估计敏感性 |
| Problem | 评价行对应、变量类型、范围映射、缓冲区隔离 | [转换回归](test_problem_conversion_regressions.py)：Decimal 极端范围；[Eval 协议](test_problem_eval_contract.py) | 任意用户回调的业务公式由使用者负责 |
| 公共运行/存储 | 生命周期、回滚、故障清理、领域/ID、reader 往返 | [故障注入](test_runtime_failures.py)、[四领域真实往返](test_closeout_runtime_and_plot_data.py) | 保存快照不等于精确断点续跑 |
| 可视化 | artist 数据、符号/坐标、参考点、变量标题 | [绘图数据](test_closeout_runtime_and_plot_data.py)、[SQLite 绘图](test_viz_sqlite_plots.py) | 外观审阅另见六类导出图；无跨平台字体一致性保证 |
| 工具/集成 | 评分、拆分、导入、文档工作流、随机状态 | [metric](test_util_metric.py)、[split](test_util_split.py)、[文档流程](test_documented_workflows.py)、[用户流程](test_user_workflows.py) | 源码测试不替代安装包和多平台测试 |

## 运行分组

使用 conda `py312`；设置 BLAS 线程数是为了减少小矩阵测试的线程调度开销。下列命令在仓库根目录执行。

```bash
# 完整回归：默认包含统计验证
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run --no-capture-output -n py312 pytest -q -W error --strict-markers

# 本轮已审阅并标记的数值依据/精度案例（不是全部数值测试的全集）
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run --no-capture-output -n py312 pytest -q -W error -m numerical

# 已标记的采样矩/分布验证
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run --no-capture-output -n py312 pytest -q -W error -m statistical

# 日常局部迭代可排除以上统计案例；不能替代最终全量验收
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run --no-capture-output -n py312 pytest -q -W error -m 'not statistical'
```

`numerical` 表示有已审阅的数值参照、不变量或独立测试集断言。`statistical` 是其中需要有限样本容差的采样矩/分布案例；它不是“所有随机测试”或“所有慢测试”。未标记不代表没有数值验证，后续触达测试时可继续补标。没有按运行时间自动跳过、重试失败、更换种子或放宽阈值。

并行启动多个 pytest 进程时，各自指定不同的 `--basetemp`。

## 专项审计与正式回归的分工

| 专项 | 正式回归已保留/本轮补充 | 较大审计保留位置 |
|---|---|---|
| DoE 独立构造 | 本轮补强 LHS 分层/中心点和 FFD 网格；新增 Saltelli、Morris、FAST 共 10 项 | [194 条审计记录说明](../agent/1003-doe-review.md)及其原始输出 |
| 推断分布 | 有限固定种子的解析/积分分布回归 | [多分布及种子复测](../agent/1002-inference-distribution-review.md)、[修复验收](../agent/1002-inference-distribution-fixes.md) |
| 优化搜索质量 | 预算、结果、选择逻辑和指标正确性 | [困难案例](../agent/1002-optimization-hard-cases.md)：多维、多种子、预算比较 |
| 绘图 | 数值/坐标/读取结果回归 | [六类图形审计脚本](../agent/verification/check_closeout_figures.py)和[图片](../agent/verification/1004-figures/preview.png) |
| 安装包 | 包入口测试、全量 pytest | [3.12/3.14 本地安装包验收](../agent/1004-closeout.md)；本轮测试整理后尚未重建安装包 |

历史审计可能包含修复前反例，查看对应修复报告后再解释，不应直接把旧结果当作当前缺陷。体量较大的专项脚本不自动纳入默认 pytest；具备明确预期且运行成本低的案例优先迁入。

## 维护约定

- 一个已确认缺陷至少保留一个可追溯的最小复现；同一算法的不同边界、返回协议和公共调用路径不能仅因名称相近而删除。
- 只抽取确实相同的准备/断言。独立数值参照不得调用被测实现计算预期值。
- 普通 Delta 距离参照与并列邻居加权参照分开维护；两者适用条件不同。
- 公共辅助模块不命名为 `test_`，测试文件不互相导入；随机对象每次新建，不共享可变实例。
- 合并/迁移时保存原节点到新节点的对应关系；扩大覆盖时记录新增依据，不能用减少数量替代质量判断。
