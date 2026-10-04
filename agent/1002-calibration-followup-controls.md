# 校准模块后续问题处理（2026-10-02）

本轮对应用户提出的五项：SUFI2 参数校验、范围收缩、GLUE 加权区间、IES 收敛控制、ES/IES 边界影响。明确缺陷已修复，新增可选控制；没有把探索策略或边界处理宣称为普遍正确的后验估计。

## 实现与默认行为

### SUFI2

- 在模拟前验证 maxIters、样本数、精英数及范围保护参数。`maxIters=0` 不再触发内部 NoneType，RuntimeWarning 后做一次筛选；负数/非整数等无法合理执行的配置仍明确拒绝。
- 内部采样默认保留 `explorationFraction=0.1` 原始域探索，局部非离散参数宽度至少为原始宽度的 `minRangeFraction=0.05`。首轮完整原始域；后续每轮全局样本数为比例向上取整，局部/全局合计仍为 nSamples，不增加模拟样本预算。
- 固定参数仍固定，整数及离散采样保持合法；探索允许重新引入此前被排除的原始离散选项。局部范围按精英包络加宽并限制在原始边界内。
- `updatedLb/updatedUb/updatedVarSet` 保持精英包络含义；history 增加实际局部采样范围及 explorationCount。外部提供 X 时只筛选、不重新采样。保护参数均为 0 时可以复现纯包络收缩。
- 这修补的是当前简化采样策略的过早冻结风险，不是宣称已实现完整文献 SUFI-2。

### GLUE

- `run` 新增 `logLikelihood=None, interval=0.95`。可选回调接收完整展平观测、行为样本模拟和 mask，每个行为样本返回一个 log 权重。
- 减最大 log 值后归一化，允许 -inf 表示零质量；NaN/+inf/全部零质量明确拒绝。默认等权并记录 weighting=uniform，不把 RMSE/NSE 自动转换为概率。
- 返回 behavioralWeights、effectiveSampleSize、ppuLower/ppuUpper，区间采用每个未屏蔽输出的逆加权经验 CDF，不插值。SQLite 往返已验证。
- best 仍按原评分选择。权重的统计解释取决于用户提供的先验/提议样本及似然，区间不额外包含未来观测噪声、不保证名义覆盖率。

### IES

- 新增 `adaptive=False, tolerance=1e-6, maxBacktracks=8`。**回溯为可选功能**，默认固定步仍保持上轮经过 Equinor 对照的方程及模拟预算。
- 开启后基于原始先验与固定扰动观测的 RML 目标检查真实模拟；恶化则步长逐次减半，最多 1+maxBacktracks 次尝试。目标使用初始先验子空间，而非配置的 RMSE 等汇总指标；裁剪后离开该仿射支撑的候选不会被错误赋予零先验惩罚。
- 零/奇异 R 采用硬观测残差优先、软观测与先验代价次之的退化扩展，没有把零噪声残差忽略。无法获得可接受步时 RuntimeWarning 并保留上一接受集合；模拟器本身的非法输出/异常不被吞掉。
- stopReason 区分 step_tolerance / line_search_stalled / iteration_budget；lineSearch 保存真实目标、步长及接受状态。小步只代表当前更新停滞，不是全局收敛证明。
- 这是固定阻尼 GN 的步长回溯，不是自适应调节 lam 的完整 LM 算法。[EnRML 理论参考](https://npg.copernicus.org/articles/26/325/2019/)用于明确先验项和固定扰动目标；新控制为本项目实现。

### ES / IES 边界

- 保留默认 boundHandling="clip"，增加可选 "rescale"：从当前成员出发，整条更新方向缩到最大可行步的 99%，减少直接裁剪引起的边界堆积。
- boundEffects 新增调整比例、调整前后均值/极差、是否保持无约束矩的标识；IES 回溯的 boundUpdates 另标记该试步是否接受。
- rescale 已在边界且向外时仍可能停住，**两种方式都不是精确截断高斯采样**。边界导致的统计改变只能在此范围内缓解、明确记录，不能标为根本消除。

## 测试与独立实验

新增 [27 项测试](../tests/test_calibration_followup_controls.py)：预检不调用模拟器、零轮 warning、单精英保护、混合域全局探索、手算权重/有效样本量/经验分位数、mask、零权重及非法权重、SQLite、立方过冲回溯、独立 RML 目标复算、失败保留、线性零/正噪声固定点、边界散布。

原 6 项混合域纯收缩测试显式设置两个保护参数为 0，保留旧数学断言；另 3 项验证默认新采样保护，不以删除检查迁就算法变化。

专项 **356 passed，2.12 秒**，全量 py312 **2733 passed，56.80 秒，`-W error` 零未捕获警告**。Ruff 与触达文件差异空白检查通过。日志：[专项](verification/1002-calibration-followup-targeted.txt)、[全量](verification/1002-calibration-followup-full.txt)。

[独立实验脚本](verification/check_calibration_followup.py)保存 [28 条数据](verification/1002-calibration-followup.json)及[输出](verification/1002-calibration-followup.txt)：

- SUFI2 单精英 10 个种子：9 个最终误差改善，1 个较差。seed=7 从 0.003303 增至 0.005632，原样保留；保护不能保证每次最终批次都优于旧版，更不保证全局最优。
- IES 立方/指数/正弦，各 3 种子，固定步/回溯两种模式共 18 次。回溯接受的实际 RML 目标不恶化。立方均按步长容差停止；指数 seed11 与正弦 seed11/47 停在 line_search_stalled 并告警；其他有按预算结束的情况。没有将这些状态称为全部收敛。最终目标往往与固定步相近，主要收益是拒绝恶化和诚实的终止诊断。

中文/英文 API、指南和测试导航同步。未提交/推送、未重建 wheel、未新增依赖；一般大规模 R 优化等原暂缓事项保持。
