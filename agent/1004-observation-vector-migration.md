# 2026-10-04 观测接口一维化验收

用户明确要求“直接完成”，本轮已完成正式迁移。此前“暂定/未迁移”记录为历史状态。

## 当前协议

| 对象 | 正式形状/语义 |
|---|---|
| `obs` | 非空数值 ndarray `(nObs,)`；可省略以运行纯模拟 |
| `mask` | 可选 `(nObs,)`，True 忽略；必须与 obs 对齐，不能脱离 obs 提供 |
| `simFunc(X)` / `Eval.sims` / `SimContext.sims` | 数值 ndarray `(nSamples,nObs)`；无 obs 时仍要求非空列的二维矩阵 |
| 单观测 / 单样本 | 分别保留 `(1,)` / `(1,nObs)`，不 squeeze |
| 一维 X | 仍表示一个参数样本；单参数多样本用 `(N,1)` |
| `obsLabels` | 每个观测点一个标签，长度 nObs；默认 obs_1、obs_2 等 |
| ES/IES 的 R | 对应未被 mask 的观测，形状 `(nValidObs,nValidObs)`，顺序与有效模拟列一致 |

旧二维 obs、三维 sims、列数不匹配均明确拒绝；不自动猜测展平/转置顺序。用户可在模型回调返回前显式将网格 reshape 为二维，并按相同顺序准备 obs/mask。masked NaN 仍允许；未屏蔽 NaN 仍拒绝。既有数值 mask 转 bool 行为未改变。

`evaluate(X,target=...)` 仍是正式外部评价入口，返回 Eval；objFunc/conFunc 仍是用户回调。目标和约束矩阵的协议未改。`flattenSim` 保留为二维矩阵校验/返回入口，不再接受三维输入；flattenObs 返回现有向量，flattenMask 在省略 mask 时提供全 False 向量。

## 结果与存储

- ModelProblem 移除 obsShape、seriesLabels；CalResult 移除 nTime/nSeries，保存 nObs、obsLabels、一维 obs/mask。
- SQLite run 表移除 nTime/nSeries，使用 nObs；校准数据库 `PRAGMA user_version=1`。
- result/reader 汇总均为 `n_obs`、`obs_labels`，`n_output=n_obs`。不再输出 n_time/n_series。
- CalReader 拒绝旧 schema，明确要求重新运行；list_runs 跳过不兼容数据库。没有提供历史数据库转换，也没有改写现有数据库。
- 四种方法的数据库往返均核对结果、问题对象、标签、观测、掩码和模拟矩阵。

## 数值及回归证据

1. 修改生产代码前保存四方法 × 有/无 mask 的 42 个结果数组，包含 samples、simulations、scores、bestDecs、bestSim 及可用 weights。固定初始集合、种子和 R，保留 masked NaN。
2. 新接口复跑，42 个数组 **逐元素完全一致**，没有放宽容差或仅比较形状。见 [基线 NPZ](verification/1004-observation-migration-baseline.npz)、[对照脚本](verification/check_observation_migration.py)、[输出](verification/1004-observation-migration-parity.txt)。脚本的 --baseline 分支用于旧实现阶段，当前代码应运行默认对照分支。
3. 新增 53 项协议/单例维度/顺序/存储测试、3 项实际执行校准文档的测试。四算法的同步观测排列不变性通过；ES 非对角、非均匀 R 按有效观测重排后更新不变。中英文 README 校准示例另外实际执行通过。
4. **conda py312 全量 3178 passed，115.27 秒，`-W error --strict-markers`**。见 [完整日志](verification/1004-observation-migration-full.txt)。运行解释器 `/home/wmtsky/anaconda3/envs/py312/bin/python`，BLAS/OMP 各 1 线程。
5. 生产代码和 .github、新增协议测试、文档测试、文档 Python 辅助模型的 Ruff 检查通过；生产代码格式检查、git diff --check 通过。额外扫描整个 tests 仍有 18 项既有 E701/E702/E731 风格问题，本轮未将该扫描列为通过，也未为此重排无关测试。

原 3122 项均保留对应职责；旧协议断言按新协议更新，3 个测试函数重命名见 [映射记录](verification/1004-observation-test-mapping.json)。新总数 3178 = 3122 + 56。旧科学数值、统计精度断言和阈值未放宽；没有通过隐藏错误/跳过案例来完成迁移。

中英文 Problem/Calibration/API/Quick Start/Examples、README、SQLite 演示辅助模型、Changelog、测试职责导航同步更新。此前网格/展平双布局审计和历史 agent 脚本按原快照保留，不应直接用于新协议。

## 交付边界

本轮完成源码接口迁移及回归，不代表任意用户观测排列都可被框架推断正确：相同长度但顺序不一致仍由用户负责。算法更新公式没有改变。

没有提交、推送、触发远程 CI、重建 wheel 或上传 PyPI。此前 3094 项安装包验收是旧快照；最新源码的安装包与跨平台矩阵仍需按现有发布流程验收。版本仍为待发布的 2.1.7。
