# 2026-10-04 测试结构整理与补强

## 结果

完成一次全目录静态盘点、重复候选审阅、辅助代码合并、数值断言补强和测试导航整理。**未删除既有案例、未降低容差、未修改种子或生产算法、未设置默认跳过。**

- 整理前：167 个测试文件、1000 个测试函数，参数化展开为 3084 项。
- 整理后：168 个测试文件、1003 个测试函数，参数化展开为 **3094 项**。
- [节点逐项对照](verification/1004-test-node-preservation.json)：原 3084 个节点全部保留，仅新增 10 个节点。
- 完整 py312 回归：**3094 passed，101.53 秒**，`-W error --strict-markers`；[完整日志与最慢案例](verification/1004-test-organization-full.txt)。预期 warning 仍由测试显式捕获。
- 第一轮改动专项：41 passed，1.69 秒；随后对最终辅助函数合并和标记版本执行了上述完整回归。

这次压缩的是重复辅助实现，没有为了让数字变小删除有效回归。上轮 3084 项约 100 秒，本轮耗时接近；没有宣称提速。

## 审查依据与实际改动

[AST 盘点脚本](verification/audit_test_structure.py) 扫描函数体，去掉文档字符串后寻找完全一致的候选；[整理前](verification/1004-test-structure-before.json)和[整理后](verification/1004-test-structure-after.json)均保留。未发现函数体完全相同的测试函数，重复候选主要是模拟目标、辅助断言和局部测试替身。该扫描不证明不存在语义重复，候选仍需结合输入、装饰器、调用路径审阅。

1. 两处优化结果断言合并到 `optimization_test_support.assertBenchmarkResult`，三个 smoke 文件共 14 个案例使用它。保留原断言，并新增有限值/边界检查及独立 Sphere、ZDT1 公式验证，核对保存的目标是否对应返回的参数；不会调用被测 Problem 求预期值。这验证结果一致性，不保证收敛到全局最优。
2. 两处等价的普通 Delta 稠密距离参照合并到 `analysis_test_support.pairwiseDelta`。并列截止邻居的加权参照保留独立实现，不能用普通排序参照替代。
3. 保留 LHS 原有 5 个案例的输入和 ID，补上每一维的完整分层检查，中心类型额外检查层中心。FFD 原案例由仅检查数量/边界补强为比较完整笛卡尔网格。
4. 从独立 DoE 审计迁入 10 项低成本明确判定的构造验证：Saltelli 一/二阶行替换 4 项、Morris 轨迹/层网格 4 项、FAST 独立三角波公式 2 项。保留更大样本/种子审计作为专项证据。
5. 已审阅的 149 项标记为 `numerical`，其中 8 项采样矩/分布案例标记为 `statistical`。pytest 默认仍执行全部；两组选择已用 `--collect-only --strict-markers` 核对，见[数值选择](verification/1004-numerical-selection.txt)、[统计选择](verification/1004-statistical-selection.txt)。
6. 新增[职责与数值依据导航](../tests/TEST_MATRIX.md)，说明模块入口、参照来源、选跑命令及已知限制，原测试导航/历史迁移记录保留。

没有把相似的短目标函数一律搬到公共模块：闭包、pickle、单行/批量语义和故障注入路径不同，强行合并会增加耦合。没有把短链靠近高概率点的 smoke 检查当成采样分布正确性的证据。

## 当前分布与运行分层

以下以测试文件主要职责分类，跨模块测试归 integration。不是生产覆盖率，尤其 runtime/viz 的跨领域案例也在 integration 中。

| 主要职责 | 展开项数 |
|---|---:|
| analysis | 369 |
| calibration | 418 |
| doe | 108 |
| inference | 340 |
| integration | 24 |
| optimization | 517 |
| problem | 199 |
| runtime | 46 |
| surrogate | 1051 |
| util | 6 |
| viz | 16 |

`numerical` 是本轮明确审阅的首批选择，不是仓库全部数值断言的全集；未标记不等于无数值验证。`statistical` 指有限样本矩/分布检查，不等于所有使用随机数的测试，也不等于所有慢测试。

本轮 8 项 statistical 的 call 阶段合计约 12.55 秒。最慢单项还包括 DTLZ7 前沿几何和 MARS 拟合。因此不把 `not statistical` 称为“极速全覆盖”。日常可按模块或标记选跑，最终仍应完整验收。

[机器清单](verification/1004-test-suite-after.json)保留每个节点、标记、领域及 setup/call/teardown 实测耗时，可用于后续维护；[生成脚本](verification/check_test_suite_inventory.py)运行 pytest 并输出该记录。

## 验证与边界

触达测试通过 Ruff lint/format，差异空白检查通过；默认测试范围与 CI 未被缩减。此次未重建 wheel，前轮 3.12/3.14 安装包的 3084 项结果仍是当时快照，不能写成新版本 3094 项安装包通过。本轮没有提交、推送或发布。

这是一轮测试工程整理和明确缺口补强，不是对全部 1000 个函数重新做逐行数学证明。后续触达时继续补充 marker 和独立参照；较长的多种子精度/混合诊断、远程多平台 CI、约束引导优化与精确断点续跑边界保持。
