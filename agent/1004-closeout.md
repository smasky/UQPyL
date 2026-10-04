# 2026-10-04 本地收尾验收

本轮补查公共运行/存储、可视化的数据正确性，以及 Python 3.12/3.14 安装包。算法模块沿用此前逐模块审计与修复结论，并通过本轮全量回归。没有提交、推送或发布。

## 确认并修复的问题

| 位置 | 原问题 | 现在的行为 |
|---|---|---|
| `viz/analysis.py` | 绘图再次除以敏感性总和，改变数值和符号；纵轴固定 0–1；不同结果标签可能错位 | 直接绘制选定指标，显式选择 `metric` / `outputIndex`；保留负值和大于 1 的值，按参数名称对齐 |
| `viz/optimization.py` | 不同 FE/迭代位置的历史按行平均，横坐标采用最后一次运行 | 不一致时 warning，仅用实际共有坐标；不插值、外推；无共同坐标明确拒绝 |
| `viz/optimization.py` | 线性轴的统计区间负下界被截成正数 | 线性轴保留负下界，对数轴才使用正数下界 |
| `viz/optimization.py` | 二维 Pareto 参考点按行取坐标，三维忽略参考点 | 每行一个参考点，二维/三维均正确显示并检查形状 |
| `viz/inference.py` | 选择第三个变量后，轨迹标题仍显示第一个变量 | 标题保持原变量编号，数据与编号一致 |
| `core/runtime_storage.py` | 数据库另取创建时间，可能与内存结果相差一秒 | 保存运行状态的创建时间，缺少该字段时才取当前时间 |

敏感性非有限指标发出 `RuntimeWarning` 并跳过对应柱，不修改结果；缺失指标、标签集合不同或无法比较的历史仍明确报错。这些处理没有将无效值替换成看似正常的数值。

中英文分析/优化指南及测试导航已同步。新增 `tests/test_closeout_runtime_and_plot_data.py` 共 11 项。

## 运行与存储的证据

四领域使用真实 GA、MH、Morris、GLUE 运行，检查 SQLite 完整性、完成状态、资源关闭、创建时间、reader 往返及复用后旧结果不变。

优化 reader 按既有协议恢复保存的种群快照，并非所有内存中的逐代种群。对此检查最终数值、标量历史和保存数据的对应关系；其他三个领域完整比较结果树。没有把快照协议误当作精确断点续跑。

现有故障注入回归同步通过，包含初始化/模型/保存/结束阶段异常、事务回滚、序列化失败、KeyboardInterrupt、清理失败不覆盖原异常、领域误读和运行 ID 冲突。

## 图形数据验收

[独立脚本](verification/check_closeout_figures.py) 生成并核对六类图：带符号的 Morris 指标、负值优化统计区间、Pareto 参考点、选择变量的推断轨迹、后验分布/区间、替代模型预测散点。检查实际 artist 数值并导出图片，已查看[预览](verification/1004-figures/preview.png)。三变量分布图仍使用矩形网格，存在一个空子图；这是布局表现，不影响数据。

详细输出：[检查记录](verification/1004-figures/checks.json)、[日志](verification/1004-figure-audit.txt)。校准模块没有独立绘图 API，本轮对其验证的是结果持久化。

## 测试与构建

- 初始旧专项：75 passed，16.89 秒，见 [日志](verification/1004-closeout-initial.txt)。
- 修复后的运行/可视化专项：73 passed，9.11 秒，见 [日志](verification/1004-closeout-targeted.txt)。两个专项选取的文件范围不同，不能按总数相减理解。
- conda py312 源码全量：**3084 passed，100.23 秒**，见 [日志](verification/1004-closeout-full.txt)。
- 全量均使用 `-W error`；预期的 warning 由相应测试显式捕获，未用全局忽略掩盖。
- wheel 在两份独立源码副本中构建，避免覆盖工作区原生扩展。安装包测试使用仓库外新建 venv，确认包来自 site-packages、全部 10 个原生扩展成功导入，并执行依赖检查和全量测试。

| Linux 安装包 | 全量结果 | Python 行覆盖率 | Python 分支覆盖率 |
|---|---|---|---|
| Python 3.12.0 | 3084 passed in 134.52s | 94.15% | 83.22% |
| Python 3.14.7 | 3084 passed in 119.77s | 94.14% | 83.22% |

[3.12 日志](verification/1004-wheel312-test.txt)、[3.14 日志](verification/1004-wheel314-test.txt)、[版本/覆盖率/wheel SHA256](verification/1004-closeout-summary.json)。覆盖率只针对 Python 源码，不代表 C/C++ 分支覆盖率，也不能替代精度验证。

文档更新后专项复查：15 passed，1.21 秒，见[日志](verification/1004-closeout-docs.txt)。

## 格式与验证边界

既有改动中 28 个文件未通过 CI 的 Ruff 格式检查，本轮进行了机械格式化。[AST 检查记录](verification/1004-format-ast.json) 证明格式化前后可执行 AST 不变（忽略文档字符串）。这不是新增 28 项逻辑修复。最终 Ruff lint、format 检查和 `git diff --check` 通过。

本轮仅实测 Linux Python 3.12 和 3.14；没有推送触发远程 CI，不能代表 Windows/macOS 或其余 Python 版本已通过。构建日志中的 setuptools 许可证元数据弃用提示没有阻止构建，不属于算法运行告警。

目前主要模块及公共基础设施都有审计/回归记录，已确认的本轮缺陷已修复；这不构成对所有输入绝无缺陷的证明。MARS 近似精度、MCMC 慢混合、随机优化有限预算下的搜索能力仍属于需按实际案例判断的限制。约束引导优化和精确断点续跑继续暂缓。
