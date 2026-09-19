# 项目代码复审清单 · 2026-09-18

后续状态：用户已授权并完成 A01—A06 修复，全量 1459 项通过，见 [修复记录](0918-a01-a06-fixes.md)。A07—A15 已于 2026-09-19 完成，全量 1498 项通过，见 [修复记录](0919-a07-a15-fixes.md)。下文为修复前审查及原始复现记录。

## 范围与证据

基于 dev `32dc639` 加当前未提交的 LBFGSB 修复。全包 169 个 Python 文件均通过 AST 解析；通过模式扫描及调用链抽查覆盖 problem、optimization、surrogate、analysis、inference、calibration、runtime/reader、viz。重点复核可变状态、边界输入、随机流、数值评分与错误出口；不是逐行形式化验证，也未重新审计 MARS 的 Cython 数值实现。

本轮没有修改生产代码，没有重新运行未修改代码的全套测试。上一轮 1409 项通过不能证明这些未覆盖边界正确。[复现脚本](verification/project_review_0918.py) 在 conda py312 下运行，[结果](verification/0918-project-review-probes.json)保存实际输出。超时复现使用独立子进程并终止，避免挂住审查进程。

P1：可能挂起或静默错误；P2：边界功能/接口问题；P3：维护性。以下均为新审查条目，旧 N/S/D 完成状态不撤回。Scaler 项是显式共享实例，不是已经修复的核默认对象共享；预算超出、Boxmin 默认、MOASMO 约束暂缓均按已确认决定，不列问题。

## 功能与逻辑

| 编号 | 级别 | 问题 / 证据 | 位置 | 建议 |
|---|---|---|---|---|
| A01 | P1 | NBI 单目标死循环：`comb(H1+1,0)` 恒为 1，`uniformPoint(8,1)` 进入后 5 秒未返回。MOEAD/NSGAIII/RVEA 入口未拒绝单目标，可能触发 | [uniform_point.py:17](../UQPyL/optimization/core/uniform_point.py#L17) | 工具函数处理 M=1；多目标算法明确检查目标维数，非法维数/点数入口报错 |
| A02 | P1 | 两模型传同一 Scaler 实例，训练第二个后第一个预测被改变；复现最大变化 0.9897 | [base.py:33](../UQPyL/surrogate/base.py#L33) | 明确 Scaler 所有权，安装时复制或禁止跨模型共享；测试 xScaler/yScaler |
| A03 | P1 | 重拟合不是原子操作：先重拟合 Scaler，再初始化组件/优化，途中失败仍可能保留旧 fitState。注入组件初始化异常后 predict 仍成功，结果变化 0.9897 | [base.py:74](../UQPyL/surrogate/base.py#L74)、[fit:304](../UQPyL/surrogate/base.py#L304) | 选择失败后整体失效，或成功后统一替换状态；不能新预处理配旧参数 |
| A04 | P1 | 指标未统一样本形状，`mse([1,2,3], [[1],[2],[3]])` 经广播得到非零三列结果，相同数值被误评分；r_square/nse 同样直接相减 | [metric.py:3](../UQPyL/surrogate/metric.py#L3) | 统一单输出为 (n,1)，检查行数和输出数，拒绝隐式跨样本广播 |
| A05 | P1 | AutoTuner 默认 10% 验证比例可只有 1 个点，R² 分母为 0；所有候选无有效分数时 gridTune 静默选首个并返回 -inf。8 点数据已复现；optTune 没有与 gridTune 一致的非有限评分检查 | [auto_tuner.py:99](../UQPyL/surrogate/auto_tuner.py#L99)、[244](../UQPyL/surrogate/auto_tuner.py#L244) | 验证集大小及可评分性入口检查；所有候选失败明确报错；统一两种调参失败协议 |
| A06 | P1 | RSA 把“组内输出值相同”当成不可分析，二值输出 `Y=(X>.5)` 对完全决定它的 X 返回 S1=0；代码因此跳过全部有效分组 | [rsa.py:145](../UQPyL/analysis/methods/rsa.py#L145) | 检查被比较的输入样本组是否满足统计要求，别用组内 Y 的唯一值数屏蔽分组；补离散/阈值输出回归 |
| A07 | P2 | DeltaTest 未处理一维删除后零列输入、以及 nNeighbors>=nSamples；一维 8 点和默认两样本分别触发 KDTree/数组 IndexError | [delta.py:76](../UQPyL/analysis/methods/delta.py#L76)、[213](../UQPyL/analysis/methods/delta.py#L213) | 明确单输入语义或入口拒绝；验证邻居数为合法正整数且小于样本数 |
| A08 | P2 | rank_score 展平多输出，却仅遍历 nSamples 个元素，后半输出被遗漏；第二输出反序仍返回 1.0。单样本也存在零分母 | [metric.py:30](../UQPyL/surrogate/metric.py#L30) | 逐输出计算或明确仅接受单输出，定义少样本和 ties 的返回语义 |
| A09 | P2 | smooth_curve 的 window 大于历史长度时，convolve 的结果比输入长，后续赋值广播失败；3 个点、默认 window=10 已复现 | [common.py:17](../UQPyL/viz/common.py#L17) | 限制窗口或短序列直接返回，覆盖空/1点/短历史 |
| A10 | P2 | DEMC 默认 nChains=1，但运行要求 >=3；默认构造无法正常运行 | [demc.py:26](../UQPyL/inference/methods/demc.py#L26)、[147](../UQPyL/inference/methods/demc.py#L147) | 合法默认值，并在构造时校验 |
| A11 | P2 / 静态确认 | 代理辅助优化只把子 seed 传给内层优化器，_fitSurrogate 不设置模型 RNG；MultiSurrogate.rng 也未分发到子模型。仅给外层 run(seed=...) 不能保证随机 GPR/KRG 的拟合复现 | [_base.py:9](../UQPyL/optimization/expensive/_base.py#L9)、[base.py:336](../UQPyL/surrogate/base.py#L336) | 明确外层 seed 是否覆盖代理模型，按子流传递；补两个全新实例的端到端复现测试。这里未量化最终优化结果偏差 |

A03 为故障注入验证，证明异常路径状态会混用，不声称普通输入每次必然触发。A11 是随机流调用链问题，未把某个具体优化结果差异写成已复现。

## 代码风格与可维护性

| 编号 | 级别 | 问题 | 位置 / 建议 |
|---|---|---|---|
| A12 | P2 | 可选依赖导入捕获全部 Exception，真实编程错误也会变成 MARS=None；AutoTuner 也捕获所有异常并 print，难以区分候选数值失败与代码错误 | [analysis/__init__.py:10](../UQPyL/analysis/__init__.py#L10)、[methods/__init__.py:10](../UQPyL/analysis/methods/__init__.py#L10)、[auto_tuner.py:141](../UQPyL/surrogate/auto_tuner.py#L141)。限定异常范围并保留原因；提供结构化失败诊断或 warnings |
| A13 | P3 | 存在残留死代码和无用导入：AMH/DEMC 的 _check_alpha 全仓无调用，实际使用基类 _check_gamma_；KRG 仍导入 GA/r_square/RandSelect 而无使用 | [amh.py:155](../UQPyL/inference/methods/amh.py#L155)、[demc.py:125](../UQPyL/inference/methods/demc.py#L125)、[kriging.py:14](../UQPyL/surrogate/kriging/kriging.py#L14)。按调用关系局部删除，不再留重复校验器 |
| A14 | P3 | 内部命名、公共导出命名仍混用；BaseReader 输出 run_id 却同时输出 dbPath/fileName，违反当前导出 snake_case 约定；内部仍有 _check_alpha、check_bound 等旧风格 | [runtime_reader.py:66](../UQPyL/core/runtime_reader.py#L66)。先统一实际返回协议及其消费者；内部名称随触达整理，避免全仓机械重命名 |
| A15 | P3 | 示例与接口失配：FFD 示例仍导入不存在的 UQPyL.problems；注解也有 r_square 标 ndarray 实际返回 scalar 等偏差 | [full_fact.py:13](../UQPyL/doe/methods/full_fact.py#L13)、[metric.py:3](../UQPyL/surrogate/metric.py#L3)。补关键示例可执行检查，校正返回注解 |

## 推荐处理顺序

1. A01/A02/A03：挂起和模型状态污染。
2. A04/A05/A06/A08：评分与分析的静默错误。
3. A07/A09/A10/A11/A12：边界、复现与错误诊断。
4. A13—A15：清理和规范，随功能修复触达。

本清单不代表已经授权逐项修复。当前生产修改仅为上一轮尚未提交的 LBFGSB 修复，本轮新增的是审查报告、复现脚本和证据。
