# 推断公共结构统一（2026-10-03）

本轮统一公共参数校验、接受后状态更新、链记录和采样策略诊断。保留五种算法的提议、随机数调用、更新次序和适应阶段；没有将不同算法强制套入一个运行循环。

## 实现

- `InferenceABC.validateParameters` 集中校验链数、预热/正式步数、初始化上限、输出/保存频率、提议分布和 logProbFunc。每次运行重新检查，覆盖对象复用与 `set()` 修改。方法通过 `minChains` 声明最少链数；DREAM 覆盖该方法并调用父类后检查自己的配对、档案、周期等参数。
- gamma 使用同一检查函数。统一接受有限非负实数标量、参数向量、单行或完整链×参数矩阵。MH_Gibbs 原有重复检查删除，支持其接口原已声明的列表，也支持整数标量。零值保持允许；不再接受错误宽度、布尔值、NaN/inf 或负数。gamma 检查统一在首次用户模型评价之前进行。
- `updateChainState` 负责接受后的参数/目标/约束同步写入，`recordChainState` 负责当前占据状态、logProb 和接受标记追加。调用顺序由各方法保持，DREAM 的跳跃增益仍在覆盖当前点之前计算，DEMC 仍在下一条链提议前更新当前链。
- 删除五个重复的 `setProblem` 实现与 MH_Gibbs 的重复 gamma 实现，删除 AMH 的重复 gamma 检查。
- `setSamplerDiagnostics` 统一构建 `diagnostics["sampler"]`；所有方法提供 `boundary_policy`、`update_mode`、`adaptation_phase`、`proposal_family`、`proposal_settings`。提议设置至少含 gamma/distribution；DEMC 另含默认单位跳跃概率与噪声尺度，DREAM 另含 snooker 概率/配对数/jitter，原有档案和冻结信息继续保留。
- 返回结果继续为 `InfResult`，导出策略字段采用 snake_case。诊断保存至 SQLite，结果/导出嵌套副本隔离。

| 方法 | 边界 | 更新方式 | 适应阶段 |
|---|---|---|---|
| MH | reflect | independent | none |
| MH_Gibbs | reflect | coordinate_wise | none |
| AMH | reject | independent | formal_sampling |
| DEMC | reject | sequential | none |
| DREAM_ZS | reject | independent_given_archive | warmup_only |

数值算法默认值没有统一为同一值；不同方法需要不同设置。预热前已有的回调行为也保留，避免纯结构调整改变调用顺序。新增诊断字段描述策略，不代表收敛证明。

## 精确等价对照

任何生产改动之前保存了 **176组** 基线：11种方法/提议配置×4种问题×2个预热长度×2个种子。包括 MH/MH_Gibbs/AMH 的 Gaussian 与 uniform、DEMC 默认与显式 gamma、DREAM 的 ps=0/.1/1；问题覆盖连续、约束、混合变量、固定变量，同时覆盖最小化/最大化方向与自定义 logProbFunc。

重构后逐值比对全部一致，没有使用近似容差：

- 参数轨迹、目标、约束、logProb、接受标记与接受率、最佳参数/目标/约束及可行标记。
- FEs、迭代数、停止原因。
- 每次模型评价的输入、批次形状及顺序。
- 每次自定义 logProb 的输入及调用顺序。
- 运行完成后继续抽取的8个随机数。

运行时间和新增策略元数据不作为数值等价项。输入检查收紧和 MH_Gibbs 新接受的合法格式是有意的接口行为变化，不归为旧输入轨迹对照。

文件：[捕获/对照脚本](verification/check_inference_structure_parity.py)、[重构前基线](verification/1003-inference-structure-before.json)、[最终对照结果](verification/1003-inference-structure-parity.json)。脚本 `capture` 仅用于产生新基线，复核本次应使用 `compare`，不要覆盖已保存基线。

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/wmtsky/anaconda3/bin/conda run --no-capture-output -n py312 python agent/verification/check_inference_structure_parity.py compare
```

## 回归与文档

新增 [28项公共结构测试](../tests/test_inference_common_structure.py)：五方法公共无效参数、gamma无效输入不评价模型、gamma合法格式精确等价、复用后重新校验、修改后propDist检查、策略字段、SQLite往返与副本隔离。初次测试指出 MH/MH_Gibbs/DEMC 的 gamma 检查晚于模型调用，已统一提前。

既有推断专项在首轮重构后273项通过；最终全量 **2939 passed，104.95秒，-W error，无未捕获告警**，见 [日志](verification/1003-inference-structure-full.txt)。本轮使用 conda py312、单线程 BLAS；单独指定 pytest basetemp，避免并发目录冲突。

Ruff、触达文件差异检查通过。中英文用户/API 文档与测试导航已同步。本轮没有重复执行上一轮90次长链审查；完整回归包含既有分布正确性测试，另用176组精确轨迹对照验证此次结构调整。

未提交、推送或重建 wheel。DEMC 少链双峰混合限制保持；约束引导优化仍按用户要求暂缓。
