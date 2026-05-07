# UQPyL 整体包架构整理计划

## 结论

当前 `UQPyL` 已经不是“若干工具模块的集合”，而是在演进成一个以 `problem` 为中心的统一 UQ 框架。

后续架构应围绕：

1. `problem` 作为全包唯一协议中心
2. `surrogate / optimization / analysis / inference / doe` 作为各自自治的领域子系统
3. 极小公共层，只保留真正跨模块的能力
4. 当前阶段先稳住已有结构，不做大规模目录迁移

## 当前整体认识

### 顶层模块职责

| 模块 | 当前职责 | 判断 |
| --- | --- | --- |
| `problem` | 问题定义、空间、`Eval`、输入输出协议 | 应继续强化为全包中心 |
| `doe` | 采样设计 | 边界清晰 |
| `surrogate` | 代理模型、调参、内部优化器 | 应继续内部收敛 |
| `optimization` | 优化算法、种群、runtime、storage、verbose | 已形成完整子系统 |
| `analysis` | 敏感性分析、runtime、result | 已形成完整子系统 |
| `inference` | 推断/MCMC、chain、runtime | 已形成完整子系统 |
| `util` | 历史残留的杂项承载层 | 不适合作为长期稳定抽象 |

### 目前最核心的问题

当前最主要的问题不是某个文件放错位置，而是全包层面还缺少正式统一的抽象边界：

- `problem` 已经开始承担协议层角色，但还主要偏静态问题
- `optimization / analysis / inference` 各自长出了 runtime/result/verbose/storage 体系，但没有统一设计原则
- `surrogate` 作为领域子系统已经比较独立，但预处理、指标、切分等能力仍散落在顶层 `util`
- 顶层 `util` 已经不再满足“跨模块、低耦合、稳定公共层”的定义

## 目标架构方向

### 一、`problem` 作为唯一中心协议

`problem` 后续应承担“全包任务定义协议层”的职责，而不是仅仅承载静态优化测试函数。

建议后续形成下列协议族：

- `ProblemBase`
- `StaticProblemBase`
- `DynamicProblemBase`
- `AssimilationProblemBase`
- `Space`
- `Eval`
- `SimResult` / `Trajectory`
- `ObservationSet`
- `ObservationOperator`

### 二、静态 / 动态 / 同化三层问题协议

建议后续问题协议分为三层：

| 层 | 含义 | 服务对象 |
| --- | --- | --- |
| `StaticProblem` | 静态映射，输入一次得到目标/约束 | `optimization`、`surrogate`、大部分 `analysis` |
| `DynamicProblem` | 参数 + 时序驱动 -> 轨迹/状态序列 | 时序 surrogate、动态分析 |
| `AssimilationProblem` | 在动态问题上叠加观测、误差、窗口语义 | `EnKF`、`ES-MDA`、`IES`、未来同化方法 |

注意：

- `AssimilationProblem` 不应强行塞进现有 `objFunc(X)` 语义
- 统一的是“问题描述框架”，不是所有算法共享同一个求值接口

### 三、各领域子系统保持自治

建议后续每个领域子系统内部自带自己的局部生态，不再依赖顶层杂糅层：

| 子系统 | 内部应承担的能力 |
| --- | --- |
| `surrogate` | preprocess、metric、tuning、internal optimizers |
| `optimization` | population、operators、runtime、result、storage |
| `analysis` | runtime、result、record、输出检查 |
| `inference` | chain、runtime、result、log-prob 协议 |

### 四、公共层极小化

未来顶层只保留真正跨模块、无领域归属的能力。

当前判断：

| 文件 | 长期归属建议 |
| --- | --- |
| `plot.py` | 顶层公共展示层，可考虑独立为 `plotting/` 或 `viz/` |
| `metric.py` | 更偏 `surrogate` |
| `scaler.py` | 更偏 `surrogate` |
| `poly.py` | 更偏 `surrogate` |
| `split.py` | 当前更偏 `surrogate` 调参与数据切分 |
| `verbose.py` | 计划废弃，不建议迁移 |

因此，`util` 目录长期看不应继续作为稳定公共层存在。

## 当前阶段策略

### 原则

当前阶段不建议立刻做大规模迁移，先稳住结构，避免把“长期抽象方向讨论”和“短期代码搬迁”混在一起。

### 当前阶段只做三件事

1. 明确 `problem` 是唯一中心协议方向
2. 在新增代码中尽量减少对顶层 `util` 的扩散依赖
3. 将未来架构演进路线写清楚，等核心 reconstruction 稳定后再迁移

## 分阶段实施建议

### Phase 1：稳结构，不大搬

- 保持现有顶层目录结构
- 新增设计优先围绕 `problem` 中心展开
- 不再把新的 surrogate 专属能力放进顶层 `util`
- `verbose.py` 只减引用，不做迁移

### Phase 2：补协议，不急搬家

- 在 `problem` 中正式抽出 `Static / Dynamic / Assimilation` 三层语义
- 明确动态问题的 `simulate / observe / residual / misfit` 接口
- 让未来同化方法围绕 `AssimilationProblem` 设计，而不是直接借优化目标接口

### Phase 3：按边界迁移

- 将 `scaler / poly / split / metric` 迁入 `surrogate`
- 将 `plot` 独立为公共展示层
- 清空并移除顶层 `util`

## 当前不做的事

- 现在不重排整个目录树
- 现在不同时做 runtime 统一抽象和 util 迁移
- 现在不为了未来同化场景去改写全部静态问题接口

## 建议的后续讨论顺序

建议后续架构讨论按以下顺序推进：

1. 确认 `problem` 的三层协议是否成立：`Static / Dynamic / Assimilation`
2. 确认顶层公共层是否只保留展示能力
3. 确认 `surrogate` 是否要内部收纳 preprocess / metric / tuning
4. 最后再做目录迁移与导入收敛

## 当前最终判断

一句话总结：

`UQPyL` 应该朝“以 `problem` 为中心的统一 UQ 任务框架”演进，而不是继续维护一个模糊的顶层 `util` 杂项层；但当前阶段应先稳住已有结构，先定协议和分层方向，再做迁移。
