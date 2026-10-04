# 2026-09-19 三轮全局审查：模块串联、使用体验、算法与性能、代码风格

用户决定：C20 精确断点续跑暂缓，推断与优化算法均不继续扩展 checkpoint；保留现有结果保存和部分结果读取。当前 C15 一般 R 优化、C17 约束选点也继续暂缓。下一阶段建议先进行端到端使用流程复核，再验证独立安装包；不发布正式版。

后续状态：2026-09-19 后续处理：**C01—C14、C16、C18、C19（单划分入口）、C21—C24 已完成本轮范围**；C15（一般 R 的稠密内存）、C17（约束引导选点）、C20（精确续跑）剩余部分均暂缓。具体限制与后续验收已记录在 [C15—C24 处理记录](0919-c15-c24-fixes.md)，不得计为全部修复。累计新增 167 项回归，最新全量 **1906 passed，零警告**（conda py312，`-W error`）；Ruff 格式/静态检查通过，136 个机械整理文件的可执行 AST 未变化。C21 现代 R-hat 与 bulk/tail ESS 已按需实现并通过固定参考对照，见 [诊断记录](0919-c21-diagnostics.md)。继续保留开发状态，不发布正式版。 C15 默认零误差已增加规模选择：观测较多时薄 SVD，小系统保留原求解；一般 R 优化按当前需求暂缓，见 [C15 补充记录](0919-c15-zero-noise.md)。 下文保留审查时的原始证据与分类。

## 范围与证据

基线为当前工作区：dev 的 e4a6aad 加尚未提交的 B01—B06 修复。本轮不修改生产代码、不提交、不发布。前一轮 1555 项通过是修复基线，不等于本次新增审查项已修复。

- 静态扫描 UQPyL 下 **169 个 Python 文件、1000 个函数**，覆盖 analysis/calibration/core/doe/inference/optimization/problem/surrogate/viz。按热点人工追踪；没有逐行审计所有 Cython/C++ 核心。
- 第一轮：检查 DOE→Problem→analysis、surrogate→optimization、ModelProblem→calibration、inference→result→storage/reader 的形状、坐标、状态、配置和输出。
- 第二轮：用真实模型、目标函数、重复运行及受控故障复现；检查用户传入参数是否真的生效。
- 第三轮：反查保存/读回、换种子和新实例对照，计数性能开销，剔除不成立的疑点，再统计风格。
- [运行探针](verification/review_global_0919.py)、[最终结果](verification/review_global_0919.json)、[静态扫描](verification/review_style_0919.py)、[扫描统计](verification/review_style_0919.json)。
- 所有运行探针使用 conda py312，OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。脚本生成证据，不修改源码。受控模拟器异常为验证失败路径，不是真实外部服务故障。
- P1：静默错误结果/数据污染或明显成本风险；P2：局部错误、使用障碍；改进和设计取舍单独分类。改动成本“小/中/大”为初步判断，不是工时承诺。

## 一、已复现的具体问题与使用障碍

| 编号 | 级别 | 问题与证据 | 源码入口 | 修复方向、成本与验收 |
|---|---|---|---|---|
| C01 | P1 | **分析入口忽略 DOE 的坐标标记。** bounds=[0,100]×[0,1]，Y=X0+X1。Saltelli 的 real 数据交给 Sobol，S1≈[0.999894,0.000100]；同 seed 的 unit 数据带 meta 交入却得到≈[0.5000,0.5015]，没有拒绝或转换。另三组 seed 重现。 | [analysis/base.py:102](../UQPyL/analysis/base.py#L102)、[base.py:208](../UQPyL/analysis/base.py#L208) | 明确 meta.output 的处理：自动解码或在不支持时入口拒绝；涉及 Y 已提供时仍需区分分析坐标，不能盲目变换。成本中；验收 real/unit 流程一致或明确报错，并覆盖混合变量。当前 DOE 文档说 unit 用于内部流程，因此这是缺少边界拦截，不把 unit 分析称作原已承诺功能。 |
| C02 | P1 | **推断返回历史与运行态共享。** 同一 MH 先运行 8 个采样点，再运行 3 个；第一次结果 decs 仍为 8 点，history 却变为第二次的 3 条，且两次 history 是同一对象。 | [inference/runtime/result.py:145](../UQPyL/inference/runtime/result.py#L145) | 返回时隔离历史与嵌套可变结果，成本小；验收复用实例、修改返回结果不会污染其他 run/运行态。 |
| C03 | P1 | **最大化推断结果的目标符号不一致。** res.objs 为内部负值，bestObjs 已恢复正值；末样本真实目标为+1.13090786，结果/SQLite snapshot/读回 artifact 均为-1.13090786。 | [inference/runtime/result.py:201](../UQPyL/inference/runtime/result.py#L201)、[storage.py:111](../UQPyL/inference/runtime/storage.py#L111) | 统一公开结果与存储边界的目标方向；保留内部 logProb 定义。成本中；验收 min/max 和即时/持久化输出逐点对齐 Problem.evaluate。 |
| C04 | P1 | **ES/IES 更新未落实问题边界。** 一维 bounds=[0,1]、初始成员0.2/0.8、观测10，真实 ES 把10传入模拟器并返回为 posteriorDecs。IES 使用同样的直接更新路径。 | [calibration/methods/es.py:49](../UQPyL/calibration/methods/es.py#L49)、[ies.py:97](../UQPyL/calibration/methods/ies.py#L97) | 必须明确受约束更新策略或拒绝不支持的有界配置；投影、变换、拒绝步会改变数值行为，不能随手 clip 后宣称公式保持不变。成本中；验收模拟器永远不收到违反已声明边界的候选，混合变量支持另行明确。 |
| C05 | P1 | **Morris 归一化指标受输出量级影响。** Y=X0+2X1 的 S1_norm=[1/3,2/3]；仅把 Y 乘1e-10便变为[0,0]。三个额外 seed 同样复现。 | [analysis/methods/morris.py:128](../UQPyL/analysis/methods/morris.py#L128) | 避免用默认绝对 isclose 判断非零总效应。成本小；验收缩放前后归一化结果不变，mu/sigma 随尺度合理变化，真正常数仍得到零指标。 |
| C06 | P2 | **MOASMO.nPop 没有生效。** 传8、24、100时内层 NSGAII 的 nPop 都为50。静态扫描也确认构造参数未被使用。 | [optimization/expensive/moasmo.py:35](../UQPyL/optimization/expensive/moasmo.py#L35) | 将参数用于默认内层优化器，明确用户提供 optimizer 时谁优先。成本小；验收默认和自定义 optimizer 两条路径。 |
| C07 | P2 | **MOASMO 自动建立的代理集合跨问题残留。** 同实例先跑2目标再跑3目标，复用旧的2模型集合；已花8次真实评价后才因输出列数报错。 | [optimization/expensive/moasmo.py:86](../UQPyL/optimization/expensive/moasmo.py#L86) | 区分自动创建与用户提供的组件，自动模型按当前问题重建；自定义不匹配入口拒绝。成本中；验收目标数变化时与新实例结果/预算一致。 |
| C08 | P2 | **MARS 改变输入维度后无法重新 fit。** 先 fit 一维，再 fit 合法二维数据，旧 basis_.num_variables 提前拒绝；新 MARS 实例对同一二维数据能成功训练预测。 | [surrogate/mars/mars.py:321](../UQPyL/surrogate/mars/mars.py#L321) | 清理旧维度状态或区分训练/预测的列数校验，成本小；验收1→2→1维重训、失败后重训。属于 B04 修复之外的重新训练路径。 |
| C09 | P2 | **通用 set/get 与真实运行配置存在双份状态。** GA(maxIters=1).set("maxIters",0) 后 get 返回0，真实 maxIter 仍为1，实际也执行1轮。 | [optimization/base.py:248](../UQPyL/optimization/base.py#L248) | 统一可配置字段的写入路径，或明确拒绝通过 set 修改不支持字段；不能存一个看似生效的值。成本中；验收 set/get、实际运行与导出配置一致。 |
| C10 | P2 | **负数代理散点图裁掉实际数据。** yTrue=yPred=[-10,-5]，默认坐标范围为[-9,-5.5]，两个端点均落在图外。 | [viz/surrogate.py:4](../UQPyL/viz/surrogate.py#L4) | 按数据跨度加边距，成本小；验收正数、负数、跨零、常数数据全部可见。 |
| C11 | P2 | **OptResult 自身缺少 run 身份信息。** 运行对象有 runId，返回结果 summary 的 run_id/method/problem_name/维度/created_at 全为 None，extra 也没有补齐。其他领域的结果已有这些信息。 | [optimization/runtime/result.py:75](../UQPyL/optimization/runtime/result.py#L75) | 返回值携带最小实验元数据，并与 reader 对齐。成本中；验收不依赖原算法对象也能关联一个结果与对应数据库/配置。 |
| C12 | P2 | **部分无效配置在调用昂贵模型后才校验。** ES 只给1个 ensemble member，模拟器已经执行1次，随后才报“至少两个成员”；R 的校验也位于首批模拟之后。 | [calibration/methods/es.py:49](../UQPyL/calibration/methods/es.py#L49)、[ies.py:97](../UQPyL/calibration/methods/ies.py#L97) | 能由输入直接判断的配置先校验；不为此增加模型调用。成本小；验收非法成员数/协方差形状在模拟器调用计数0时失败。 |

## 二、已确认的性能热点

| 编号 | 优先级 | 证据与影响 | 改进方向与验收 |
|---|---|---|---|
| C13 | 高 | **推断每步重扫、复制、解码整段历史。** 2条链、40/80/160采样点，长度大于初始批次的历史解码总行数分别1638/6478/25758。关闭打印和保存后依然发生，累计处理量近似平方增长。[InfState._collect](../UQPyL/inference/runtime/result.py#L201) | 增量统计、只处理新增样本，完整结果在需要时构建。成本中；以处理行数线性增长为验收，同时核对样本、接受率、best和历史数值完全一致。 |
| C14 | 高 | **优化默认诊断可能比搜索贵得多。** 4目标 tradeoff、NSGAII nPop=12/maxIters=2，仅36次廉价目标评价，自动HV调用3次，每次默认100万随机点；本机约99.3%耗时在HV，verbose/log/save全部关闭。[OptState._updateMulti](../UQPyL/optimization/runtime/result.py#L191)、[HV](../UQPyL/optimization/metric/hv.py#L4) | 暴露指标开关、按轮次计算频率和估计预算，保持历史对齐。成本中；关闭/降低诊断成本不应改变搜索轨迹。99.3%仅对该廉价四目标小例子成立，不外推到真实昂贵目标。 |
| C15 | 中高 | **ES/IES 用观测维度稠密矩阵，且有重复分解。** IES 3轮、固定20×20的R，eigh调用6次：每轮重复检查固定R，再分解更新矩阵；满秩更新随后还会solve。观测数10000时单个float64方阵就是800 MB，不是峰值内存实测。[ensembleGain](../UQPyL/calibration/methods/_ensemble.py#L34)、[IES](../UQPyL/calibration/methods/ies.py#L97) | 先复用固定R验证结果与分解；再评估 ensemble 空间/低秩求解，但保持奇异、半正定和显式正则化语义。成本中到大；分别做数值等价、分解次数和内存基准。 |

## 三、算法效果与使用体验的设计改进

这些是已观察到的能力边界或配置行为；**不把“可能更好”写成已证实的算法缺陷，也不直接改默认值。**

| 编号 | 观察 | 建议与验收 |
|---|---|---|
| C16 | **算法能力缺少入口声明。** 标注为单目标的 GA 接收2目标 Problem 并完成16次评价，结果也有两列；选择阶段使用按目标顺序的比较，不会告诉用户与 NSGAII 的策略差异。[GA.run](../UQPyL/optimization/soea/ga.py#L67) | 声明目标数、约束、变量类型、是否需不确定性输出等能力；拒绝不适配组合或明确提供的扩展语义。成本中；用能力矩阵驱动入口检查和示例，错误组合在评价前发现。 |
| C17 | **昂贵优化缺少约束驱动的选点。** 真实EGO：min x，约束x≥0.9，初始样本已有可行点0.9；下一次选到≈0.000355，违反量≈0.899645。代码只拟合目标，EI不使用约束；这一个例子不构成整体效果基准。[EI](../UQPyL/optimization/expensive/ego.py#L113)、[代理训练](../UQPyL/optimization/expensive/_base.py#L10) | 明确仅支持无约束，或设计可行性/约束代理选点策略。成本大；需在相同真实评价预算、多seed、不同可行域比例下比较可行解发现率和目标值。不得仅凭单例替换算法策略。 |
| C18 | **AutoTuner 默认嵌套调参的含义与成本需要更直观。** 网格l=[0.1,0.8]，joint调用似然3次、返回0.1；separate调用21次、最终返回1.0。文档已经说明separate会内部调参，因此这是模式选择与结果解释问题，不是“网格实现错误”。本例nRestartTimes=0，未使用默认4次额外重启。[AutoTuner](../UQPyL/surrogate/auto_tuner.py#L89) | 明示“精确候选”与“作为内部拟合起点/配置”的区别；报告外层候选、内部拟合后参数、全量重拟合参数及开销。成本中；先比较效果/成本，再讨论默认选择，不动已确认的Boxmin与重启约定。 |
| C19 | **调参验证策略难与用户数据结构衔接。** KFold单独存在，但AutoTuner入口固定RandSelect比例，没有传入splitter/固定验证集/分组或时间划分的入口。[split.py](../UQPyL/surrogate/split.py)、[AutoTuner._splitData](../UQPyL/surrogate/auto_tuner.py#L56) | 支持显式验证划分，再考虑多折/多seed；这会增加拟合次数，预算应可见。成本中；固定划分可复现，scaler仅拟合训练部分；不声称当前随机划分在所有数据上都产生泄漏。 |
| C20 | **失败后能查看稀疏快照，但不能读出正常结果或续跑。** 控制MH第7次评价失败，数据库正确标failed，保留迭代0/2/4；load_result报无artifact，最后快照仅存每条链最后一个点，不能据此重建完整链。[InfReader](../UQPyL/inference/runtime/reader.py#L93) | 分开设计部分结果读取与精确checkpoint。前者可先做且标记不完整，后者需要RNG、链、适应状态、archive等。成本前者中、后者大；不能把load_algorithm或最后一个种群称作精确恢复。 |
| C21 | **结果还不足以直接回答“为什么停、结果可靠吗”。** 优化结果没有明确stopReason；推断diagnostics容器存在，但本次追踪未见基础链收敛/有效样本诊断的生成路径，主要输出接受率等摘要。 | 先记录停止原因与诊断可用性，再增加按需/结束时计算的诊断。不默认每步增加高成本计算，不用单一接受率宣称已收敛。成本中；验收不同终止路径可解释，退化链输出明确状态而非误导性数值。 |

## 四、代码风格与一致性

| 编号 | 静态证据 | 建议 |
|---|---|---|
| C22 | **同层命名混用、自定义dunder。** 8处非Python协议的双下划线方法，例如__check_X_Y__、__X_transform__；内部驼峰、下划线、X_init等混用；部分toDict同时含snake_case摘要与camelCase数据字段。 | 固化现有约定：内部驼峰、导出字段snake_case、私有方法单下划线；保留X/Y等数学符号与必要第三方例外。方法/字段迁移需单独处理，不和算法改动混在一起。 |
| C23 | **机械格式缺少统一约束。** 169文件中1971行有尾随空白（含空白行）、33个分号token、61行超过120字符；构造参数缩进方式也不同。pyproject未配置统一格式/静态检查规则。 | 先确定格式规范与范围，再独立提交自动格式整理；这不是正确性缺陷。原生/移植核心按需豁免，避免格式化时夹带行为修改。 |
| C24 | **公共接口说明与配置词汇不一致。** 55处Args文档、99处:param文档；MH使用maxIters、AMH使用maxIterTimes；EGO构造器不暴露surrogate/optimizer，而ASMO/MOASMO暴露。 | 统一文档模板，说明数据形状、坐标、量纲、修改状态和返回值；整理相同含义的配置名和注入入口。成本中；用户能按同一模式切换同类算法，无需读内部字段才能配置。 |

## 排序、验收与排除项

建议顺序：
1. C01—C05：先处理静默结果错误和领域边界。
2. C06—C12：配置生效、重复使用、结果追溯和明显使用障碍。
3. C13—C15：有计数或测量证据的性能改进；先给出基准和数值等价标准。
4. C16—C21：能力边界与产品设计，分别制定最小可用方案，不一次引入大框架。
5. C22—C24：命名/格式/文档规范分开整理；纯格式可单独提前，但不遮盖功能diff。

排除/限制：
- MultiSurrogate在人工覆写整个fit且破坏失效契约时可产生新旧结果混合；用真实内置fit和非法第二列数据复核后，predict会报RuntimeError。因此不列为已确认内置模型缺陷。脚本保留正反证据。
- 不把D02“按迭代边界检查预算”、Boxmin默认、重启约定重新作为bug提出。
- B01—B06仍视为已完成；本轮C08是旧basis参与下一次训练的不同路径。
- 旧静态报告中的疑点不直接继承。静态扫描是覆盖范围说明，不等于每个算法的数学正确性已经证明。
- 无生产改动，本轮不重复运行未改源码的全量测试；报告内运行数值与性能证据来自上述专项探针。后续修复必须新增正式回归并重新跑所需测试。
