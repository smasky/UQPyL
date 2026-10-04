# Optimization API

多目标优化器新增 `hvFlag=True`、`hvFreq=10`、`hvSamples=10_000` 构造参数，也支持 `set/get`。初始化、每完成 hvFreq 轮和结束时记录 HV；关闭计算或未到计算轮次时，历史对应项为 None，启用且存在可行解时最终补齐最后一项。参考点仍在首次出现可行档案时固定，不受计算频率影响；档案未变化时在应记录的轮次复用缓存。HV 频率、historyFreq 与 saveFreq 相互独立，reader 保留缺失指标及对应迭代/FE 坐标。

hvSamples 控制四目标及以上的 Monte Carlo 预算，较小默认值降低诊断成本，也降低估计精度；两、三目标仍精确计算。可设 `hvFreq=1, hvSamples=1_000_000` 使用此前自动诊断频率与预算，或 `hvFlag=False` 关闭。HV 使用独立固定种子，不消耗搜索 RNG。结果 extra 保存 hv_enabled、hv_freq、hv_samples、hv_reference_point、hv_normalized，SQLite 可读回；MOASMO 默认内层 NSGAII 关闭未使用的 HV 诊断。独立调用 `HV()` 的 nSamples 默认仍为 1_000_000。

优化通用配置可在运行前通过 `set/get` 修改：`maxIters`、`maxFEs`、`maxTolerates`、`tolerate`、输出开关及频率、`historyFreq`、`hvRefPoint` 都直接对应实际运行属性；`maxIters` 对应内部 `maxIter`。读取、`exportConfig`、日志及持久化配置使用同一来源；`hvRefPoint` 赋值和读取均复制数组。

MOASMO 的 `nPop` 用于默认 NSGAII；传入自定义 optimizer 时保留其自身种群数。自动创建的代理集合每次运行重建，支持输入维度和目标数变化；传入的集合保留模型实例引用，`MultiSurrogate` 的输出数及模型列表在真实评价前校验。自定义包装器须提供 `fit`/`predict`，若暴露 `n_surrogates` 则同样支持提前检查输出数，不额外调用模型。

`OptResult.summary()` 和 `toDict()` 包含 `run_id`、`method`、`problem_name`、`n_input`、`n_output`、`n_con`、`created_at`，不依赖开启 SQLite 保存。这些元数据随每份结果独立保存，不受后续运行或 reset 影响；保存后可通过 `OptReader.load_result()` 一致读回。

代理辅助优化会用外层随机流给每次代理拟合分配子 seed，包括 MultiSurrogate 的子模型；内置模型在相同数据、初始配置和外层 seed 下可复现。短于平滑窗口的历史序列直接保留，不再发生长度错配。 各 Reader 的 `list_runs()` 使用 `run_id`、`created_at`、`finished_at`、`final_fes`/`final_iters`（适用时）、`db_path`、`file_name`；数据库列名及内部对象字段保持原协议。

MOEAD、NSGAIII、RVEA 要求至少两个目标，在评价前拒绝单目标问题。NBI 参考向量工具允许单目标，返回唯一方向 `[1]`；点数和维数须为正整数。

## `UQPyL.optimization`

`optimization` 模块运行单目标、多目标和昂贵模型优化算法。

## 导入

```python
from UQPyL.optimization.soea import GA, PSO, DE
from UQPyL.optimization.moea import NSGAII
from UQPyL.optimization.runtime import OptReader
```

## 公共对象

| 类别 | 对象 |
|---|---|
| 单目标算法 | `GA`, `PSO`, `DE`, `SCE_UA`, `ML_SCE_UA`, `CSA`, `ABC` |
| 多目标算法 | `NSGAII`, `NSGAIII`, `MOEAD`, `RVEA` |
| 昂贵优化算法 | `EGO`, `ASMO`, `MOASMO` |
| 结果对象 | `OptResult`, `OptHistory` |
| 种群对象 | `Population` |
| 保存结果 reader | `OptReader` |

## 通用调用

```text
result = algorithm.run(problem, seed=123)
```

优化算法也支持在 `run()` 时提供初始种群：

```text
result = algorithm.run(problem, initialPop=None, seed=123)
```

其中 `initialPop` 可以是：

- 形状为 `(n, nInput)` 的决策矩阵
- `Population` 对象

如果传入的初始种群还没有评价，UQPyL 会自动用真实 `Problem` 完成评价；如果成员数少于算法初始化所需数量，UQPyL 会自动补齐剩余成员。

构造函数通常接受：

| 参数 | 含义 |
|---|---|
| `nPop` | 种群规模。 |
| `maxFEs` | 迭代边界检查的评价停止阈值；初始化和已开始的轮次完整执行。 |
| `maxIters` | 已完成迭代数的上限，不含初始化；0 表示只初始化。 |
| `maxTolerates` | 单目标连续停滞的迭代数上限；None 关闭此停止条件。 |
| `tolerate` | 目标改善的绝对阈值；None 关闭停滞停止条件。 |
| `verboseFlag`, `verboseFreq` | 终端输出控制。 |
| `logFlag` | 是否写文本日志。 |
| `saveFlag`, `saveFreq` | 是否保存 sqlite 和快照间隔。 |
| `historyFreq` | 完整内存快照间隔，默认 10；None 仅保留最终快照。 |

初始化记作第 0 代，不累计停滞。每完成一代，比较更新前后的历史最优解：可行解的目标改善
必须严格大于 `tolerate` 才重置停滞计数，较小改善仍会更新最优结果。带约束时，加权违反量
下降或首次变为可行即视为进展，即使目标值变差也重置计数。因此 `maxTolerates=2` 表示连续
两代停滞后停止；启用此条件时设为 0 表示只初始化。多目标算法不使用这项单目标停滞条件。

终止检查本身不增加计数。返回结果、历史、打印和 SQLite 快照统一使用已完成的迭代数。
自定义算法在初始化后调用 `update(pop)`，每完成一代调用 `update(pop, completed=True)`。
初始化评价仍计入 `maxFEs`，即使超过阈值也完整执行；达到阈值后不再开始下一轮。包括 MOEAD、MOASMO 在内，不逐次评价检查预算，也不按剩余预算截断末轮。实际 `FEs` 可以超过 `maxFEs`；`maxIters` 独立限制完成的迭代轮数。

## 算法选择

| 任务 | 推荐起点 |
|---|---|
| 普通单目标连续优化 | `GA`, `PSO`, `DE` |
| 水文/参数校准类全局搜索 | `SCE_UA`, `ML_SCE_UA` |
| 多目标 Pareto 搜索 | `NSGAII` |
| 参考方向多目标优化 | `NSGAIII`, `RVEA`, `MOEAD` |
| 昂贵单目标优化 | `EGO`, `ASMO` |
| 昂贵多目标优化 | `MOASMO` |

## `OptResult`

| 字段 | 含义 |
|---|---|
| `bestDecs` | 最佳决策行；多目标时通常是 Pareto 决策矩阵。 |
| `bestObjs` | `bestDecs` 对应目标值。 |
| `bestCons` | 对应约束值，无约束时为 `None`。 |
| `bestFeasible` | 最佳解是否可行。 |
| `bestMetric` | 多目标或特定算法的进展指标。 |
| `FEs` | 函数评估次数。 |
| `iters` | 迭代数。 |
| `history` | 运行历史。 |
| `summary()` | 摘要字典。 |

## `Population`

`Population` 存储优化中的成员。

| 字段 | 含义 |
|---|---|
| `decs` | 决策矩阵。 |
| `objs` | 目标矩阵。 |
| `cons` | 约束矩阵。 |
| `cv` | 约束违反度。 |
| `feasible` | 可行性标记。 |

## `OptReader`

用于读取 `saveFlag=True` 保存的优化 sqlite。

| 方法 | 含义 |
|---|---|
| `list_runs(result_dir)` | 列出优化结果。 |
| `get_run_summary()` | 读取运行摘要。 |
| `get_run_params()` | 读取运行参数。 |
| `list_snapshots()` | 列出保存的快照。 |
| `load_last_snapshot_members()` | 读取最后一个快照中的成员。 |
| `load_result()` | 重建 `OptResult`。 |

## 下一步

| 目标 | 阅读 |
|---|---|
| 用户指南 | [Optimization](../optimization.md) |
| 定义优化问题 | [Problem API](problem.md) |
| 代理辅助优化 | [Surrogate API](surrogate.md) |


## 优化中的坐标约定

内置优化器在 `[0,1]^d` 内保存搜索种群并执行搜索，使用独立的 `searchLb/searchUb`；`problem.lb/ub` 仍描述原问题的边界，不会被覆盖。`initialPop` 中的决策变量按真实值解释，在入口编码一次；预评估 Population 的目标值也应使用原始目标方向。

```text
DOE(output="unit") → 单位区间种群 → 搜索/边界修复
                                  → problem.unit_to_space(U) → 真实模型评估
```

评估不会改写种群编码。`OptResult`、结果历史、日志、SQLite 和 NPZ 对外提供真实决策值及原始方向的目标值；算法运行态内部仍按最小化分数比较。运行态的历史/最优决策快照已经解码，不能直接当作搜索种群传入内部算子。

EGO、ASMO、MOASMO 在训练、预测和候选去重时统一调用 `problem.canonicalize_unit(U)`：连续维保留单位坐标；整数、离散维使用对应区间中点。训练集重复真实解只保留首次观测。其内层子问题是边界 `[0,1]` 的连续问题，代理目标已经按最小化方向处理。默认代理模型不额外缩放输入；用户显式配置的 scaler 仍由模型一致地用于训练和预测。

ASMO 的 `euclidThres` 现在是单位空间的距离阈值。有限整数/离散域若无法再找到新解，代理辅助优化可在预算用尽前停止；小型有限域会枚举剩余代表编码。其他推断算法的内部坐标并不由这一优化约定决定。

### 约束权重

`Problem(conWgt=[10, 1], nCon=2, ...)` 为每个约束指定一个有限非负权重，长度必须等于 `nCon`。`None` 表示不加权；零权重表示忽略该约束，包括其可行性判断。

```python
CV = np.sum(np.maximum(0, cons * conWgt), axis=1)
```

优化器在初始种群入口和实际评估后复制当前 Problem 的权重，覆盖初始 Population 自带的权重；子种群、选优、合并和替换保留该配置。权重应在一次运行期间保持不变。

保存的约束值仍是原始 `cons`，不预先乘权重。打印、日志和存储中的违反程度使用加权结果；结果 extra、历史快照、SQLite、NPZ 中的 `constraint_weights` 保存权重，`OptReader` 读取 Population 时恢复权重，便于继续正确选优。

### 多目标档案协议

- `bestDecs/bestObjs/bestCons`：历史可行非支配档案；无可行解时为空数组。
- `candidateDecs/candidateObjs/candidateCons`：无可行解阶段的历史最小违反度候选（最多 10 个）；可行后为 None。
- `minViolation`：无可行解时的历史最小加权违反度，可行后为 0。
- `bestMetric`：固定参考点、原始尺度的可行档案 HV；无可行解、关闭计算或未到计算轮次的中间状态为 None。
- `appearFEs/appearIters`：最后一次档案变化的批次/迭代；无可行解时跟随违反度降低。
- `hvRefPoint`：NSGAII、NSGAIII、MOEAD、RVEA、MOASMO 可选构造参数，按原始目标方向解释。
- `Population.getParetoFront()`：当前种群的可行前沿；`getInfeasibleCandidates(k=10)`：另取不可行候选。
- `OptReader.load_candidates(snapshotId)` / `load_last_candidates()`：读取单独保存的候选。

`toDict()` 的新增字段使用 `candidate_decs/candidate_objs/candidate_cons/min_violation`；NPZ 新增字段采用相同命名。详细规则见 [优化说明](../optimization.md#带约束多目标的结果与进展)。

### 高维 HV 的分块计算

`HV(popObjs, refPoint=None, normalize=True, nSamples=1_000_000, rng=None, *, batchSize=4096)` 在四个及以上目标时使用 Monte Carlo 估计。`batchSize` 是正整数，控制每批生成的采样点数；内部每次最多与 256 个解比较，避免一次生成完整的“采样点 × 解 × 目标”数组。最后不足一批的采样点也会计入。

独立 HV 函数的默认采样总量保持 1_000_000，优化器自动诊断使用上述 hvSamples 配置。相同初始状态的 NumPy 随机生成器会产生相同采样序列和 HV 估计，计算后随机状态也一致；减少 batchSize 只调整中间内存占用，不降低采样精度。小于四个目标时仍使用原有精确计算。batchSize 属于 HV 函数，与下文的 historyFreq 历史策略独立。

### 内存历史与 SQLite 保存频率

所有内置优化器支持 `historyFreq=10`，也可在运行前通过 `algorithm.set("historyFreq", value)` 调整。它控制内存中的完整种群与最优解/非支配档案快照，与 SQLite 的 `saveFreq` 相互独立。

| 配置 | 完整内存快照 |
|---|---|
| `historyFreq=10`（默认） | 初始更新、迭代编号为 10 的倍数的更新，以及最终更新 |
| `historyFreq=1` | 每次更新 |
| `historyFreq=None` | 仅最终更新 |

最终更新若已记录，不会重复追加快照。无论频率如何，`iterToFEs`、最优值、HV、非支配解数量和改善标记等轻量统计仍逐次更新记录；最终最优解和完整非支配档案不裁剪。这里的频率不会限制算法实际维护的当前非支配档案规模。

`history.populations` 与 `history.bests` 一一对应，使用 **`history.snapshotIterToFEs`** 确定各快照的 `[iteration, FEs]`；不要再用完整统计序列 `iterToFEs` 的位置索引稀疏快照。快照中的迭代编号对应实际状态更新，最终 `result.iters` 可能还计入一次终止检查。`toDict()` 导出字段为 `snapshot_iter_to_fes`，`result.extra["history_freq"]` 记录本次策略。决策值与目标方向仍按真实问题输出。

例如 `GA(historyFreq=None, saveFlag=True, saveFreq=20)` 在内存中保留每轮统计和最终快照，SQLite 按 20 轮间隔及最终结果保存完整快照。周期性 SQLite 保存不复制整段内存历史。NPZ 继续采用原有最终结果及统计曲线协议，不自动增加全部种群历史。

稀疏记录降低内存增长速度，不是固定内存上限；长任务可选择 `None`，避免完整历史快照持续累积。


预评估 `initialPop` 必须满足样本行数、目标/约束列数，且 `nCon > 0` 时必须提供 cons；缺失或维度错误在搜索前报错，完整评价数据直接复用，完全未评价的种群仍正常评价。

`OptReader.load_result()` 从已保存快照恢复真实 `OptResult`，保留缺失 HV、可行性与对应坐标；不能恢复未保存的迭代。绘图对指标和坐标成对过滤。`load_algorithm()` 恢复预算、停止条件和输出配置等简单参数；代理模型/内部优化器等组件需手动恢复并会警告，不等于恢复搜索运行态。随机种子可从运行元数据读取后显式传入新运行。耗时随运行更新，不依赖 verbose 开关。


持久化结果带有模块标识；reader 会拒绝其他模块及没有标识的旧数据库。每次运行都有基于 UUID 的独立 ID，数据库和日志共用，即使关闭 SQLite 保存也有 ID。所有 reader 支持 `with` 和重复 `close()`。内部运行对象统一用 `state`、`params`，返回结果的正式字段不变。

## 能力检查、组件与停止原因

`AlgorithmClass.getCapabilities()` 返回目标数范围、变量类型、约束处理及方差需求。
单目标算法要求恰好一个目标，多目标算法至少两个目标；不匹配在真实模型评价前报错。代理辅助算法还检查内层优化器及已声明的代理方差能力。
`EGO(..., surrogate=model, optimizer=algorithm)` 支持直接注入组件；默认仍为每实例独立的 KRG 和 GA。

`EGO/ASMO/MOASMO` 的 `constraint_handling` 为 `evaluation_only`：真实评价与结果档案处理约束，但候选生成只学习目标，没有约束代理或可行性引导。因此不承诺适用于狭小可行域，也不宣称 MOASMO 已实现约束优化。该算法扩展暂缓，见开发审查记录。

`OptResult.stopReason`、`summary()["stop_reason"]`、SQLite reader 汇总及最终日志记录停止原因：
`max_fes`、`max_iters`、`stagnation`、`user_stop`、`no_novel_candidates`、`one_step`，通用最终出口兜底为 `completed`。
若预算同时命中，沿用原有检查顺序，先 maxFEs 后 maxIters；maxFEs 仍按迭代边界检查，末轮可以超出预算。
`no_novel_candidates` 仅表示当前选点流程没有产生新点，不证明整个搜索空间已穷尽。异常仍抛出并记录失败状态，不伪装成正常停止。


## 搜索状态与数值边界

- 单目标最优记录包含每次实际评价过的候选，即使候选随后没有进入存活种群。`appearFEs` 是该候选所在评价批次完成时的计数，预评估初始成员可为 0；历史仍只在初始化和完整迭代提交时更新。
- `SCE_UA` / `ML_SCE_UA` 的 `npg=None`、`nps=None`、`nspl=None` 分别按 `2*nInput+1`、`nInput+1`、`npg` 解析。显式传入的整数会实际生效；要求 `npg>=2`、`2<=nps<=npg`、`nspl>=1`。`ngs` 控制子种群数，初始化数量为 `ngs*npg`。每次复形更新维护排序；反射重心排除最差成员，收缩向该重心移动，ML 版本再应用其最优成员混合。
- ABC 成功改进会清零失败计数，同一来源的多次失败分别累计。正的 `employedRate` 若在小种群中舍入为零，会发出 `RuntimeWarning` 后使用一个雇佣蜂；`employedRate=1` 也可运行。`nPop<2` 或非法比例仍拒绝。
- GA 的奇数种群和单成员种群保留请求的后代数量；固定坐标不作除以零的变异。DE 的二项交叉至少选一个供体坐标，`cr=0` 不再关闭所有供体坐标；供体相同仍可能没有实际位移。
- 拥挤距离、参考方向几何、GD/IGD 和极端轴尺度的 HV 使用稳定计算。GD/IGD 仍是最近欧氏距离的算术平均，不改变目标的相对单位。最终指标本身超出浮点范围时发出 `RuntimeWarning` 并返回零或无穷大，不伪造有限值；自动 HV 参考点的扩展幅度不可表示时会 warning，并将参考点限制在有限范围，建议显式指定参考点。
- NaN、复数、缺失目标/必要约束不可进入搜索结果。单目标允许最差方向的无穷大作为显式排除惩罚（最小化为 `+inf`、最大化为 `-inf`）；它不会被替换成任意有限数。多目标几何和约束值要求有限实数。代理拟合仍要求其训练数据符合代理模型的数据约定。
- `RVEA(maxFEs=None, maxIters=...)` 可按迭代进度控制角度惩罚；不能同时取消评价与迭代预算。修正搜索步骤后，同一 seed 的轨迹可能与旧实现不同，但修正后的重复运行仍可复现。
