# Analysis API

DeltaTest 在有限上下界或输入的相减发生中间溢出时，使用分子/分母同时减半的等价计算恢复范围归一化。普通尺度路径不变，真实固定参数仍须匹配固定值；越界样本不裁剪。`analyze`、穷举和EA组合搜索共用此处理。

Morris 的 `mu`、`mu_star`、`sigma` 恢复输出单位后若超出浮点范围，会发出含字段、输出行和输入列的 `RuntimeWarning`，并在 `extra['morris_statistic_status']` 按“输出×输入”记录 `available` / `overflow` / `underflow`。确实溢出的数值仍为 inf，下溢归零仍为0，不能视为普通可用统计量；不以截断或填零掩盖溢出。归一化指标在恢复单位之前计算，保持可用，状态随SQLite结果保存。本诊断针对统计量恢复范围，不取消既有基本效应有限性检查。

## 指标含义与比较边界

同名 `S1` 并不表示所有方法都返回 Sobol 一阶方差贡献率：

| 方法 | 实际含义 |
|---|---|
| Sobol / FAST | `S1` 为一阶、`ST` 为总效应方差指标；Sobol 的 `S2` 为两变量交互项。常规方差分解假设独立输入，并依赖匹配的采样设计。 |
| RBDFAST | 估计一阶方差指标；当前实现将偏差修正结果裁剪到 `[0, 1]`，不输出总效应或二阶交互项。 |
| Morris | 按单位区间输入步长计算 `ΔY/(ΔX/input_range)`，效应保留输出单位。`S1_norm` 是绝对基本效应的相对权重，不是方差贡献率。 |
| RSA | 输出分位区间与补集之间的输入分布差异（平均两样本 Cramér–von Mises 统计量）；不是方差份额。 |
| DeltaTest | 删除变量后的近邻误差增量；Delta 为非自身 k 个邻居输出差平方的均值乘以 `1/2`。近邻距离使用参数范围缩放后的坐标。原始分数有输出平方的量纲，可为负；`S1_norm=S1/sum(abs(S1))` 保留符号和排序。 |
| MARS | 留出验证提示拟合质量，删除变量重拟合的正 GCV 增量 `max(0, GCV_reduced-GCV_full)`；不是解析方差分解。 |

`S1_norm` / `ST_norm` 用于相对展示，不应替代原始方差指标解释。尤其纯交互模型可有 `S1≈0` 而 `ST` 很大；不应仅凭一阶排名认定变量无影响。Morris 比较各参数在声明范围内的变化效应，结果会随参数范围的选择而变化。

分析入口统一处理采样坐标：meta.output="unit" 时，先通过 Problem.unit_to_space 解码，再评价和分析；已提供的 Y 必须对应这些真实样本，不会因坐标解码被重新评价或缩放。结果 X 保存真实坐标，meta.output="real"，并以 source_output="unit" 记录来源；不修改调用者的数组或 meta。省略 output 按 real 处理，未知标记报错，位置参数与关键字 meta 一致。Morris 的 S1_norm 使用按比例缩放的基本效应归一化，避免非零小量纲被误判为零；mu、mu_star、sigma 仍保留原量纲。

DeltaTest 的删变量敏感性分析要求至少两个输入；邻居数须为正整数且小于样本数。MARS 缺失编译扩展时可选组件不可用；其他导入/初始化错误会原样抛出。 各 Reader 的 `list_runs()` 使用 `run_id`、`created_at`、`finished_at`、`final_fes`/`final_iters`（适用时）、`db_path`、`file_name`；数据库列名及内部对象字段保持原协议。

RSA 支持二值和离散输出：分位数组内输出值相同不再使输入分布比较失效；分组及其补集各须至少有两个样本。非恒定输出没有可比较区域时发出 `RuntimeWarning`，返回零占位并标记 `insufficient_samples`，不能把这些零解释为不敏感。恒定输出正常返回零，不告警。

## `UQPyL.analysis`

`analysis` 模块评估输入变量如何影响目标或约束输出。

## 导入

```python
from UQPyL.analysis import RBDFAST, Sobol, Morris
from UQPyL.analysis.runtime import AnaReader
```

## 公共对象

| 对象 | 作用 |
|---|---|
| `Sobol` | 基于 Saltelli 样本的方差分解敏感性分析。 |
| `FAST` | 基于 FAST 设计的 Fourier amplitude sensitivity test。 |
| `RBDFAST` | 可用于普通样本矩阵的一阶敏感性分析。 |
| `Morris` | 基于 elementary effects 的筛选方法。 |
| `RSA` | Regional sensitivity analysis。 |
| `DeltaTest` | 基于近邻的变量敏感性分析。 |
| `MARS` | MARS-based 分析；可选依赖不可用时可能为 `None`。 |
| `AnaResult` | `analyze()` 返回的标准结果对象。 |
| `AnaMetric` | `AnaResult` 中的一张指标矩阵。 |
| `AnaReader` | 读取 `saveFlag=True` 保存的 sqlite 结果。 |

## 通用调用

```text
result = method.analyze(
    problem,
    X,
    Y=None,
    meta=None,
    target="objs",
    index="all",
)
```

| 参数 | 含义 |
|---|---|
| `problem` | `ProblemBase` 实例。 |
| `X` | 输入样本矩阵。 |
| `Y` | 与 `X` 对应的输出矩阵；不传时由方法内部评估。 |
| `meta` | `sampleWithMeta()` 返回的采样元数据。部分方法必需。 |
| `target` | 分析输出块，通常为 `"objs"` 或 `"cons"`。 |
| `index` | 输出列选择：`"all"`、整数或整数列表。 |

Sobol、FAST 和 RBDFAST 在计算方差或频谱功率前，对有限输出做内部中心化和缩放。
改变输出单位（包括乘以负数）后，敏感度在浮点精度范围内保持一致。各输出列独立处理，
FAST 按各轨迹块处理；结果中的原始 `Y` 保留原值。真正恒定的输出沿用返回零指标的约定，
NaN 或无穷值会抛出 `ValueError`。输入数值舍入时已经丢失的变化无法通过内部缩放恢复。

运行控制参数：

| 参数 | 含义 |
|---|---|
| `verboseFlag` | 打印简洁运行摘要。 |
| `logFlag` | 写文本日志。 |
| `saveFlag` | 保存 sqlite 结果。 |

## 方法和设计匹配

| 方法 | 需要的样本设计 | 主要指标 |
|---|---|---|
| `Sobol` | `SaltelliDesign.sampleWithMeta()` | `S1`, `S1_norm`, `ST`, `ST_norm`, 可选 `S2` |
| `FAST` | `FASTDesign.sampleWithMeta()` | `S1`, `S1_norm`, `ST`, `ST_norm` |
| `Morris` | `MorrisDesign.sampleWithMeta()` | `mu`, `mu_star`, `sigma`, `S1_norm` |
| `RBDFAST` | 普通样本矩阵；非恒定列须无重复取值，N>2M | `S1` |
| `RSA` | 普通样本矩阵即可 | `S1`, `S1_norm` |
| `DeltaTest` | 普通样本矩阵即可 | `S1`, `S1_norm` |
| `MARS` | 普通训练式样本 | `S1`, `S1_norm` |

Sobol 对真正恒定输出返回零；若混合样本有输出变化，但 A/B 基础样本的方差为零，则明确抛出 `ValueError`，提示增加基础样本量。此时不能从基础样本估计方差贡献，也不能当作总体恒定输出。

Sobol 元数据仍使用 `designType="saltelli"`。`secondOrder` 应为布尔值，`N` 应为正整数，排除布尔值 N。一阶块长度为 `problem.nInput + 2`，二阶为 `2 * problem.nInput + 2`，可选 `blockSize` 应为对应正整数，行数应为 `N * 块长度`。这些字段缺失、类型无效、相互矛盾或行数不一致时，每次运行发出一次 `RuntimeWarning`，不再因这类元数据问题抛异常。

告警后会检查候选一阶/二阶采样块中的 A/B 混合坐标复制关系。实际结构能唯一确认布局，或剩余一致字段能选定受结构支持的布局时，按恢复后的阶数和实际基础样本量继续计算；整块缺失/多出也会告警，不会删除、补齐或重排行。无法确认完整布局时，返回零占位并标记 `not_estimated`，不调用模型。提供的 Y 按输出选择保留，未提供则结果 Y 为 `None`；只有声明的 secondOrder 为布尔 True 时占位结果才包含 S2。这些零不能解释为不敏感。

`result.extra["sobol_design"]` 保存 `status`（`validated` / `recovered` / `not_estimated`）、`n_samples`、`effective_n`、`effective_block_size`、`effective_second_order`、`metrics_available`、`recovery_basis`、`issues`。原始元数据保留在 `result.meta`，settings 的 secondOrder 使用有效阶数，无法判断时为 `None`；诊断及缺失 Y 的结果可通过 SQLite / `AnaReader` 保存与读取。

元数据一致时沿用原计算路径，不验证任意提供的 Y 是否对应 X；布局可恢复也不保证截取/增加块后的采样质量或统计准确性。缺少整个 meta、错误 designType、数组形状/非有限输出及既有 A/B 零方差条件仍按原错误协议处理。

FAST 的元数据须包含正整数 `M`、`N`，满足 `N > 4*M**2`；输入须按原块顺序保留恰好 `N * problem.nInput` 行。提供 `blockSize` 时须等于 `N`。缺行、多行、块长度或元数据不一致会明确抛出 `ValueError`，不再截断或忽略额外行。

RBDFAST 要求正整数 `M`，样本量 `N > 2*M`，X/Y 均有限。固定样本列返回零；非恒定列若存在重复取值则明确拒绝，因为当前频谱排序无法可靠处理并列。这包含许多整数/离散样本，并非新增离散变量估计公式。连续随机样本的现有计算保持。

RSA 要求 X/Y 有限，NaN/inf 明确抛出 `ValueError`。输出分区仍使用线性分位数，但跨零极值插值避免减法溢出，并保留巨大与微小输出共存时的原始排序；保存的 Y 不变。合法恒定输出仍返回零统计量。

`nRegion` 须为至少 2 的整数，不接受布尔值；无效配置及空输入/输出矩阵抛出 `ValueError`。对于非恒定输出，若所有区域都无法满足“区域和补集各至少两个样本”，则每个输出发出一次 `RuntimeWarning`，建议增加样本或减少区域数，继续返回 `S1=S1_norm=0` 占位。只要还有可比较区域，就按这些区域求均值；二值输出存在空区域本身不会触发告警，也不会自动修改 `nRegion`。

`result.extra["rsa_regions"]` 保存 `n_regions`、`n_samples` 及按所选输出行排列的 `outputs` 列表。每项包含 `output_label`、`status`（`estimated` / `constant_output` / `insufficient_samples`）、`valid_region_count` 和 `region_sample_counts`，通过 `AnaReader` 读取保存结果时仍保留。`estimated` 只表示有可比较区域，不代表统计精度已经得到保证。

## `AnaResult`

| 字段或方法 | 含义 |
|---|---|
| `method` | 分析方法名。 |
| `problemName` | 问题名。 |
| `target` | 被分析的输出块。 |
| `settings` | 方法设置。 |
| `meta` | 采样元数据。 |
| `metrics` | `AnaMetric` 列表。 |
| `X`, `Y` | 记录的输入和输出矩阵。 |
| `runtime` | 运行时间。 |
| `metricNames` | 指标名列表。 |
| `getMetric(name)` / `result[name]` | 读取指定指标。 |
| `summary()` | 紧凑摘要。 |
| `toDict()` | 可序列化字典。 |

## `AnaMetric`

| 字段 | 含义 |
|---|---|
| `name` | 指标名，如 `S1`、`ST`、`mu_star`。 |
| `values` | 指标矩阵。行是输出，列是变量或变量组合。 |
| `rowLabels` | 输出标签。 |
| `colLabels` | 输入标签或输入组合标签。 |
| `colDim` | 列维度类型，如 `decsDim1` 或 `decsDim2`。 |

## `AnaReader`

用于读取 analysis sqlite 结果。

| 方法 | 含义 |
|---|---|
| `list_runs(result_dir)` | 列出结果目录中的 analysis runs。 |
| `get_run_summary()` | 读取运行摘要。 |
| `get_run_params()` | 读取运行参数。 |
| `get_metrics()` / `get_metric(name)` | 读取指标。 |
| `get_artifacts()` | 读取 `X`、`Y`、`settings`、`meta` 等 artifact。 |
| `load_problem()` | 读取保存的问题对象。 |
| `load_result()` | 重建完整 `AnaResult`。 |

## 下一步

| 目标 | 阅读 |
|---|---|
| 用户指南 | [Analysis](../analysis.md) |
| 生成兼容样本 | [DOE API](doe.md) |
| 建模协议 | [Problem API](problem.md) |


内置分析方法先将一维外部 Y 整理为列，再选择输出；统一检查 X/Y 行数及输入维度，保留所选的 problem 输出标签。导入请使用 `UQPyL.analysis` 公共入口或 `UQPyL.analysis.methods`，旧外层转发模块已删除。运行态为 `method.state`，运行参数为 `method.params`。


持久化结果带有模块标识；reader 会拒绝其他模块及没有标识的旧数据库。每次运行都有基于 UUID 的独立 ID，数据库和日志共用，即使关闭 SQLite 保存也有 ID。所有 reader 支持 `with` 和重复 `close()`。内部运行对象统一用 `state`、`params`，返回结果的正式字段不变。

## MARS 的拟合质量与配置

`MARS(maxDegree=2, maxTerms=40, minValidationR2=0.8, nValidationRepeats=1, stabilityTolerance=0.05, gcvImprovementTolerance=0.02)` 默认允许二阶交互，使用成对分段基函数。至少需要 20 行有代表性、可交换的样本；固定局部种子 0 留出 20%（至少 5 行）验证，输入和输出缩放只使用训练行。有效基函数上限会按训练样本量收紧，避免 GCV 有效参数数目达到样本数。完整模型和删变量模型均用同一训练子集拟合，验证行不参与拟合。

非恒定输出的完整代理若验证 R² 低于阈值，发出 `RuntimeWarning`，并继续返回重要性排名；恒定输出返回零分。`result.extra["mars_validation"]` 保存训练/验证数量、有效项数上限、各输出的验证 R² 和完整/删变量 GCV。删变量后 GCV 改善不会再被取绝对值变成正重要性。

输出采用二的幂次预缩放，中心和最终缩放范围只从训练行拟合；正 GCV 增量先归一化，再恢复输出平方单位。原始分数下溢为零时发出 `RuntimeWarning`，保留缩放后计算的归一化结果；原始量溢出双精度范围时明确抛出 `ValueError`。每个输出的诊断增加 `scaled_base_gcv`、`scaled_removed_gcv`、`scale_mantissa`、`scale_exponent` 和 `raw_underflow`，缩放系数表示为 `scale_mantissa * 2**scale_exponent`。恒定性在原始值上判断，避免极小/极大常数的均值舍入产生假变化；结果 `Y` 仍为原始输出。

这是开发阶段的行为变更：旧版使用所有样本及绝对 GCV 差，新版使用训练子集及正增量，所以原始数值可能变化。验证阈值是告警阈值，不是置信水平或全面正确性保证；高阶交互、噪声和分布外数据可能需要其他配置或方法。重复、分组或时序数据不能仅凭随机留出证明泛化能力。自适应基函数选择也可能受输出单位变换后的舍入影响，原始分数只满足近似尺度一致性，不保证逐位相同。

默认增加 GCV 搜索诊断：删去某个变量后的 GCV 改善若同时超过训练输出方差的 `gcvImprovementTolerance`（默认 2%）和完整模型 GCV 的一半，就发出 `RuntimeWarning`。它用于提示贪心基函数搜索可能不稳定，即使验证 R² 很高也需复核贡献。非恒定输出诊断增加 `gcv_improvement_fraction`、`gcv_improvement_variable`（零起始输入索引，无改善为 null）和 `gcv_search_unstable`；这是启发式提示，不会改写原始分数，也不能覆盖所有偏差。

设置 `nValidationRepeats=3` 会按局部种子 0、1、2 重复训练/留出。正式 `S1`、`S1_norm` 仍为种子 0 的结果。`result.extra["mars_stability"]` 保存各划分的权重、R²/GCV，以及每变量的最小/最大/平均权重、标准差（ddof=0）和最大权重范围。范围超过 `stabilityTolerance=0.05`（5 个百分点）时发出 `RuntimeWarning` 并继续返回。默认一次划分标记 `assessed=False`、`stable=None`；多次划分的 `stable` 只表示权重变化是否小，不能解读为已证明准确或置信区间。额外划分复用已有输出，不额外评价原模型；代理拟合耗时大致随重复次数增加。默认二阶和一次拟合的工作量保持。

## Morris 的效应单位

`Morris()` 统一使用标准单位区间基本效应 `ΔY/(ΔX/input_range)`。同步调整边界后，输入的正比例单位变换或平移不改变效应。仅输入步长无量纲，`mu`、`mu_star`、`sigma` 仍保留输出单位；`S1_norm` 是平均绝对效应的相对比例，不能视作 Sobol 方差贡献。

连续/整数变量采用声明的 `ub-lb`，数值离散变量采用实际取值集的最大值减最小值；范围必须有限且严格为正，无序类别没有新增数值敏感性定义。样本元数据为单位坐标时仍先正常解码，模型与结果样本继续使用真实坐标，采样设计不变。`result.extra["morris_effects"]` 保存 `effect_mode="unit"`、`effect_units="output"` 和 `input_ranges`，并随结果持久化。当前开发接口已删除 `effectMode` 参数及物理斜率选项，直接调用 `Morris()` 即可；旧物理步长计算属于历史行为。

分析至少需要两条完整轨迹，才能计算样本标准差 `sigma`；单轨迹或空样本明确抛出 `ValueError`。X/Y 差分使用带符号浮点计算数组，避免 uint8 回绕及布尔差分丢失方向；原始输入/输出数组和类型仍保留在结果中。X/Y 须为有限值。

## DeltaTest 的距离和带符号归一化

`analyze`、`findCombEA` 和 `findCombVio` 都先按参数范围缩放完整输入，再删变量或选择子集。连续/整数变量采用 `(X-lb)/(ub-lb)`；数值离散变量采用 `varSet` 实际数值的最小/最大值，因为编码上下界并不代表这些数值的范围。参数范围为零的维度映射为零，样本中不变的维度在删变量分析中贡献为零。上下界须有限且有序，固定输入须等于声明值；提供的越界样本不会被裁剪。评价和结果保存仍使用真实坐标，不修改调用者数组。

`S1` 保留正负号；`S1_norm` 除以绝对值总和，总分为负或正负抵消时不会反转排名。非零行的绝对值之和为 1，带符号之和不一定为 1；全零行仍为零。这是带符号相对分数，不能解释为方差贡献比例。非恒定输出若全部分数非正，会发出 `RuntimeWarning` 并继续返回结果；恒定输出返回零且不触发这一告警。

第 k 个距离存在并列时，将剩余邻居名额均匀分配给截止距离上的全部并列点，严格更近的邻居仍保留完整权重，自身按行身份排除。距离的相对差不超过 `8 * eps * max(1, 输入维数)` 时按舍入级并列处理，绝对容差为零。三个入口采用同样的规则，使相同行仅重新排列后保持相同的估计（允许求和舍入差异）。普通路径只查询 k+2 个近邻，并列边界逐行处理；大规模相同坐标组用输出矩汇总，不建立完整两两距离矩阵。

`analyze` 对每个输出独立中心化/缩放，先对缩放后的增量做带符号归一化，再恢复原始平方单位。原始分数下溢为零时发出 `RuntimeWarning`，归一化仍保留有效信息，因此可能出现原始分数全零而归一化非零的结果；不能将其解释为输出恒定。原始分数超出双精度上限时明确抛出 `ValueError`。`result.extra["delta_scaling"]["outputs"]` 记录 `scale_mantissa`、`scale_exponent` 和 `raw_underflow`，缩放系数为 `scale_mantissa * 2**scale_exponent`；保存的 `Y` 不变。

`findCombVio` / `findCombEA` 对输出逐列中心化后使用一个共同幅值缩放，保留原多输出权重，按行、邻居和输出列的平方差均值的一半比较组合。大常数列不会抹掉另一列的微小变化；所有输出一起正/负缩放时，组合比较在浮点精度内保持。不同非恒定输出若幅值相差过大，最小列的平方贡献仍可能低于精度边界；这里不隐式改成逐列标准化。

EA 返回的 `bestObjs` 和目标历史现在使用**缩放后的输出平方单位**，保存到数据库也采用此尺度。`result.extra["delta_selection"]` 标记 `objective_units="scaled_output_squared"`、输出汇总方式、`scale_mantissa`、`scale_exponent`、`raw_underflow` 和 `raw_overflow`。可表示时，物理目标等于缩放目标乘 `(scale_mantissa * 2**scale_exponent)**2`；极端尺度不要直接用双精度平方恢复。物理目标下溢/溢出会发出 `RuntimeWarning`，搜索仍使用有限的缩放目标。空组合无效，惩罚为 inf。穷举仍返回标签；GA 可通过 `findCombEA(..., seed=17)` 重现搜索，但不保证得到穷举最优。不同数据集的缩放目标不能忽略尺度记录直接横比。

更换输入单位时应同步更换上下界或离散数值集。这是开发阶段的行为变更，原始分数及组合搜索结果可能与旧版不同。本轮没有为无序类别引入专门距离定义，也没有证明任意高维/噪声模型下的排名可靠性。
