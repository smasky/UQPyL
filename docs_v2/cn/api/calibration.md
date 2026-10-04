# Calibration API

> 2.1.7 开发接口：`obs` / `mask` 为 `(nObs,)`，`simFunc(X)` 返回 `(nSamples,nObs)`。不自动展平、不接受旧网格协议；观测按数组位置对应，不需要观测标签。`CalResult` 与 SQLite 汇总以 `nObs` / `n_obs` 表示观测数，`n_output=n_obs`；移除 `nTime/nSeries/seriesLabels`。旧校准数据库不兼容，需使用新接口重新运行。

## 统一结果接口

所有方法成功运行后返回 `CalResult`。通用代码优先使用下列字段，新对外协议采用 snake_case；已有 bestDecs/bestSim 保持。

| 字段 | 含义 |
|---|---|
| `bestDecs`, `bestSim` | 最佳参数 `(1,nInput)` 与完整展平模拟 `(1,nObs)` |
| `best_score` | 最佳成员原始评分，按所选指标解释 |
| `best_index` | 最佳成员在通用 samples/simulations/scores 中的行号 |
| `samples` | 主结果参数集合 `(nSamples,nInput)` |
| `simulations` | 同一行对应的完整展平模拟 `(nSamples,nObs)`，保留被 mask 的列 |
| `scores` | 主集合逐行原始评分 `(nSamples,)`，评分时应用 mask |
| `sample_kind` | behavioral / sampling_ensemble / updated_ensemble |
| `weights` | 与主集合逐行对应的显式权重，未提供时 None，不表示零权重 |
| `intervals` | 带类型及来源的区间列表；未估计时 `[]` |
| `uncertainty` | 可选独立先验加权结果，未估计时 None |

GLUE 的主集合是通过阈值的行为样本，权重与筛选后行顺序一致；SUFI2 是最后一轮完整搜索集合，类型 sampling_ensemble；ES/IES 是更新后集合，类型 updated_ensemble，名称不承诺精确贝叶斯后验。

intervals 每项包含 kind、space（parameter/simulation）、lower、upper、probability、sample_source、indices。simulation 区间只覆盖未屏蔽的展平观测，indices 指向 simulations 的对应列。parameter 区间的 indices 为参数列。probability 是分位水平，不是已验证的覆盖率。区间来源 samples 表示主集合；uncertainty.samples 表示单独先验池。

SUFI2 独立先验加权的 samples/weights 位于 uncertainty 内，绝不会把它们的权重放到主搜索集合的 weights 上。主集合采样包络和独立后处理区间会分别列出。旧 posteriorDecs/behavioralDecs/eliteDecs 及诊断/extra 字段保留作方法特定数据；其中 SUFI2 posteriorDecs 仅为最后一轮搜索集合，通用调用应使用 samples/sample_kind。

`summary()` 和 CalReader.get_run_summary() 同步提供 best_score、best_index、sample_kind、n_samples、has_weights、interval_count。新数组/嵌套区间与运行态、旧字段分别复制；修改返回结果不回写算法，不增加模型评价。

```python
result = method.run(problem, X, **runOptions)
print(result.bestDecs, result.best_score)
print(result.sample_kind, result.samples.shape)
print(result.scores[result.best_index])
for interval in result.intervals:
    print(interval["space"], interval["kind"], interval["sample_source"])
```

校准 `rmse` 对每条模拟的残差采用二进制尺度计算，不要求中间平方能够表示，因此极小/极大但最终可表示的 RMSE 不再被误算为零或无穷大；保留 mask、多模拟行及原输入。有限输入的相减若溢出，会先在半尺度求差再恢复。只有最终结果确实无法表示、舍入为零或无穷大时，发出 `RuntimeWarning: RMSE exceeds floating-point range; underflow is returned as zero and overflow as infinity.`。MSE/MAE 的实现与语义未在此修改。


SUFI2 内部 LHS 采样保留连续/整数/离散类型。首轮离散值来自完整 `varSet`，不使用编码 lb/ub 去裁剪真实选项；后续按精英真实值的 min/max 保留原选项中的区间内值，并保持原选项顺序（不是只保留实际出现的精英取值）。整数由合法整数区间解码；范围退化为单个值时继续采样该值。`diagnostics["updatedVarSet"]` 与每轮历史记录当前离散选项，`updatedLb/updatedUb` 仍为精英真实值边界。原问题不被修改。外部 X 按真实参数域校验后原样评价，不重新映射；非法整数/离散值在模拟前拒绝。


`nse`、`r2`、`pbias`、`pearson_r`、`kge` 和 `rfactor` 按指标定义保持正比例单位换算不变：有效数据不会仅因数值很小被判为零。真实恒定观测、恒定模拟（相关系数/KGE）、零观测和（PBIAS）与零观测均值（KGE）仍按原协议报错。mask 在计算前应用，多行模拟分别返回分数。MSE/MAE/RMSE 保留其有量纲定义。

ES/IES 在首次模拟前按未屏蔽观测数量校验观测误差协方差的形状、有限性、对称性和半正定性；有效观测必须非空且有限。IES 同时提前校验运行时 `lam` 和非负整数 `maxIters`，`maxIters=0` 仅评价初始集合。协方差校验结果在迭代间复用，不增加模拟次数。

ES/IES 仅支持连续变量与箱型边界。初始 ensemble 必须有限、位于已声明边界内且至少有两个成员；整数、离散变量及一般约束会在模拟调用前明确拒绝。每次更新先按边界投影，再调用模拟器；固定维度保持固定。diagnostics["boundUpdates"] 逐次记录 adjusted_members 和 adjusted_values。投影会改变原本越界的更新，是明确的受边界限制版本；范围内更新保持原公式，不增加模型评价次数。非有限更新直接报错，不通过裁剪掩盖数值错误。

各 Reader 的 `list_runs()` 使用 `run_id`、`created_at`、`finished_at`、`final_fes`/`final_iters`（适用时）、`db_path`、`file_name`；数据库列名及内部对象字段保持原协议。

## `UQPyL.calibration`

`calibration` 模块通过比较仿真和观测来估计模型参数。所有校准方法都使用 `ModelProblem`。

## 导入

```python
from UQPyL.calibration import GLUE, SUFI2, ES, IES, CalReader
```

## 公共对象

| 对象 | 作用 |
|---|---|
| `GLUE` | Generalized Likelihood Uncertainty Estimation。 |
| `SUFI2` | Sequential Uncertainty Fitting。 |
| `ES` | Ensemble Smoother。 |
| `IES` | Iterative Ensemble Smoother。 |
| `CalResult` | 校准结果对象。 |
| `CalHistory` | 校准历史。 |
| `CalReader` | 读取保存的校准 sqlite。 |

## 通用调用

```text
result = method.run(modelProblem, X=None, seed=123)
```

| 参数 | 含义 |
|---|---|
| `modelProblem` | `ModelProblem`。 |
| `X` | 候选参数矩阵；部分方法可内部生成。 |
| `seed` | 随机种子。 |

运行控制参数：

| 参数 | 含义 |
|---|---|
| `verboseFlag` | 打印运行摘要。 |
| `logFlag` | 写日志。 |
| `saveFlag` | 保存 sqlite 结果。 |

## 方法

| 方法 | 关键参数 | 主要输出 |
|---|---|---|
| `GLUE` | `metric`, `threshold` | `behavioralDecs`, `behavioralSims` |
| `SUFI2` | `eliteSize`, `nSamples`, `maxIters` | `eliteDecs`, `updatedLb`, `updatedUb`, `pfactor`, `rfactor` |
| `ES` | ensemble 参数矩阵 `X` | `posteriorDecs`, `posteriorSims` |
| `IES` | `maxIters`, `lam`, `seed` | `posteriorDecs`, `posteriorSims`, history |

## `CalResult`

每次返回的结果都是独立快照，复用或重置算法不会改变旧结果。结果出口对 `history`、
`diagnostics`、`extra` 和 `settings` 做深复制，包括其中嵌套的列表、字典及数组。
修改这些结果字段也不会影响算法的运行状态或配置。复制过程不增加模型评价次数。

| 字段 | 含义 |
|---|---|
| `method` | 校准方法名。 |
| `bestDecs` | 最优参数行。 |
| `bestSim` | 最优参数对应的仿真。 |
| `behavioralDecs`, `behavioralSims` | GLUE 接受的样本。 |
| `eliteDecs`, `eliteSims` | SUFI2 elite samples。 |
| `posteriorDecs`, `posteriorSims` | ES/IES posterior ensemble。 |
| `diagnostics` | 分数、mask、边界、pfactor/rfactor 等方法特定信息。 |
| `history` | 迭代历史。 |
| `summary()` | 摘要字典。 |

## 指标方向

| 指标 | 更好方向 |
|---|---|
| `mse`, `mae`, `rmse`, `pbias` | 越小越好 |
| `nse`, `r2`, `pearson_r`, `kge` | 越大越好 |

GLUE 的阈值判断会依据指标方向执行。

## `CalReader`

| 方法 | 含义 |
|---|---|
| `list_runs(result_dir)` | 列出校准结果。 |
| `get_run_summary()` | 读取摘要。 |
| `get_run_params()` | 读取参数。 |
| `load_problem()` | 读取保存的 `ModelProblem`。 |
| `load_result()` | 重建 `CalResult`。 |

## 下一步

| 目标 | 阅读 |
|---|---|
| 用户指南 | [Calibration](../calibration.md) |
| 仿真问题 | [Problem API](problem.md) |
| 候选参数采样 | [DOE API](doe.md) |


ES 分别评价初始与更新后集合各一批；IES 在迭代间传递完整模拟，只需初始一批加每轮一批；SUFI2 从已评价样本中提取 elite 模拟。mask 在同一批完整模拟上选择有效观测。SQLite 保存一份 result 和一个小 summary，不再重复保存各个大数组；完整字段通过 `load_result()` 访问，`get_run_summary()` 只读取小摘要。


持久化结果带有模块标识；reader 会拒绝其他模块及没有标识的旧数据库。每次运行都有基于 UUID 的独立 ID，数据库和日志共用，即使关闭 SQLite 保存也有 ID。所有 reader 支持 `with` 和重复 `close()`。内部运行对象统一用 `state`、`params`，返回结果的正式字段不变。

## 能力与协方差求解

`getCapabilities()` 区分 ES/IES 的连续箱约束与 GLUE/SUFI2 的一般约束未参与处理状态。
ES/IES 默认 `r=None` 表示零观测误差，当有效观测数多于集合成员数时，对缩放后的集合异常矩阵做薄 SVD，不再构造零 R 或 Cyy；诊断 `solver="svd"`。观测数不超过成员数时，保留较小的稠密求解，避免小问题额外开销。显式传入 R 时，仍在校验数值秩后复用特征分解计算增益，满秩为 `eigh`，秩不足为 `pinv`。ES 对全零 R 使用零噪声投影路径；IES 显式 R 沿用通用路径。
固定 R 每次运行只验证一次；IES 另计算一次观测扰动的平方根。lam 的新含义见下方更新方程说明。
默认零噪声增益仍采用观测维度的特征值截断阈值；lam 不再抬高观测零空间的秩。显式一般 R 的稠密优化仍暂缓。

### ES / IES 更新方程与不确定性

ES 使用确定性的对称平方根更新：均值按 Kalman 增益更新，中心化集合经平方根变换。在线性模型、没有边界裁剪时，更新后的样本均值和协方差符合以初始集合矩为先验的高斯条件公式。非线性仍属于集合线性化近似。

IES 使用保留原始先验的随机 Gauss–Newton EnRML。新增 `seed=None`；设置整数可复现，同一次运行的扰动观测只生成一次，迭代中不重新抽样。`lam=0.0` 为完整 GN 步；正值是无量纲、固定的先验度量阻尼，使 Hessian 为 `(1+lam)*C_prior^-1 + H.T@R^-1@H`（实际用增益形式计算，不要求先验可逆）。**lam 不再是旧版 `Cyy+R+lam*I` 的观测空间 ridge**，固定步模式没有自动调节阻尼或接受/拒绝机制；可选回溯见下文。原始先验协方差与成员在所有迭代中保留；多轮更新不会把同一批观测当作独立的新数据反复同化。

IES 的有限随机集合不保证单次样本矩严格等于总体后验矩；非线性不保证得到真实后验。零/奇异 R 为伪逆扩展，无法分辨的回归方向保留前次斜率，避免硬观测导致集合收缩后重新退回先验；这不保证不一致硬观测能被满足。参数回归按初始集合的各列尺度计算。`diagnostics["regressionRanks"]` 记录各轮可分辨秩，`updateMethod` 区分两种更新，IES 另记录 `damping` 与 `seed`。任何边界裁剪均会改变无约束公式的统计性质。

`maxIters` 是固定迭代预算，不是收敛保证；评分指标不参与更新方程。同种子比较指标时，更新集合相同。ES 仍评价两批，IES 仍评价初始一批加每轮一批。

### 范围保护、加权区间和可选回溯

SUFI2 默认 `explorationFraction=0.1`、`minRangeFraction=0.05`。内部采样首轮仍覆盖原始域；之后每轮 `ceil(nSamples*explorationFraction)` 个样本来自原始合法域，其余来自精英包络附近。非离散参数的局部采样宽度不小于原始宽度的指定比例；原本固定的参数仍固定。整数/离散合法性保持，探索样本可重新引入此前未入选的合法离散值。这些保护减少永久排除区域的风险，不保证全局最优，也不是重新实现完整文献 SUFI-2。两个参数设为 0 可选择纯精英包络收缩。

`updatedLb/updatedUb` 继续表示精英真实包络；history 新增 `samplingLb/samplingUb`（该轮局部采样范围）和 `explorationCount`。外部提供 X 时不重采样。`maxIters=0` 会 RuntimeWarning 后执行一轮筛选；负数、非整数、非法样本数/精英数仍在模拟前拒绝。

GLUE 新增 `run(..., logLikelihood=None, interval=0.95)`。可选回调签名为 `logLikelihood(obs, sim, mask=mask)`，输入完整展平观测、**行为样本**模拟、展平 mask，返回每个行为样本的 log 权重。回调负责使用 mask 和定义噪声模型；允许 -inf 表示零权重，不接受 NaN/+inf/全部零质量。内部减去最大 log 权重再归一化。未提供时行为样本等权，明确记录 `weighting="uniform"`，不把评分自动解释为概率。似然加权的统计解释要求候选采样分布及权重符合使用者的先验/提议设定；不能自动校正任意候选集合的采样偏差。

GLUE 的 diagnostics 新增 `behavioralWeights`、`effectiveSampleSize=1/sum(w**2)`、`interval`、`ppuLower/ppuUpper`。区间逐个未屏蔽输出按加权经验 CDF 的逆计算（阶梯分位数，不做线性插值），表示行为样本模拟的加权范围；未额外加入未来观测噪声，不承诺名义覆盖率。best 样本仍按配置的评分选取。

IES 新增 **可选** `adaptive=True`（默认 False）、`tolerance=1e-6`、`maxBacktracks=8`。开启后用固定原始先验和固定扰动观测的 RML 目标检查实际模拟结果；全步恶化则逐次减半，最多尝试 1+maxBacktracks 步。没有可接受步时 RuntimeWarning，保留上一接受集合并结束。零/奇异 R 下，硬观测残差优先于软观测与先验代价；这是退化扩展。越出初始先验仿射支撑的裁剪候选不当作零惩罚接受。`lineSearch` 记录步长、接受状态及 [硬残差, 先验+软残差] 目标；`stopReason` 区分 `step_tolerance`、`line_search_stalled`、`iteration_budget`。步长很小或预算耗尽均不证明达到真实后验或全局最优。回溯可能增加模拟批次，默认关闭时保留固定步方程和原调用预算。

ES/IES 新增 `boundHandling="rescale"`，默认仍是 `"clip"`。rescale 从当前成员沿拟议方向缩短整条步长，越界时采用最大可行步的 99%，减少直接贴边堆积；已在边界且向外的成员仍可能停住。`boundEffects` 记录调整比例、调整前后均值/极差以及 `unconstrained_moments_preserved`。这只是边界处理选择，不是精确截断高斯采样，两种模式都可能改变统计矩。

### 精度修正：区间来源与局部导数

SUFI2 现在从该轮**完整采样集合**计算 `ppuLower/ppuUpper`、P-factor/R-factor，原精英输出分位数另存 `elitePpuLower/elitePpuUpper`。`intervalKind="sampling_envelope"` 明确这是当前采样分布的输出包络，包含搜索/探索策略的影响，不能解释为参数可信区间。内部重采样时，后续每轮先保留一个当前最佳成员，防止丢失历史好解；其余名额分给局部和全局探索，合计仍为 nSamples。history 新增 `retainedCount` 和原量纲 `bestScore`。nSamples=1 时只能保留 incumbent，无法同时探索。

SUFI2 `run` 新增 `logLikelihood=None, uncertaintyX=None, uncertaintySamples=2048, interval=0.95`，用于**独立的先验重要性加权后处理**，不改变优化筛选结果。未提供似然时 `uncertaintyStatus="not_estimated"`，不会凭精英范围构造可信区间。提供似然后，uncertaintyX 应为未按这次观测筛选过的先验样本；省略时使用独立随机流在原始合法域采 uncertaintySamples 个 LHS 样本，隐含原始域均匀先验。回调接收完整展平观测/模拟和 mask，返回每个独立先验样本的 log 似然，接口同 GLUE。非均匀提议需由用户在 log 权重中正确加入先验/提议比，不能把收缩集合直接当先验。

结果在 `extra["uncertainty"]`：`method="prior_importance_weighting"`、`prior_source`、`samples`、`weights`、`parameter_mean/variance/lower/upper`、`simulation_lower/upper`、`interval`、`effective_sample_size`。参数区间与模拟函数区间分开，模拟区间不包含额外未来观测噪声。有效样本数低于 20 时 RuntimeWarning，状态为 low_effective_sample_size；20 是诊断阈值而非精度保证。新增一次独立样本池模拟，需要计入成本。原始域采样不是完整文献 SUFI-2 的贝叶斯实现，而是明确单独命名的后处理方法。

IES 新增 **`localLinearization=True`**（默认 False）。每个成员分别用有限差分估计模型导数，而非全部成员共用一个回归斜率；仍保留原始先验和固定扰动观测。配合 `adaptive=True` 可以检验实际 RML 目标下降。每轮最多额外 2×非固定参数个数的批量模拟，边界处采用截短/单侧差分，固定参数不扰动。`linearization="member_finite_difference"`、`derivativeBatches` 记录方式和额外批次数；每个成员的增益求解诊断保存在 covarianceSolves 的 members 内，regressionRanks 对本模式为 None。

该模式更接近逐成员随机 MAP 优化，代价更高，适用于可计算且局部平滑的模拟器；不自动切换默认算法，不保证精确非线性后验、多峰质量或不可微模型的精度。差分步无法在浮点数中表示时会明确失败，避免把无效导数当作零。

### 误差指标的浮点范围与低 ESS 提示

MSE、MAE、RMSE 使用按行二进制缩放残差，避免中间相减、平方或求和溢出导致本可表示的最终误差变成 inf。mask、批量形状和指标物理单位不变；例如 MSE `[1.4e154,0]` 对零观测为约 `9.8e307`，MAE `[1e308,1e308]` 对 `[-1e308,1e308]` 为 `1e308`。真正超出浮点范围的最终值仍返回 inf 或下溢为零，并发出对应指标的 RuntimeWarning，不通过填有限值掩盖范围限制。

GLUE 记录 diagnostics["uncertaintyStatus"]。ESS<20 时为 low_effective_sample_size，否则为 estimated；后者不保证区间准确。显式提供 logLikelihood 且 ESS<20 时发出 `GLUE uncertainty effective sample size is below 20; weighted intervals may be unreliable.`，仍返回原权重、分位数及结果，不改成等权、不人为扩大区间。未提供似然的等权筛选也记录小 ESS 状态，但不额外发出似然加权告警。ESS 阈值采用浮点容差，避免 20 个等权样本因舍入被误报；SUFI2 的同类检查同步使用该容差。
