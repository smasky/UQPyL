# Calibration

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

`calibration` 模块通过比较仿真结果和观测数据来估计模型参数。

当你有观测数据、仿真模型、参数边界，并且希望找到能让仿真接近观测的参数时，使用校准模块。

校准方法使用 `ModelProblem`，不是普通 `Problem`。

在 UQPyL 里，校准问题的标准主链是：

```text
X -> simFunc(X) -> sim -> calibration metric / score -> parameter update or selection
```

对应到建模对象上就是：

```text
obs + simFunc + 参数边界 -> ModelProblem -> calibration.run(...) -> CalResult
```

也就是说，校准不是把普通 `Problem` 换个模块继续用，而是明确建立在 `ModelProblem` 这条仿真型问题主线上。

## 选择校准方法

| 方法 | 适合场景 | 主要输出 |
|---|---|---|
| `GLUE` | 已有候选参数，希望按阈值筛出 behavioral samples。 | `behavioralDecs`, `behavioralSims` |
| `SUFI2` | 需要 elite samples 和更新后的不确定性边界。 | `eliteDecs`, `updatedLb`, `updatedUb`, `pfactor`, `rfactor` |
| `ES` | 需要一次 ensemble smoother 更新。 | `posteriorDecs`, `posteriorSims` |
| `IES` | 需要多轮 ensemble smoother 更新。 | `posteriorDecs`, `posteriorSims`, 迭代历史 |

实践默认：已有候选样本时先用 `GLUE`；关心参数不确定性边界时用 `SUFI2`；做 ensemble smoothing 时用 `ES` 或 `IES`。

## 校准工作流

```text
obs + simFunc + 参数边界 -> ModelProblem -> calibration.run(...) -> CalResult
```

| 步骤 | 动作 |
|---|---|
| 准备观测 | `obs` 使用一维数组，shape 为 `(n_obs,)`。 |
| 定义仿真 | 写 `simFunc(X)`，支持批量参数行。 |
| 构建 `ModelProblem` | 提供 `nInput`、`lb`、`ub`、`simFunc`、`obs`，可选 `mask`。 |
| 选择方法 | 使用 `GLUE`、`SUFI2`、`ES` 或 `IES`。 |
| 读取结果 | 查看 `bestDecs`、`bestSim`、posterior、elite 或 behavioral samples。 |

## 构建 `ModelProblem`

推荐把 `ModelProblem` 理解成校准工作的标准建模容器：

- `simFunc(X)` 负责生成原始仿真输出
- `obs` 提供观测参照
- `mask` 控制哪些观测位置参与评分
- 校准方法再基于这些信息计算 metric、筛选样本或更新参数

对校准来说，`ModelProblem` 不要求必须定义 `objFunc`。只要有 `simFunc + obs`，一个 simulation-only `ModelProblem` 就已经是合法的校准容器。

这个 toy model 中，两个参数直接对应两个观测时刻：

```text
params [p1, p2] -> simulation [p1, p2]
obs = [1.0, 2.0]
```

```python
import numpy as np

from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)


obs = np.array([1.0, 2.0])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2))
    sim[:, 0] = X[:, 0]
    sim[:, 1] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, name="ToyModel")

sim = problem.simFunc([[1.0, 2.0]])

print(sim.shape)
print(problem.flattenObs())
print(problem.flattenMask())
```

| 对象 | 形状 | 含义 |
|---|---|---|
| `X` | `(n_samples, n_input)` | 候选参数行。 |
| `obs` | `(n_obs,)` | 观测数据。 |
| `simFunc(X)` | `(n_samples, n_obs)` | 每个候选参数对应的仿真结果。 |
| flattened simulation | `(n_samples, n_obs)` | 内部评分布局。 |

## 运行 GLUE

`GLUE` 会对每个候选参数评分，并保留通过阈值的 behavioral samples。

对于 `rmse` 这类越小越好的指标：

```text
score <= threshold
```

对于 `nse` 这类越大越好的指标：

```text
score >= threshold
```

```python
import numpy as np

from UQPyL.calibration import GLUE
from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)


obs = np.array([1.0, 2.0])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2))
    sim[:, 0] = X[:, 0]
    sim[:, 1] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, name="ToyModel")
X = np.array([[1.0, 2.0], [1.0, 2.4], [0.0, 0.0]])

result = GLUE(metric="rmse", verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X, threshold=0.3)

print(result.bestDecs)
print(result.bestSim)
print(result.behavioralDecs)
print(result.diagnostics["scores"])
print(result.diagnostics["behavioralMask"])
```

| 输出 | 含义 |
|---|---|
| `bestDecs` | 最优参数行。 |
| `bestSim` | 最优参数对应的仿真输出。 |
| `behavioralDecs` | 通过阈值的候选参数行。 |
| `scores` | 每个候选参数的指标值。 |
| `behavioralMask` | 每个候选参数是否通过阈值。 |

## 指标方向

| 指标 | 更好方向 |
|---|---|
| `mse` | 越小越好 |
| `mae` | 越小越好 |
| `rmse` | 越小越好 |
| `nse` | 越大越好 |
| `r2` | 越大越好 |
| `pbias` | 越小越好 |
| `pearson_r` | 越大越好 |
| `kge` | 越大越好 |

阈值方向由指标决定。不要把 `rmse` 的阈值逻辑直接套到 `nse` 上。

## 使用 Mask

`mask` 用来在评分时忽略部分观测值，形状必须与 `obs` 一致。

```python
import numpy as np

from UQPyL.problem import ModelProblem


obs = np.array([1.0, 10.0, 2.0, 20.0])
mask = np.array([False, True, False, True])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 4))
    sim[:, 0] = X[:, 0]
    sim[:, 2] = X[:, 1]
    sim[:, [1, 3]] = 999.0
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, mask=mask, name="MaskedToyModel")

print(problem.obs.shape)
print(problem.mask.shape)
print(problem.flattenMask())
```

被 mask 的位置不会参与校准评分。

## 运行 SUFI2

`SUFI2` 会选择 elite samples，并根据 elite samples 更新参数不确定性边界。

```python
import numpy as np

from UQPyL.calibration import SUFI2
from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)


obs = np.array([1.0, 2.0])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2))
    sim[:, 0] = X[:, 0]
    sim[:, 1] = X[:, 1]
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, name="ToyModel")
X = np.array([[1.0, 2.0], [1.0, 2.4], [0.0, 0.0]])

result = SUFI2(verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X, eliteSize=2)

print(result.bestDecs)
print(result.eliteDecs)
print(result.diagnostics["updatedLb"])
print(result.diagnostics["updatedUb"])
print(result.diagnostics["pfactor"], result.diagnostics["rfactor"])
```

| 输出 | 含义 |
|---|---|
| `eliteDecs` | 排名前 `eliteSize` 的参数行。 |
| `updatedLb`, `updatedUb` | 从 elite samples 得到的新边界。 |
| `pfactor` | 观测被不确定性带包住的比例。 |
| `rfactor` | 不确定性带宽度相对观测变化的大小。 |

## 运行 ES / IES

`ES` 执行一次平方根集合更新；`IES` 进行保留原始先验约束的随机迭代更新。

```python
import numpy as np

from UQPyL.calibration import ES, IES
from UQPyL.problem import ModelProblem

np.set_printoptions(precision=4, suppress=True)


obs = np.array([1.0, 2.0])


def simFunc(X):
    X = np.atleast_2d(X)
    sim = np.zeros((X.shape[0], 2))
    sim[:, 0] = X[:, 0]
    sim[:, 1] = X[:, 1] ** 2
    return sim


problem = ModelProblem(nInput=2, ub=3.0, lb=0.0, simFunc=simFunc, obs=obs, name="NonlinearToyModel")
X = np.array([[0.0, 0.5], [2.0, 1.0], [1.5, 2.0]])

esResult = ES(verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X)
iesResult = IES(maxIters=4, lam=1e-6, seed=42, verboseFlag=False, logFlag=False, saveFlag=False).run(problem, X)

print(iesResult.bestDecs)
print(iesResult.posteriorDecs.shape)
print(iesResult.posteriorSims.shape)
print(len(iesResult.history.metricsHistory))
print(np.mean(esResult.diagnostics["scores"]), np.mean(iesResult.diagnostics["scores"]))
```

用 `history.metricsHistory` 查看迭代级摘要。

## 读取 `CalResult`

| 字段 | 含义 |
|---|---|
| `bestDecs` | 当前指标下最优参数行。 |
| `bestSim` | `bestDecs` 对应仿真，按观测向量布局展平。 |
| `behavioralDecs`, `behavioralSims` | GLUE 通过阈值的样本。 |
| `eliteDecs`, `eliteSims` | SUFI2 elite samples。 |
| `posteriorDecs`, `posteriorSims` | ES 或 IES 的 posterior ensemble。 |
| `diagnostics` | 方法相关的分数、mask、边界和摘要。 |
| `history.metricsHistory` | 迭代方法的每轮摘要。 |
| `summary()` | 适合报告的紧凑字典。 |

## 常见错误

| 错误 | 修正 |
|---|---|
| 用普通 `Problem` 做校准 | 使用带 `simFunc` 和 `obs` 的 `ModelProblem`。 |
| `simFunc` 返回形状错误 | 返回 `(n_samples, n_obs)`。 |
| `obs` 是二维网格或列向量 | 按与模拟列相同的顺序显式整理成一维，如 `obsGrid.reshape(-1)`。 |
| 忘记指标方向 | 对越小越好的指标用 `<= threshold`，对越大越好的指标用 `>= threshold`。 |
| GLUE 阈值过严 | 查看 `diagnostics["scores"]` 后调整阈值。 |
| `mask` 形状不对 | 保证 `mask.shape == obs.shape`。 |
| 需要恢复时间/站点网格 | `bestSim` 为 `(1,nObs)`；使用自行保留的网格尺寸和顺序恢复。 |

## 下一步

| 目标 | 阅读 |
|---|---|
| 构建仿真问题 | [Problem](problem.md) |
| 生成候选参数集 | [Design of Experiment](doe.md) |
| 查构造参数和结果字段 | [Calibration API](api/calibration.md) |
| 对比推断工作流 | [Inference](inference.md) |
| 查看完整工作流 | [Examples](examples.md) |

### ES / IES 最优样本的指标方向

最终后验集合按指标本身的方向选出 best：RMSE/MSE/MAE 越小越好，NSE/KGE/R² 等越大越好。内部 `normalizedScore()` 对越大越好的指标取负号、对 `pbias` 取绝对值，统一供最小化比较；它不把分数缩放到 0–1。诊断、打印和保存仍使用原始指标值，指标选择不改变 ES/IES 的集合更新公式。


### PBIAS 的距零判据

使用 `metric="pbias"` 标签时，选优比较 `abs(PBIAS)`；GLUE 的阈值是非负、有限的百分数容差，例如 5 表示 `-5% <= PBIAS <= 5%`，包含边界。ES、IES、SUFI2 同样沿用距零选优。

原始公式保持 `100 * sum(sim - obs) / sum(obs)`，打印、保存及 diagnostics 保留正负号。绝对值只取在最终指标外，不对逐时误差取绝对值后求和；正负误差相抵时 PBIAS 可以为零，这不表示每个时刻都准确。上述特殊规则由字符串标签启用，直接传入自定义指标函数仍沿用默认最小化约定。


### ES / IES 的协方差退化处理

观测维数超过集合规模减一、重复观测或集合没有差异时，样本协方差可能秩不足。ES/IES 共用以下求解规则：满数值秩时使用线性求解；否则使用对称特征分解构成截断伪逆。特征值不大于 `n_valid_obs * eps * max(abs(eigenvalues))` 的方向被舍弃；零秩时增益为零，集合保持不变。伪逆不能补充集合没有表达的信息，也不保证所有观测都能拟合。

默认 R 仍为零，IES 的 lam 仍默认 0；不自动添加观测噪声或岭项。R 必须形状正确、有限、对称且半正定；浮点舍入范围内的不对称被对称化，极小负特征值截为零。IES lam 必须是有限非负标量。

ES 的 `diagnostics['covarianceSolve']` 和 IES 的逐轮 `diagnostics['covarianceSolves']` 记录 solver、rank、dimension、cutoff。一般 R 使用观测空间特征分解；默认零噪声且观测数大于成员数时使用薄 SVD。

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
