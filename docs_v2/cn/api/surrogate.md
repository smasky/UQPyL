# Surrogate API

SVR 达到 `maxIter` 后继续返回近似模型，并发出 Python `RuntimeWarning`，不再直接输出 C++ 终端提示。`fitState["solver"]` 保存 `iterations`、`maxIterations`、`iterationLimitReached`；达到上限不代表已收敛，未达到上限也不保证预测精度。AutoTuner 的每个 `candidates[i]["fit"]["solver"]` 和 `final_refit["solver"]` 分别保存对应拟合的 `iterations`、`max_iterations`、`iteration_limit_reached`；`status="finished"` 只表示调用完成。近似候选仍按验证分数参与选择；告警不自动增加预算。用户显式将 warning 升为异常时遵循原失败失效协议。


GPR 的 `C` 是加到核矩阵对角线的观测噪声方差/正则项，不是标准差。默认初值仍为 `1e-9`；`C_attr` 默认在 `[1e-12, 1]` 按对数坐标搜索，允许近乎无噪声及含噪声拟合。`C_attr=None` 固定 C，显式 `C_attr` 优先。C 使用预处理后目标的方差单位；设置输出 Scaler 时按该尺度解释。该默认上限并不覆盖任意原始输出单位，较大或较小目标尺度可配置输出 Scaler 或自定义范围。扩大范围不等于固定加入更大噪声，也不保证所有数据的精度或不确定性校准。


GPR 的 RBF、Matern、RationalQuadratic 核默认长度尺度搜索范围为 `[0.01, 1e5]`（对数坐标）；长度对应预处理后的输入单位。非单位尺度输入建议配置输入 Scaler，或显式指定 `length_attr`，固定长度仍可用 `length_attr=None`。默认范围不是所有问题的精度保证。独立 MARS 现在默认 `max_degree=2`，允许二阶交互；显式 `max_degree=1` 保留加性模型，二阶可能增加拟合开销。

SVR 的 `fit()` 使用当前参数，不自动搜索超参数。需要调参时使用 `AutoTuner.gridTune`/`optTune`，在训练数据内部留出验证并在选择后全量重拟合；最终精度另用独立测试点评价。C、epsilon、gamma 默认是 log 参数，显式 `paraGrid` 应传 `np.log(...)`，例如 `{"C": np.log([1, 100]), "epsilon": np.log([0.001, 0.01]), "gamma": np.log([1, 10, 100])}`；这只是起始候选集，不是通用最优范围。输入和目标单位变化时同时考虑 Scaler 与参数范围。

GPR/KRG 返回的标准差是所选模型假设下的不确定性，不是对实际预测误差的保证；模型失配、数据不足或外推时可能过度自信。放宽搜索范围可消除已复现的 GPR 短尺度失配，但不代表任意数据上的区间已经校准。


MARS 支持同一实例以不同输入维度重复拟合；每次训练清理旧基函数、系数和轨迹，失败后须重新成功拟合才能使用。`plot_surrogate` 默认按真实值和预测值的联合跨度添加坐标边距，负数和常数数据也完整可见；显式 `ylim` 优先。

MARS 的 transform 接收已经预处理的输入；它与预测、评分和摘要接口都要求有效拟合状态。失效后 forward_trace/pruning_trace 返回 None。predict 支持关键字 missing，score/score_samples 会保留并传递该掩码；默认不允许缺失值，需显式 model.setting.set("allow_missing", True)。无输入预处理时支持 NaN 或等形布尔掩码；缺失输入与输入 Scaler/PolyFeature 联用明确报 NotImplementedError。score_samples 保持逐样本、逐输出的 1-(y-prediction)²/y² 定义，零目标仍遵循 NumPy 除零语义。gridTune 拒绝未知或不可调参数名，同时允许所选结构配置中才变为活跃的参数；校验不执行额外模型拟合。

AutoTuner 的 gridTune/optTune 在任何异常或中断后都会使模型失效，包括预处理、参数应用、搜索和最终全量重拟合失败；异常原样抛出，须重新成功拟合后才能预测，不回滚参数或 Scaler。MultiSurrogate 保留传入模型引用，但每个输出必须使用不同实例；构造、追加、拟合和预测均检查重复实例。MARS.predict_deriv 接收原始输入，返回原始输入/输出单位下的导数，形状为 (样本数, 所选变量数, 1)；支持 StandardScaler/MinMaxScaler，无预处理也可用，PolyFeature 或其他非仿射 Scaler 暂不支持并明确报错。

`rank_score` 逐输出计算 Kendall tau-b 后取平均；常数列计 0，至少需要两条样本。`MultiSurrogate.rng` 为各子模型分配独立随机流。AutoTuner 仅将线性代数/算术数值失败和非有限预测或评分视作候选失败，记录到 `candidateFailures`（candidate_index、error_type、message），每次调参重置；程序错误直接抛出，不再打印并吞掉。

代理模型会复制传入的输入/输出 Scaler，训练一个模型不会重新拟合另一个模型的 Scaler。公共 `fit()` 失败后模型失效，重新拟合成功前预测会报错。MSE/R²/NSE 将 `(n,)` 统一为单输出 `(n, 1)`，要求非空、有限且形状匹配，禁止隐式广播。AutoTuner 的汇总 R² 要求验证集至少两点、总离差平方和有限且非零；必要时增加 `ratio`。所有候选失败或评分非有限时明确报错，不再任取首个候选。直接调用 R²/NSE 时仍保留恒定目标产生非有限结果的原语义。

GPR/KRG 的 `nRestartTimes` 表示首次搜索之外的额外重启次数；`0` 只搜索一次。默认 `None` 对局部优化器（Boxmin/LBFGSB/MP）采用 4 次重启，即总共 5 次；EA 保留 1 次额外重启。局部优化首次从当前配置参数出发（重复拟合时包含上次拟合值），后续在参数优化坐标的边界内均匀随机采样，log 参数因此按对数空间采样。可设置 `model.rng = np.random.default_rng(42)`；相同数据、初始参数和 RNG 状态可复现。最终按返回点复算的有限训练目标选优；全部候选无效时明确报错。更多重启不保证预测误差降低。

`LBFGSB` 返回搜索过程中实际评价过的最佳有限点（含数值差分探测点），并保持点与目标值对应；这不代表求解器已收敛。其 `lastResult` 保留最后一次调用的原始 SciPy 结果，可检查 `success/status/message`，其中的 `x/fun` 可能不同于包装器返回值。目标函数须确定性；没有有限候选时明确报错。可显式传入 `LBFGSB(options={"maxls": 50})` 增加线搜索步数上限，但不保证改善所有问题。

## `UQPyL.surrogate`

`surrogate` 模块训练用于昂贵仿真、目标函数或中间响应面的预测模型。

## 导入

```python
from UQPyL.surrogate.rbf import RBF
from UQPyL.surrogate.gp import GPR
from UQPyL.surrogate.kriging import KRG
from UQPyL.surrogate import MinMaxScaler, StandardScaler, KFold, AutoTuner
```

## 公共对象

| 类别 | 对象 |
|---|---|
| 基础接口 | `SurrogateABC`, `MultiSurrogate` |
| RBF | `RBF`, `Cubic`, `Linear`, `Multiquadric`, `ThinPlateSpline`, `Gaussian` |
| Gaussian process | `GPR` |
| Kriging | `KRG` |
| Regression | `LinearRegression`, `PolynomialRegression` |
| Optional models | `MARS`, `SVR` |
| Scalers | `Scaler`, `MinMaxScaler`, `StandardScaler` |
| Splitting | `KFold`, `RandSelect` |
| Metrics | `r_square`, `nse`, `mse`, `rank_score`, `sort_score` |
| Tuning | `AutoTuner` |

`MARS` 和 `SVR` 可能依赖可选编译依赖，未安装时不可用。

## 通用拟合和预测

```text
model.fit(xTrain, yTrain)
yPred = model.predict(xPred)
```

所有单个替代模型统一只拟合一列输出：`yTrain` 接受 `(nTrain,)` 或 `(nTrain,1)`，
预测均值、不确定性为 `(nPred,1)`。多列 Y 在预处理、`fitModel`/`fitHyper` 和 AutoTuner 搜索前明确拒绝，
请使用 `MultiSurrogate` 为每列建立独立模型。AutoTuner 逐个模型调参；Scaler 和通用评分函数仍可处理多列数组。
原始 `fit`/`prepareTrainingData` 可接收向量，预处理后的 `fitModel`/`fitHyper` 使用 `(nTrain,1)`。

| 调用 | 返回 |
|---|---|
| `model.predict(X)` | 预测均值。 |
| `model.predict(X, returnStd=True)` | `(mean, std)`，仅支持 uncertainty 的模型。 |
| `model.predict(X, returnVar=True)` | `(mean, var)`，仅支持 uncertainty 的模型。 |

当前通常用 `GPR` 或 `KRG` 获取 uncertainty 输出。

## 模型

| 模型 | 主要参数 | 说明 |
|---|---|---|
| `RBF` | `kernel`, `C_smooth`, `scalers`, `polyFeature` | RBF 响应面模型。 |
| `GPR` | `kernel`, `optimizer`, `nRestartTimes`, `C` | Gaussian process regression，支持 uncertainty。 |
| `KRG` | `kernel`, `regression`, `optimizer`, `nRestartTimes` | Kriging 模型，支持 uncertainty。 |
| `LinearRegression` | `lossType`, `fitIntercept`, `C` | 线性/岭/Lasso 回归。 |
| `PolynomialRegression` | `degree`, `onlyInteraction`, `lossType` | 多项式回归。 |
| `MARS` | `max_terms`, `max_degree`, `penalty` | MARS，可选依赖。 |
| `SVR` | `symbol`, `kernel`, `C`, `epsilon`, `gamma` | Support vector regression，可选依赖。 |

Lasso 在独立工作数组上中心化，拟合时不会改写传入或模型保存的训练数据，
因此预处理后的数据可以在调参候选间复用。此行为也适用于 `PolynomialRegression(lossType="Lasso")`。

`LinearRegression` 和 `PolynomialRegression` 的 Origin/Ridge/Lasso 都遵循上述单输出约定；
多输出由 `MultiSurrogate` 统一拆列、训练和合并，不在单个模型中建立多输出路径。

### MultiSurrogate

`MultiSurrogate(m, models_list=[...])` 需要 m 个不同的模型实例与 m 列 Y，
`fit(X,Y)` 为每个模型传入 `(nTrain,1)`，成功返回容器自身。输入/输出样本数、维度和模型数在拟合前校验；
拟合失败或中断会使所有子模型失效，重新成功拟合后才能预测。

| 调用 | 返回 |
|---|---|
| `predict(X)` | `(nPred,m)` 均值 |
| `predict(X, returnStd=True)` / `returnVar=True` | 两个 `(nPred,m)` 数组；需要所有子模型支持 uncertainty |
| `predict_deriv(X, variables=None, missing=None)` | `(nPred,nVariables,m)`；需要所有子模型支持导数 |

每列按对应模型的 Scaler 还原。`supportsUncertainty` 根据所有子模型能力计算；
不确定性是各输出的边际标准差/方差，不提供跨输出协方差。子模型的预测必须为一列且样本轴一致。

标准差直接按输出单位还原，不先构造原单位方差。若请求的方差本身超出浮点范围，
发出 `RuntimeWarning`，下溢返回零、上溢返回 infinity；这时可改用 `returnStd=True`。
自定义输出 Scaler 须分别实现 `inverse_transform_std` / `inverse_transform_var` 才能提供对应结果。

`RBF.C_smooth` 为有限、非负标量，零表示不平滑。平滑项只修改训练核块的对角线，
保留多项式趋势及其约束。按当前核定义，Cubic、ThinPlateSpline、Gaussian 使用
`+C_smooth`；Linear、Multiquadric 的正值核为条件负定，因此使用 `-C_smooth`。
这与 [SciPy 的平滑方程](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html)
一致，但需注意 SciPy 对 Linear、Multiquadric 使用相反的核符号。
训练数据满足系统可解条件时，开启平滑仍保留 Cubic、ThinPlateSpline 的线性趋势，
以及 Linear、Multiquadric 的常数趋势；Gaussian 没有趋势项。平滑强度作用于预处理后的训练空间。
RBF 通常通过 LU 三角求解保留完整趋势约束。遇到重复输入或不可辨识趋势等奇异情况时，
发出 `RuntimeWarning`，改用保留多项式约束的最小二乘近似，并继续提供预测。
`fitState["linearSolve"]` 记录求解方式、退化原因、秩、相对训练残差和约束残差；
正常求解的 `method` 为 `lu`，近似求解为 `constrained_lstsq`。训练残差按预处理后的目标计算。
同一输入对应相互矛盾的观测时不保证精确插值，退化趋势也不保证唯一外推。
若连近似求解都失败或产生非有限结果，则停止拟合，不保留旧拟合状态。

Lasso 会把整数或混合 dtype 的 X/Y 复制到共同浮点工作数组，保留原始数据。
`nu-SVR` 默认 nu 搜索区间为 `[1e-5,1]`；参数及自定义搜索范围须满足 `0 < nu <= 1`。

`GPR` 内部的 MP、EA 优化器统一最小化负对数边际似然。
`fitState["objective"]` 记录同一目标在预处理后训练数据上的值，越小越好；该值可以为负数。
GPR 在预处理及直接拟合入口要求非空、有限、样本数一致的训练数据，C 为有限非负标量；
对数搜索的 C 下界必须严格为正。非法数据/参数在后端和搜索前拒绝，失败重拟合使旧预测状态失效。

`GPR`、`KRG` 默认使用 `Boxmin`。它将有限参数边界映射到内部正区间 `[1, 2]`
执行乘法搜索，评价和返回参数时恢复为传入坐标（配置 log 时就是 log 坐标）。
评价点保持在边界内，固定参数保持固定，越界初始点会被裁剪。
此映射只用于超参数搜索，不改变模型输入的尺度。
`Boxmin`、`LBFGSB` 使用局部随机数生成器，初始化不会改变 NumPy 全局随机状态。

## Scaler 和特征

| 对象 | 含义 |
|---|---|
| `MinMaxScaler(min_=0, max_=1)` | 将每个特征映射到指定范围。 |
| `StandardScaler(muX=0, sitaX=1)` | 标准化到指定均值和标准差尺度。 |
| `PolyFeature(degree=2, includeBias=False, onlyInteraction=False)` | 多项式特征扩展。 |

Scaler 方法：

StandardScaler 保留样本标准差（`ddof=1`），使用内部二进制尺度计算，避免原始平方造成下溢/溢出。
原始中心与目标中心分别保存，常数列在非默认目标中心下也能往返。

| 方法 | 含义 |
|---|---|
| `fit(trainX)` | 拟合统计量。 |
| `transform(trainX)` | 转换数据。 |
| `fit_transform(trainX)` | 拟合并转换。 |
| `inverse_transform(trainX)` | 逆变换。 |

## 切分和指标

R²/NSE 在内部安全尺度上计算平方和，保持多个输出原有的相对权重；真正恒定输出保留 NumPy warning 语义，
完全匹配时为 NaN，否则为负无穷，可由调用方 `np.errstate` 管理告警。
AutoTuner 直接检查原始输出是否恒定，极小或极大的非恒定输出不会仅因平方和超范围而被拒绝。

| 对象或函数 | 含义 |
|---|---|
| `KFold(n_splits=5)` | K 折索引切分。 |
| `RandSelect(pTest=5)` | 随机 train/test 切分，`pTest` 是测试集百分比。 |
| `r_square(true_Y, pre_Y)` | R-squared。 |
| `nse(true_Y, pre_Y)` | Nash-Sutcliffe efficiency。 |
| `mse(true_Y, pre_Y)` | Mean squared error。 |
| `rank_score(true_Y, pre_Y)` | 排序一致性指标。 |
| `sort_score(true_Y, pre_Y)` | 排序索引距离。 |

## `AutoTuner`

```text
AutoTuner(model, optimizer=None)
```

| 方法 | 含义 |
|---|---|
| `gridTune(xData, yData, paraGrid=None, ratio=10, owner=None, tuneMode="separate")` | 显式网格搜索参数。 |
| `optTune(xData, yData, paraList=None, ratio=10, owner=None, tuneMode="separate")` | 使用优化器搜索参数。 |

调参后，`AutoTuner` 会把最优参数应用到模型，并用全量数据重新拟合。

返回 `(bestParams, bestScore)`：参数为解码后的实际值，分数为搜索时验证集上的 R²，越大越好。
`tuneMode="joint"` 直接使用候选参数拟合；`"separate"` 还会执行模型内部调参。

`optTune` 在搜索开始时固定参数切片与边界，向量参数占据完整维度。
联合选择核时，同名参数的维度、边界、类型和 log 配置必须一致；不兼容时明确报错，
应统一配置或分别搜索。先应用核等结构参数，再写入有效数值参数；被选核不存在的参数跳过，
对应返回值为 `None`。

显式 `paraGrid` 沿用编码坐标：log 参数传自然对数，数值类别传分箱坐标。
向量作为一整个候选，例如二维长度尺度：

```python
paraGrid = {"l": [np.log([0.2, 3.0]), np.log([2.0, 0.2])]}
```

上述网格包含两条向量候选，不会拆成四个标量。省略 `paraGrid` 时仅评估当前参数组合一次，
不会重复对 log 参数取指数。直接调用 `applyParameterValues` 时传展开的一维编码数组；
跨核切换且候选包含暂时无效的参数时，可用 `paraInfos` 传入 `Setting.getParaInfos` 返回的固定切片。

## 下一步

| 目标 | 阅读 |
|---|---|
| 用户指南 | [Surrogate Modeling](../surrogate.md) |
| 训练数据采样 | [DOE API](doe.md) |
| 代理辅助优化 | [Optimization API](optimization.md) |


GPR/KRG 只预测均值时跳过不确定性求解。GPR 的内置核直接计算自核对角，避免创建预测点数平方的矩阵；自定义 GP 核可实现 `diag(X)`，否则使用有限分块的通用路径。GPR/KRG/RBF 共用核模板安装和参数合并流程，各核家族数学运算保持独立。未实现的 surrogate ensemble 占位已删除。

## 调参报告与自定义验证划分

`AutoTuner.gridTune()`、`optTune()` 新增仅关键字参数 `splitIndices` 和 `splitter`，两者互斥。
`splitIndices=(trainIdx, validationIdx)` 接收非空的一维整数索引，两组不能重复、重叠或越界；允许留出未参与本次选择的行，例如时间间隔。
`splitter` 可以是函数或带 `split(X)` 方法的对象，返回同样的一对索引；签名接受 `seed` 或 `rng` 时会注入独立随机流。传入 splitter 的 X 是副本。
默认仍按 `ratio` 指定的**验证集百分比**随机划分。显式划分不使用 ratio；`lastSplit` 和报告记录实际索引及种子。

当前一次调参使用一个训练/验证划分；`trainFolds, validationFolds = KFold(...).split(X)` 后，可将 `(trainFolds[0], validationFolds[0])` 作为固定划分传入，不能直接传整个多折结果，也不自动汇总多折分数。
选择期间 Scaler 只拟合训练行；选优后模型在**全部输入数据**上重拟合，包括选择阶段留出的行。若某些样本必须永久保留为外部测试集，不要传给本次调参。

`tuneMode="joint"` 按外层候选调用 `fitModel`；默认 `"separate"` 调用 `fitHyper`，候选可以作为内层优化起点，拟合后参数可能变化。
二者都保留现有返回二元组；返回的参数来自最终重拟合，分数来自选择阶段的验证集，并非最终模型在外部测试集上的分数。
`getReport()` 返回独立副本，包括 `candidates`、候选编码值、拟合前后参数、验证分数、最终重拟合参数、`fit_calls`、已跟踪的目标函数调用数、耗时及失败信息。
参数快照是实际模型值，`candidate_encoded` 则保留 Setting 的编码（例如 log 坐标）。`tracked_objective_evaluations` 只统计可挂接的 `_objfunc`，不代表所有模型内部计算量；不可跟踪的单次拟合记为 `None`。
报告不增加模型拟合或目标评价；新的调参调用重置报告。Boxmin 默认及额外 4 次重启约定不变。

```python
bestParams, score = tuner.gridTune(
    X, Y, paraGrid=paraGrid, splitIndices=(trainIdx, validationIdx),
    tuneMode="joint", seed=123,
)
report = tuner.getReport()
```
