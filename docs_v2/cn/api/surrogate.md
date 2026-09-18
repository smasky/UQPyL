# Surrogate API

GPR/KRG 的 `nRestartTimes` 表示首次搜索之外的额外重启次数；`0` 只搜索一次。默认 `None` 对局部优化器（Boxmin/LBFGSB/MP）采用 4 次重启，即总共 5 次；EA 保留 1 次额外重启。局部优化首次从当前配置参数出发（重复拟合时包含上次拟合值），后续在参数优化坐标的边界内均匀随机采样，log 参数因此按对数空间采样。可设置 `model.rng = np.random.default_rng(42)`；相同数据、初始参数和 RNG 状态可复现。最终按返回点复算的有限训练目标选优；全部候选无效时明确报错。更多重启不保证预测误差降低。

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

`RBF.C_smooth` 为有限、非负标量，零表示不平滑。平滑项只修改训练核块的对角线，
保留多项式趋势及其约束。按当前核定义，Cubic、ThinPlateSpline、Gaussian 使用
`+C_smooth`；Linear、Multiquadric 的正值核为条件负定，因此使用 `-C_smooth`。
这与 [SciPy 的平滑方程](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html)
一致，但需注意 SciPy 对 Linear、Multiquadric 使用相反的核符号。
训练数据满足系统可解条件时，开启平滑仍保留 Cubic、ThinPlateSpline 的线性趋势，
以及 Linear、Multiquadric 的常数趋势；Gaussian 没有趋势项。平滑强度作用于预处理后的训练空间。

`GPR` 内部的 MP、EA 优化器统一最小化负对数边际似然。
`fitState["objective"]` 记录同一目标在预处理后训练数据上的值，越小越好；该值可以为负数。

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

| 方法 | 含义 |
|---|---|
| `fit(trainX)` | 拟合统计量。 |
| `transform(trainX)` | 转换数据。 |
| `fit_transform(trainX)` | 拟合并转换。 |
| `inverse_transform(trainX)` | 逆变换。 |

## 切分和指标

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
