# Surrogate API

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

## 下一步

| 目标 | 阅读 |
|---|---|
| 用户指南 | [Surrogate Modeling](../surrogate.md) |
| 训练数据采样 | [DOE API](doe.md) |
| 代理辅助优化 | [Optimization API](optimization.md) |
