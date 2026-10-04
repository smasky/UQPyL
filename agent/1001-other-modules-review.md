# 2026-10-01 其他模块数值审查

2026-10-02 用户统一单模型单输出，多输出由 MultiSurrogate 管理，见 [当前接口与验证](1002-surrogate-single-output.md)。OM03 现在以单模型拒绝多列 Y、容器预测数值与形状正确作为验收；独立脚本输出改为 1002-single-output 文件，保留 169 组零差异；最新 py312 全量 **2490 passed，58.58 秒，`-W error` 零未捕获警告**。下方计数和原生多输出修复描述为 10 月 1 日阶段证据。

2026-10-01 阶段更新：**OM01–OM04 已按用户要求全部修复**，见 [校准指标修复](1001-calibration-metric-scaling-fix.md)、[MCMC 提议尺度修复](1001-inference-proposal-scale-fix.md)及 [回归多输出 / 离散映射修复](1001-regression-discrete-fixes.md)。当时 py312 全量 **2401 passed，56.89 秒，`-W error` 零未捕获警告**；独立复核 **169 组零差异**。下方实验与计数为最初只检测阶段的证据；脚本现在验证四项正常行为，另写 after-om04 文件，原检测及 after-om01/after-om02 JSON/日志保持。极小尺度 RMSE、SA09/SA11、MARS 局限和既有暂缓项仍在后续范围内。

本轮按用户“看看其他模块”的要求审查，**只检测，尚未修复生产代码**。确认三个主要问题及一个范围较窄的辅助接口问题。普通单位、两输出训练和整数数组即可触发，不依赖接近浮点上限的数据。

使用 conda `py312`；独立脚本保存 **157 组记录，其中 31 组复现异常，106 组为正常控制**。其余 20 组为同一疑点在其他配置下没有异常的对照。这些是四个问题的重复验证，不是 31 个独立缺陷，也不是新增 157 项 pytest。

现有相关测试 **183 passed，2.57 秒，`-W error`**。测试通过不覆盖下面的新场景，因此不能据此宣称这些问题不存在。

## 确认的问题

| 编号 | 优先级 | 问题与影响 | 状态 |
|---|---|---|---|
| OM01 | P1 | 校准指标用固定绝对阈值判零，非恒定小量纲数据被误判，正常 NSE 等计算被拒绝 | 已修复 · 2026-10-01，65 项新增 / 全量 2295 passed |
| OM02 | P1 | MH_Gibbs 高斯提议把方差当标准差；MH、AMH、MH_Gibbs 均匀提议把方差当半宽，输入单位影响相对步长和有限次数采样 | 已修复 · 2026-10-01，含 AMH 自适应尺度，43 项新增 / 全量 2338 passed |
| OM03 | P1 | LinearRegression/PolynomialRegression 接受多输出拟合，预测却展平；输出 Scaler 路径转为形状报错 | 已修复 · 2026-10-01，38 项新增 / 全量 2401 passed |
| OM04 | P2 | 离散映射辅助接口保留整数 dtype，小数选项写回后截断，可能产生 varSet 以外的值 | 已修复 · 2026-10-01，25 项新增 / 全量 2401 passed |

### OM01：校准指标的判零与单位有关

位置：[calibration/util.py](../UQPyL/calibration/util.py)，NSE 的判零在第 62 行，另有 PBIAS、Pearson、KGE、R-factor 的相同绝对阈值判断。R² 在这里复用 NSE。

观测 `obs=[1,2,3,4]`，模拟 `sim=[1.1,1.9,3.2,3.8]`，独立公式 NSE 为 **0.98**。同时乘 `1e-5` 后观测总离差平方和为 `5e-10`，仍非零，数学 NSE 仍为 0.98，但当前实现抛出“observation variance is zero”。另一模拟行的正确 NSE 为 0.968，同样被拒绝。

在 `1e-9` 尺度，NSE/R²、PBIAS、Pearson、KGE、R-factor 六个入口均被误判。1 与 `1e-3` 的对照正确；`1e-5` 下其余四项正确。MSE/MAE/RMSE 在四个尺度均按单位正确缩放。

根因是 `np.isclose(value, 0)` 的默认绝对容差 `1e-8`。[NumPy 官方说明](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)也指出默认绝对容差不适合比较很小的数值。

修复建议：在无量纲化后进行计算和退化检查，区分真实常数/零和较小的有效量；保留真实零方差、零和/零均值的现有错误语义。这里是正常有效数据被错误拒绝，应修正计算，不能仅将错误改为 warning 后填零。

### OM02：MCMC 提议协方差和尺度混用

位置：

- [mh_gibbs.py](../UQPyL/inference/methods/mh_gibbs.py)：第 93 行构造 `(gamma*span)^2` 协方差，第 167 行将对角值直接传给 `normal`。
- [mh.py](../UQPyL/inference/methods/mh.py)：第 163 行均匀提议。
- [amh.py](../UQPyL/inference/methods/amh.py)：第 181 行均匀提议。

`normal` 的 `scale` 参数是标准差，[NumPy 官方参数定义](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html)。一维参数范围为 1、gamma=0.1 时，协方差为 0.01，高斯提议应取标准差 0.1；MH_Gibbs 实际取 0.01。范围变为 10 时实际取标准差 1，相对步长从 0.01 变为 0.1。

三种方法的均匀提议也使用协方差对角值作为半宽，所以半宽随输入范围的平方变化。独立相同随机数对照，参数范围 0.1→1→10 时，其相对步长逐次扩大十倍。MH/AMH 的高斯提议对照保持相对步长一致。

另通过公开 `run` 入口复核：同一个平坦目标、相同 seed/gamma，8 条链，两个保存样本（初始点加一次真实提议，16 次评价），仅把范围 1 换成 10。还原为单位坐标后，三种均匀提议的最大样本差均约 **0.08245**，MH_Gibbs 高斯提议约 **0.13402**；MH/AMH 高斯对照在数值容差内一致。

影响已确认到相对步长和有限迭代样本，可能导致混合变慢或过大的提议。**本轮没有证明其渐近目标分布错误**；现有反射边界也不能仅凭这个问题认定为错误。AMH 这里只复核初始提议，未将完整适应性收敛列为已验证。

修复建议：明确协方差、标准差和均匀半宽的定义，按平方根导出尺度，补公共运行的输入单位对照。均匀半宽是否与高斯协方差匹配需要在实现中明确，不能继续混用单位。

### OM03：回归预测破坏多输出形状

位置：[linear_regression.py](../UQPyL/surrogate/regression/linear_regression.py) 第 109 行与 [polynomial_regression.py](../UQPyL/surrogate/regression/polynomial_regression.py) 第 105 行 `reshape(-1,1)`。

21 行训练数据、两个输出、三个预测点，Origin 和 Ridge 均能完成拟合。没有输出 Scaler 时应返回 `(3,2)`，实际返回 `(6,1)`；输出值本身仍是正确拟合值，但样本/输出轴被混在一起。加入 StandardScaler 后，同样的拟合被接受，预测却报 “Scaler feature count does not match fitted data”。两个模型×两种损失×两种预处理共 8 组均复现。

单输出和 MultiSurrogate 容器共 16 组对照均正确，GPR 的单/多输出对照也正确。因此不能将问题扩大为整个代理模块的多输出功能错误。**未测试多输出 Lasso 原生路径**，不将其支持或拒绝语义列为已验证。

修复建议：明确每类回归模型的输出契约。允许多输出的 Origin/Ridge 应保留 `(nPred,nOutput)`；不支持的损失应在拟合入口明确拒绝，不能接受后返回错误形状。补多输出+Scaler 预测对照。

### OM04：离散映射辅助接口截断小数

位置：[space.py](../UQPyL/problem/space.py) 第 79–89 行；`apply_var_type` 和 `transform` 共用此路径。

一维编码范围 `[0,2]`、选项 `[0.25,0.75]`，输入整数数组 `[[0],[1],[2]]`。期望映射 `[[0.25],[0.75],[0.75]]`，实际得到全零，且零不在选项中。因为映射结果写回保留整数 dtype 的副本。三个入口均复现；改用同值浮点输入后均正确，原输入未被修改。

正式 `unit_to_space` 与混合变量往返对照正确，`Problem.evaluate` 接收真实值，不走这个辅助变换。因此这是仍公开的辅助接口问题，**没有证据表明优化/推断默认解码入口受到同样影响**。

修复建议：映射前使用可容纳实际选项的浮点计算副本，保持输入不变，补整数输入且小数选项的公共接口验证。

## 正常对照与审查边界

- 优化：2D/3D HV 与矩形并集容斥参照，共 30 组，包含重排、重复点、被支配点及参考点外点；GD/IGD 六组最近距离参照；NDSort 六组独立逐层支配参照。PSO/GA/NSGAII 两个 seed 共六次约束运行，检查目标重算、约束、边界及实际评价计数。**未以随机收敛到精确最优值作为正确性的证据，也未穷尽全部优化器。**
- GPR：固定 RBF 超参数，与独立矩阵求解的均值、方差、负对数边缘似然比较；单/两输出×有/无 StandardScaler 四组通过。均值/方差最大绝对差约 `4.44e-16`，似然最大约 `7.11e-15`。未验证全部核和超参数优化的全局最优性。
- 集合校准：一般协方差增益与零噪声 anomalyGain，满秩/秩亏、小量纲和有/无噪声共八组，与独立伪逆参照一致；大观测空间零噪声分支确实使用 thin SVD。没有开展一般 R 的大规模性能审查，C15 保持原暂缓状态。
- 模型接口：Problem/ModelProblem 的完整、仅目标、仅约束六组返回值和回调对照通过；混合变量 `unit_to_space(space_to_unit(X))` 包括小数离散选项，保持真实值。

敏感性已有 SA09/SA11 待处理、MARS 高阶贡献局限和 C15/C17/C20 等暂缓项保持原状态。未重建 Python 3.14 wheel，未提交、推送或发布。本轮未增加/修改 pytest，也未重复运行未变更的 2230 项全量测试。

## 证据与复现

- [独立审查脚本](verification/review_other_modules.py)
- [全部结构化记录](verification/1001-other-modules-review.json)
- [异常摘要日志](verification/1001-other-modules-review.txt)
- [183 项相关测试日志](verification/1001-other-modules-review-pytest.txt)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  /home/wmtsky/anaconda3/bin/conda run --no-capture-output -n py312 \
  python agent/verification/review_other_modules.py
```

原检测阶段脚本断言正常参照一致与四类缺陷仍能复现。四项修复后已调整断言，验证尺度参照、实际采样、回归输出及离散映射正常；新输出见上方修复记录，历史 JSON/日志不覆盖。验证范围内四项均已解决，不将正常对照或 pytest 数量理解为所有算法已获全面正确性证明。
