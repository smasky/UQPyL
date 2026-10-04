# 2026-10-01 MCMC 提议尺度修复（OM02）

**OM02 已完成。** 修正 MH_Gibbs 高斯提议将方差当标准差的问题，以及 MH、AMH、MH_Gibbs 均匀提议将方差当半宽的问题；同时让 AMH 自适应协方差下限和历史不足的后备协方差按参数范围换算。新增 43 项测试，conda `py312` 推断专项 **222 passed，5.51 秒**，全量 **2338 passed，55.00 秒**，均以 `-W error` 运行、零未捕获警告。

## 修复后的尺度定义

初始配置中令 `span = ub-lb`，内部矩阵对角值仍为 `(gamma*span)^2`。

| 方法 / 提议 | 使用的尺度 | 本轮变化 |
|---|---|---|
| MH/AMH 高斯 | 完整协方差矩阵 | 保持，包含非对角相关项 |
| MH_Gibbs 高斯 | 所更新坐标的标准差 `sqrt(diag(cov))` | 改正原来直接使用方差 |
| 三种方法的均匀提议 | 各坐标半宽 `sqrt(diag(cov))` | 改正原来直接使用方差 |
| AMH 自适应下限 | `1e-3 * diag(span^2)` | 替换固定绝对值 `1e-3 * I` |
| AMH 历史不足、无当前协方差 | `sd * diag(span^2)` | 后备矩阵也按参数范围换算 |

例如参数范围为 1、`gamma=0.1`，内部对角值为 0.01。MH_Gibbs 现在使用高斯标准差 **0.1**；均匀提议取当前值左右各 **0.1**。范围乘 10 后，标准差和半宽均乘 10，转换回单位坐标后的相对步长保持一致。

NumPy 的 `normal(scale=...)` 接收标准差，见 [官方参数说明](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html)。`uniform(low, high)` 接收区间端点，见 [官方参数说明](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.uniform.html)。本项目明确选择让初始 `gamma` 控制高斯标准差或均匀半宽；均匀分布的实际方差是半宽平方除以 3，不强制与高斯方差相等，也没有额外乘 `sqrt(3)`。

AMH 在历史达到三点后继续使用经验协方差乘 `sd`，再加下限。固定参数的 `span=0`，不加入虚假的协方差下限；历史不足且已有矩阵时保留其独立副本。高斯相关性、坐标选择、随机数消费顺序、边界反射及接受判定保持既有流程。

## 验证证据

新测试文件：[test_inference_proposal_scales.py](../tests/test_inference_proposal_scales.py)。修复前独立运行为 **29 failed / 14 passed**，修复后 43 项全部通过，未放宽断言。

- 12 项固定随机数对照：三种方法、两类分布、一维/三维，各链和维度使用不同尺度；与独立 Generator 比较实际提议、下一随机数及输入不变性。
- 4 项分布检查：4096 次独立提议核对均值、高斯方差、均匀方差及支持区间。
- 12 项公共 `run` 检查：范围 0.1/1/10，warm-up 为 0/3，12 个正式样本；包括 AMH 自适应更新，单位坐标轨迹一致，评价次数、接受率、边界、目标和 logProb 正确。
- 7 项 AMH 协方差检查：独立外积参照、常数历史、两维异质范围、尺度 1e-3/1/1e3、历史不足后备矩阵及副本隔离。
- 6 项固定参数实际采样，2 项完整高斯相关协方差参照。

独立审查脚本也已更新，保存 [after-om02 JSON](verification/1001-other-modules-after-om02.json) 和 [摘要](verification/1001-other-modules-after-om02.txt)，保留初始审查及 after-om01 证据。当前共 **169 组记录、11 组异常、106 组正常控制**：16 组原 OM02 受影响记录与 8 组推断控制均正确，剩余异常均属于 OM03（8 组）和 OM04（3 组）。这里的独立记录不计入 pytest 数量。

公共入口独立复核使用 8 条链、3 次 warm-up、12 个正式样本，每次运行 **120 次评价**。范围 1/10 还原后，六种方法/分布组合的最大轨迹差为 **1.44e-15**；直接提议尺度对照最大差为 **8.33e-17**。

| 检查 | 结果 | 日志 |
|---|---|---|
| 新增 43 项修复前 | 29 failed / 14 passed，1.24 秒 | [before](verification/1001-inference-proposal-before.txt) |
| 全部推断测试 | 222 passed，5.51 秒 | [targeted](verification/1001-inference-proposal-targeted.txt) |
| 全量 py312 | 2338 passed，55.00 秒 | [full](verification/1001-inference-proposal-full.txt) |

代码位置：[mh.py](../UQPyL/inference/methods/mh.py)、[amh.py](../UQPyL/inference/methods/amh.py)、[mh_gibbs.py](../UQPyL/inference/methods/mh_gibbs.py)。中英文推断 API、测试导航、TODO 与交接记录已同步；相关文件 Ruff 静态与格式检查通过。

## 范围与后续

受影响提议的同 seed 轨迹会按正确步长改变；AMH 原固定绝对下限也已改为单位坐标中的统一下限。这是对提议参数使用及输入单位一致性的验证，不是全部后验分布、有限样本误差或自适应收敛的证明。

OM03/OM04、原 RMSE 在极小尺度下平方下溢、SA09/SA11 仍待处理；MARS 高阶贡献局限和 C15/C17/C20 暂缓状态保持。未扩展 gamma 配置校验、未改其他采样器、未重建 Python 3.14 wheel，未提交/推送/发布。
