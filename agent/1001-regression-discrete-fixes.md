# 2026-10-01 回归多输出与离散映射修复（OM03 / OM04）

2026-10-02 接口更新：用户决定所有单模型只做单输出，多输出统一由 MultiSurrogate 管理，见 [当前约定与实现](1002-surrogate-single-output.md)。下方原生 LR/PR 多输出是 10 月 1 日的阶段实现与验证证据，已被该决定替代；离散映射修复保持。

**OM03 / OM04 已修复。** 线性和多项式回归的 Origin/Ridge 预测保留样本轴和输出轴，输出 Scaler 可以逐列还原；离散辅助映射使用浮点副本，避免整数/布尔截断小数候选，也避免 float32 写回舍入破坏候选成员关系。现有 Lasso 明确限定单输出，多输出使用独立模型组成的 MultiSurrogate。

新增 **63 项测试**，正式修复前 **54 failed / 9 passed**，修复后 **63 passed，0.88 秒**；相关代理/Problem/Lasso 模块 **743 passed，7.63 秒**，最终全量 **2401 passed，56.89 秒**，均为 conda `py312`、`-W error`，零未捕获警告。独立 169 组记录 **零差异**，OM01–OM04 原有缺陷均不再复现。

## OM03：保留回归多输出维度

位置：[linear_regression.py](../UQPyL/surrogate/regression/linear_regression.py)、[polynomial_regression.py](../UQPyL/surrogate/regression/polynomial_regression.py)。

Origin/Ridge 求解器本来就按列正确拟合；错误来自预测阶段的 `reshape(-1,1)`。例如三个预测点、两个输出的矩阵原本是 `(3,2)`，被展平为 `(6,1)`；输出 Scaler 也因列数不符而失败。两处删除展平，统一由既有 `_inverseTransformY` 处理：二维预测保留形状，一维单输出补列轴，仍返回 `(nPred,1)`。

没有修改最小二乘和 Ridge 的拟合公式。验证涵盖三个输出（含常数列）、两维输入、是否拟合截距、无 Scaler/StandardScaler/MinMaxScaler、空/单点/多点预测、预处理后的 `fitModel` 和重复拟合改变输出数量。

现有 Lasso 原生路径使用一维目标和单组系数，不能直接处理多列目标。本轮在调用原生例程前检查单输出契约；公共 `fit` 与预处理入口 `fitModel` 会明确拒绝多输出并提示 MultiSurrogate。测试用原生调用哨兵确认错误输入不会进入求解器，并检查失败后不能预测旧拟合结果。单输出 Lasso 与由两个独立 Lasso 组成的 MultiSurrogate 继续正常工作，后者用解析软阈值公式核对，而非仅比较两个相同实现。

## OM04：小数离散值不能写回整数数组

位置：[space.py](../UQPyL/problem/space.py) 的 `map_discrete_vars`，`apply_var_type`、`Space.transform` 以及 Problem 委托入口共用该路径。

离散选项 `[0.25,0.75]`，旧编码输入整数 `[[0],[1],[2]]` 应得到 `[[0.25],[0.75],[0.75]]`；此前因副本仍是整数类型而全部归零。现在只在有离散列并执行映射时，把工作副本转换为浮点类型后写入选项。`[-0.1,0.3]` 等负数/非二进制整齐小数也能保留正式 float 取值，不受布尔、uint8 或 float32 输入缓冲区限制。

分箱及整数取整规则保持；两个变换标志仍独立起作用，原输入及只读数组不被改写。无离散映射时保持原来的副本路径。正式 `unit_to_space`、真实值往返及 `Problem.evaluate` 实际接收的值有独立正常对照；不能把此辅助接口缺陷描述成原默认优化/推断解码均错误。

## 测试和独立复核

| 测试 | 内容 | 项数 |
|---|---|---|
| [回归多输出](../tests/test_surrogate_regression_multioutput.py) | 独立最小二乘/岭矩阵参照、缩放与轴、预处理入口、重拟合、Lasso 契约及 MultiSurrogate 解析解 | 38 |
| [离散映射](../tests/test_problem_discrete_mapping.py) | 四种输入类型、三个辅助入口、小数/负数候选、只读副本、混合变量/标志、Problem 委托与正式转换 | 25 |

独立 [审查脚本](verification/review_other_modules.py) 已改为检查四项修复后的正常行为，保存 [after-om04 JSON](verification/1001-other-modules-after-om04.json) 和 [摘要](verification/1001-other-modules-after-om04.txt)。仍为 **169 组记录、106 组正常控制**，所有记录与独立参照一致；其中原 OM03 的 8 组输出形状/数值对照最大误差 **3.11e-15**，原 OM04 的 3 组整数映射对照逐值相等。最初检测以及 after-om01、after-om02 输出保持，独立记录不计入 pytest 数量。

- [正式修复前 54 fail / 9 pass](verification/1001-regression-discrete-before.txt)
- [修复后新增 63 项](verification/1001-regression-discrete-after.txt)
- [743 项相关回归](verification/1001-regression-discrete-targeted.txt)
- [全量回归日志](verification/1001-regression-discrete-full.txt)

中英文 surrogate/problem API、测试导航和交接记录同步；相关源文件、测试及独立脚本的 Ruff 静态/格式检查通过。

## 当前剩余范围

本次其他模块审查中的 OM01–OM04 均已处理，不等于已穷尽所有算法。极小尺度 RMSE 的平方下溢、敏感性 SA09/SA11 仍待处理；MARS 高阶贡献局限和 C15/C17/C20 暂缓状态保持。没有扩展多输出 Lasso 求解器、没有调整正式单位编码，未重建 Python 3.14 wheel，未提交/推送/发布。
