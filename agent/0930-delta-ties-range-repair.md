# 2026-09-30 DeltaTest 等距近邻与输出尺度修复

用户授权修复两项已复现问题：同样本仅换行顺序会改变 DeltaTest 结果，以及极小输出的平方下溢使归一化全零。本轮同时修复 MARS 的同类归一化问题及极小恒定输出误判，保留用户确认的默认二阶、R²<0.8 warning 并继续返回约定。环境 conda py312，未提交/推送。

## 最终行为

- DeltaTest 在第 k 个距离存在并列时，把剩余 k-m 个名额均匀分配给所有截止并列点；严格更近的 m 个邻居保留完整权重，自身按行身份排除。相同成对样本重新排列后结果一致，允许浮点求和舍入差异。
- DeltaTest `analyze` 和 MARS 都在安全缩放后的输出上计算增量、先归一化，再恢复原始平方单位。输出×1e-200 后，原始分数可以因双精度限制为零，但归一化保留信息，并给出明确 underflow warning。
- 真正超过双精度范围的原始分数明确抛出 `ValueError`；不返回无穷，也不泄露 Python `OverflowError`。能够表示的最终值不会因为预先形成 scale² 的中间上/下溢而丢失。
- 极小/极大有限恒定输出返回零，MARS 在原始值上判断恒定性，避免均值舍入产生假变化。NaN/正负无穷输出明确拒绝。
- 输入/输出结果数组保留真实原值，各输出独立缩放；MARS 缩放仍只从训练行拟合，没有把留出输出用于拟合预处理。

## 等距规则与资源使用

每行平均平方误差定义为：

`[sum(closer errors) + (k-m) * mean(cutoff tied errors)] / k`。

最后对行平均再乘 1/2。该规则是对合法截止并列选择的均匀期望，并非声称原论文已规定这一扩展。不能简单平均所有截止内邻居，否则会改变严格更近点与并列组之间的权重。

距离比较使用相对舍入容差 `8 * eps * max(1, nInput)`，绝对容差为零；真实相差 1e-8 的邻居没有被合并。普通路径只查询 `min(N, k+2)` 个点以探测截止额外并列。仅并列行查询截止候选，一次只处理一行；大量相同坐标采用组内均值和方差汇总，避免为每个成员构造完整并列候选集，没有引入全体 N×N 距离矩阵。KDTree 查询本身仍受数据分布和维数影响，不承诺整体始终线性。

`analyze`、`findCombEA`、`findCombVio` 共用这一近邻规则。当前输出归一化尺度修复针对 `analyze`，组合搜索的目标仍沿用原始输出平方单位，未扩大为极端尺度下的组合优化协议变更。

## 输出缩放与诊断

扩展既有 `_variance.scaleOutput`，默认调用保持 Sobol/FAST/RBDFAST 的原有行为；内部可选返回 `(spread, exponent)`，表示物理输出系数 `spread * 2**exponent`。幂次预缩放保护减法、均值与平方运算，恢复时同时拆分分数的尾数/指数，避免直接计算 scale²。

MARS 的训练路径继续采用均值中心化、最大绝对偏差缩放；72 组加性/二阶/三阶、128/512 样本、三个种子、输出系数 1/1e-6/1e-200/1e150 的对照中，新旧训练缩放逐值相同。见 [缩放保持记录](verification/0930-mars-scaling-preservation.json)。特殊恒定值修复及范围诊断属于明确的边界行为变化，不将其描述为逐位保持所有输入。

DeltaTest 的 `result.extra["delta_scaling"]["outputs"]` 保存 `scale_mantissa`、`scale_exponent`、`raw_underflow`。MARS 原诊断增加相同字段及 `scaled_base_gcv`、`scaled_removed_gcv`，原始单位 GCV 下溢时仍可检查拟合证据。原始分数全零、归一化非零时，以缩放后的增量解释归一化，不能再从已经舍入为零的原始矩阵重算。

## 回归与独立验证

新增 **43 项** 回归：

- [Delta 等距回归](../tests/test_analysis_delta_ties.py)：16 项。手算截止平均、严格近点保权重、网格/重复/投影及行排列、全重复多输出、自身排除、近而不等的距离、重复组资源路径及 EA/Vio 公共入口。
- [输出范围回归](../tests/test_analysis_output_range.py)：27 项。两方法正负极小输出、次正规数、极大但可表示输出、受控范围错误、逐列尺度和常数、极大有限常数、非有限拒绝、训练/留出隔离，以及中间 scale² 不可表示但最终值可表示的独立 Decimal 对照。

原带符号归一化的受控 Delta 测试改用单位输出跨度，使受控数值仍对应原始单位；符号、抵消、排序断言保持，并由独立距离/范围用例检验实际计算。

修复前首批 29 项为 **22 failed / 7 passed**，见 [失败证据](verification/0930-delta-ties-range-before.txt)。首轮相关 79 项通过；进一步检查及最终代码完成后，全量 **2099 passed，58.44 秒，零未捕获警告**（`-W error`），见 [最终全量日志](verification/0930-delta-ties-range-full.txt)。Ruff 静态/格式与 `git diff --check` 通过。

最终独立复核 **250 组**，不与 pytest 项数相加：

- 网格/重复/连续样本 × k=1/2/5 × 十次行排列，90 组，全部与逐对距离加权参照一致，最大绝对误差约 1.11e-16；包含之前会翻排名的重复网格案例。
- 两方法 × 八种输出尺度，16 组：有限原始范围内归一化与普通输出一致；极小尺度明确下溢提示，极大不可表示尺度明确范围错误。
- 3/10 维 × 128/512/2048 行 × 三个采样种子 × 两种噪声 × k=1/2/5/10，144 组线性筛选复核，全部找对活跃前两名。

见 [独立脚本](verification/check_delta_ties_repair.py)、[记录与耗时](verification/0930-delta-ties-range-repair.json)、[日志](verification/0930-delta-ties-range-repair.txt)。旧失败审计文件保留，不覆盖历史证据。

耗时基准包含连续与全重复 128/2048 行、k=2，每种重复三次取中位数。普通连续样本有额外近邻查询和边界检测开销，重复样本还需并列均匀期望计算；旧版任意选择的分数只作计时基线，不作为正确性参照。具体毫秒数与比值见当前 JSON，不将单机微基准推广为任意分析任务的速度承诺。

## 复现与范围

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run --no-capture-output -n py312 python -m pytest -q -W error
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run --no-capture-output -n py312 python agent/verification/check_delta_ties_repair.py
```

同步更新中英文 analysis API 与测试导航。本轮未更改原生扩展，无需重建 py312 扩展；未重建 Python 3.14 wheel，本次 2099 项结果仅属于 py312。未提交、推送或发布。

本轮不构成对任意相关输入、高维非线性/噪声问题的可靠性保证。负分数及完全冗余变量零分保留原有含义；MARS 高阶交互/GCV 搜索局限和 Morris 物理步长语义仍按已有记录处理。
