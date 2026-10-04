# 2026-10-01 七项修复后的剩余边界复核

后续状态更新：**SA08 已补齐 warning 与不足标记，SA10 按用户最新选择采用 warning / 完整结构恢复 / 无法判断时未估计**，见 [RSA 修复及验证](1001-rsa-sample-warning.md)、[Sobol 最新修复及验证](1001-sobol-metadata-fix.md)。最终 py312 全量 2230 passed；SA09/SA11 仍待处理。下文数据与“只检测”描述保留此次复核当时的证据，不能当作 RSA/Sobol 当前行为。

除 MARS 已知高阶贡献局限之外，本轮又确认 **SA08–SA11 四项待处理问题**。其中 RSA 的样本/分区不足、Sobol 的元数据一致性检查可能在普通数值范围触发；DeltaTest 和 Morris 两项涉及极端浮点范围。前轮 SA01–SA07 保持已完成，不能把本轮新问题说成前轮修复未实施。

本轮是检测，保存 **37 组边界与对照记录**，没有改生产代码或 pytest，没有重新跑全量测试/重建 3.14 wheel，未提交/推送。上一轮 **2187 passed** 的全量记录继续有效，但未覆盖本轮确认的这些场景。37 组记录含错误结果与异常，不是“37 项数值验证全部通过”。

| 编号 | 方法 | 确认问题 | 当前状态 |
|---|---|---|---|
| SA08 | RSA | 非恒定输出没有可比较区域时，仍无提示返回零；nRegion 配置缺少明确校验 | 已完成 · 后续采用 warning + 不足状态/区域计数，配置校验补齐 |
| SA09 | DeltaTest | 极大有限上下界的范围差溢出，使变化参数被当成常数，分数/组合选择错误 | 待处理 |
| SA10 | Sobol | N、blockSize 与 secondOrder 不一致未检查，可按错误步长解释样本 | 已完成 · 最新采用 warning，结构可确认时恢复配置继续，无法判断时明确标记未估计 |
| SA11 | Morris | 基本效应有限，但恢复有量纲 sigma 后溢出，返回 inf 和通用 warning | 待处理 |

## SA08：RSA 没有有效分区仍返回零

位置：`UQPyL/analysis/methods/rsa.py` 第 113–119 行。当 validCounts=0 时，汇总明确填零；没有区分“未能估计”与“统计量为零”。配置 nRegion 在第 89 行直接参与分区。

确定模型 `Y=X`、一输入 [0,1]，20 行等间距样本，默认 nRegion=20：每个区域只有一行，区域与补集无法满足当前双样本比较条件；**0 个区域可比较，却返回 S1=S1_norm=0，没有 warning**。这不能解释为参数不敏感。

| 样本数 | nRegion | 可比较区域 | 当前 S1 | 当前 S1_norm |
|---:|---:|---:|---:|---:|
| 3 | 2 | 0 | 0 | 0 |
| 20 | 20（默认） | 0 | 0 | 0 |
| 20 | 2 | 2 | 1.675 | 1 |
| 40 | 20 | 20 | 0.3375 | 1 |
| 80 | 20 | 20 | 0.66875 | 1 |

20 行、两区域的参照可独立手算：每组 10 行， pooled ranks 为 1–10 与 11–20，U=10000，`T=10000/2000-399/120=1.675`。不同 nRegion 和样本量的统计量不要求相同；这里的控制组说明有足够区域样本时可以得到有效估计，不能把无效分区的零视为效应不存在。

nRegion=1 也对该非恒定模型直接返回零；0 泄露 IndexError，-1/2.5/True 泄露底层形状或类型异常。建议明确校验至少两个区域，并在非恒定输出没有有效比较区域时记录诊断，提示减少区域数或增加样本量；保留真正恒定输出的既有零分约定。若继续返回零，必须同时清楚标注未评估，不能无提示地当作有效指标。

## SA09：DeltaTest 极端输入范围丢失变化

位置：`UQPyL/analysis/methods/delta.py` 第 342、347 行：直接计算 `spread=upper-lower`，随后除以 spread，只检查最终 scaled 是否有限。范围溢出为 inf 而有限分子除 inf 得到零时，该检查无法识别。

两输入、128 行 LHS，seed=0/3/17/41；用单位样本 u 构造 `x0=-0.9+0.8*u0`、x1=u1，声明 x0 范围 [-1,1]，`Y=3*u0+1` 只依赖第一个参数。同步把 x0 和它的边界乘 1、1e100、1e308，保持 Y 不变。

倍率 1e308 时，所有边界/输入/输出都有限，但上下界差在 double 中溢出为 inf；因为该批 x0 都为负，`X-lower` 仍有限，缩放后 x0 全为零。正确单位坐标应为 `0.05+0.4*u0`，不能是常数。

| seed | 普通尺度 S1_norm | ×1e308 S1_norm | 普通穷举选择 | ×1e308 穷举选择 |
|---:|---|---|---|---|
| 0 | [0.967725,-0.032275] | **[0,1]** | 活跃 x_1 | **无关 x_2** |
| 3 | [0.964283,-0.035717] | **[0,1]** | 活跃 x_1 | **无关 x_2** |
| 17 | [0.966861,-0.033139] | **[0,-1]** | 活跃 x_1 | x_1（选对，但分数仍错） |
| 41 | [0.968323,-0.031677] | **[0,1]** | 活跃 x_1 | **无关 x_2** |

四组 ×1e100 控制与普通尺度一致；只在 ×1e308 失效。均有 NumPy `overflow encountered in subtract`，但仍返回误导结果。三个误选案例没有触发“无正敏感性” warning，因为无关参数被算成正分。

这与已修复的**输出**平方范围问题不同，是新的**输入**范围缩放问题。三个 DeltaTest 入口共用 `_scaleInputs`；本轮实测 analyze 与 findCombVio，未运行随机 EA。建议用安全范围换算，至少在无法正确表示缩放时明确拒绝；不能把 inf 范围除出的零当作常数维度。先对每列的输入及边界做共同二的幂次缩放，再计算比值，是可进一步验证的修复方向。

## SA10：Sobol 未核对采样元数据一致性

位置：`UQPyL/analysis/methods/sobol.py` 第 83、96–100 行。只按 secondOrder 和实际行数确定步长/基础样本量，忽略 metadata 中的 N 和 blockSize。按位置分离采样块的结构参照见 [SALib Sobol 实现](https://salib.readthedocs.io/en/latest/_modules/SALib/analyze/sobol.html)。

`Y=x0+2*x1`，独立 U[0,1]，总体 S1=[0.2,0.8]。合法二阶 SaltelliDesign、N=128、seed=17，共 768 行，blockSize=6：

| 情况 | 当前行为 |
|---|---|
| 完整设计、正确 meta | S1≈[0.207217,0.801822]，符合抽样近似 |
| 删除完整最后一块，实际 N=127，meta N=128 | 接受并返回，meta 与实际样本数不一致 |
| 不改样本，只把 meta N 改成 1 | 完全相同分数，错误 metadata 被保留 |
| 只把 secondOrder 改成 False，N=128、blockSize=6 保留 | 768 可被新步长 4 整除，未拒绝；S1≈**[0.067288,0.506804]**，ST≈[0.337967,1.077817] |

这属于无效 metadata 的防护缺口，不是正确 Saltelli 样本下公式错误。建议与 FAST 一样检查 N、blockSize、secondOrder 和完整行数的一致性。删除整块若确实是用户意图，应明确更新 metadata，并说明采样质量的变化；不要默默使用相互矛盾的数据。

## SA11：Morris 派生标准差超出表示范围

位置：`UQPyL/analysis/methods/morris.py` 第 162 行。当前检查 EE 有限，但恢复 sigma 的原单位后未检查结果。

一输入 [0,1]，两条四级网格合法轨迹 `0→2/3`、`1/3→1`，有限输出依次 `[0,s,0,-s]`，可以由确定分段函数产生。基本效应是 ±1.5s，样本标准差由 [NumPy std 的 ddof=1 定义](https://numpy.org/doc/stable/reference/generated/numpy.std.html) 为 `1.5*sqrt(2)*s`。

| 输出尺度 s | 理论 sigma | 当前行为 |
|---:|---:|---|
| 1 | 2.12132034356 | 有限结果，正确 |
| 1e307 | 2.12132034356e307 | 有限结果，正确 |
| 1e308 | 2.12132034356e308 | 返回 **inf**，仅通用 `overflow encountered in multiply` |

最后一行的 Y 和每个 EE 都有限，但理论 sigma 已超过 double 的最大有限值；不能要求强行得到正确的有限 double。缺口在于没有明确结果范围诊断或可用性标记，不应把 inf 当作正常可用统计量。建议恢复各统计量单位后检查范围，并给出方法与字段明确的诊断；这是低频的表示范围边界。

## 当前结论与后续建议

RSA 无有效区域诊断和 Sobol 元数据一致性属于应补齐的普通入口保护；DeltaTest 极端输入范围会实际算错并可能选错变量，应做安全缩放；Morris 超范围 sigma 应明确诊断。建议先 RSA/DeltaTest，再 Sobol/Morris 的入口与范围处理。

RBDFAST 不支持非恒定重复取值的限制已经明确，不是本轮新增算错；它仍不提供可靠的离散变量估计。完整合法样本下的随机估计误差和没有置信区间，也不能直接称为确认的实现错误。MARS 已知局限保持，用户要求的默认二阶和 0.8/warning 没有改变。

证据：[37 组数据](verification/1001-analysis-postfix-review.json)、[运行日志](verification/1001-analysis-postfix-review.txt)、[复现脚本](verification/review_analysis_postfix.py)。脚本复用既有观测/异常记录辅助函数，非有限记录转字符串，保持标准 JSON。触达脚本 Ruff 格式/静态检查和差异检查通过。

项目根目录复现：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. /home/wmtsky/anaconda3/bin/conda run --no-capture-output -n py312 python agent/verification/review_analysis_postfix.py
```
