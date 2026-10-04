# 2026-09-29 敏感性分析数值正确性专项

## 结论

本轮优先检查“指标算得对不对”，覆盖当前 7 类敏感性方法。解析模型和手算统计量提供了比 smoke/形状检查更直接的证据，也确实复现并修复了 4 处计算问题。不能据此宣称所有方法、所有数据和参数组合均已验证。

Sobol / FAST / RBDFAST 已做一阶、总效应或交互项的解析基准；Morris 已做有符号导数及不等参数范围的对照；RSA 有手算分布统计量；DeltaTest 有不依赖 KDTree 的逐对距离参照；MARS 本轮证据限于单活跃变量识别和输出量纲一致性，不等于已经完成任意非线性代理重要性的独立数学验证。

## 四处复现与修复

1. **DeltaTest 邻居平均系数错误。** 原代码已在每个样本内部取邻居均值，末尾又除以邻居数 k，缺少正确的 `1/2` 系数。对于单输出，现明确采用 `δ = sum_i sum_j (y_i-y_neighbor(i,j))² / (2Nk)`。旧值与修复后之比为 `2/k`：k=1 时翻倍，k=3 时变成 2/3；默认 k=2 恰好掩盖了错误。
2. **DeltaTest 重复输入的自身排除错误。** KDTree 的首个零距离结果不一定是查询样本自身；原先直接丢掉第一列，可能保留自身并丢掉另一个样本。现在按行索引明确排除自身再取 k 个邻居。三个重复输入、输出 `[0,1,4]`、k=2 的手算结果为 `13/3`。
3. **DeltaTest 非零小量纲归一化被清零。** 输出乘以 `1e-6` 后，原始分数应乘以 `1e-12`、相对权重不变；原代码对总量使用绝对近零阈值，把合法非零量当成零。现在仅在总量确为零时采用零权重。
4. **MARS 同样的归一化问题。** 单活跃变量模型的原始分数约 `0.7528`，输出缩小后约 `7.528e-13`；修复前归一化结果从 `[1,0,0]` 错变成全零。修复保留非零总量下的相对权重。

修改限于 [DeltaTest](../UQPyL/analysis/methods/delta.py) 与 [MARS 分析](../UQPyL/analysis/methods/mars.py) 的这些逻辑。没有更换 MARS 拟合算法、改变 RSA 统计量、裁剪 Delta 的负分数或重新定义 Morris。

Delta 的 1-NN 参照依据 [Eirola 等，Using the Delta Test for Variable Selection，2008，第 2.2 节](https://www.esann.org/sites/default/files/proceedings/legacy/es2008-39.pdf)；当前 k-NN 入口采用同样的半均方差定义并对 k 个邻居平均。它不是 SALib 的分布型 Delta 指标。

## 可复现实验

脚本：[check_analysis_science.py](verification/check_analysis_science.py)，原始数值：[JSON](verification/0929-analysis-science.json)，汇总：[日志](verification/0929-analysis-science.txt)。运行：

```bash
conda run --no-capture-output -n py312 python agent/verification/check_analysis_science.py
```

固定种子为 3、17、41、73、101。方差方法各跑三个三维模型，共 45 组；Morris 及三个筛选方法各跑五组，共 20 组，总计 **65 组**。没有用生产敏感性实现生成期望值，也没有将模拟次数等同于统计置信度。

三个解析模型均采用独立均匀输入：

- 线性：`Y=X0+2X1`，Xi∈[-1,1]；S1=ST=[0.2,0.8,0]，二阶项全零。
- 纯交互：`Y=X0*X1`，Xi∈[-1,1]；S1=[0,0,0]，ST=[1,1,0]，S2(0,1)=1，其余零。
- Ishigami：`sin(X0)+7sin²(X1)+0.1X2⁴sin(X0)`，Xi∈[-π,π]。总方差为 `1/2+49/8+0.1π⁴/5+0.01π⁸/18`；一阶分量分别为 `(1+0.1π⁴/5)²/2`、`49/8`、0，非零交互分量为 `0.01π⁸(1/18-1/50)`。各分量除以总方差获得期望指数。

Sobol 使用二阶 Saltelli 基样本 8192（每种子实际评价 65536 行）；FAST 每变量块 4097（共 12291 行）；RBDFAST 使用 LHS 8192 行。阈值：Sobol 原始指数绝对误差≤0.03，FAST/RBDFAST≤0.06；不把归一化展示分数拿来替代原始指数。

五种子最大绝对误差如下，数值不是相对百分误差：

| 方法 / 模型 | S1 | ST | S2 |
|---|---:|---:|---:|
| Sobol / 线性 | 0.000000229 | 0.000000229 | 0.000000572 |
| Sobol / 纯交互 | 0.0000687 | 0.001100 | 0.001100 |
| Sobol / Ishigami | 0.001179 | 0.001716 | 0.001204 |
| FAST / 线性 | 0.001839 | 0.001839 | 不提供 |
| FAST / 纯交互 | 0.001657 | 0.002315 | 不提供 |
| FAST / Ishigami | 0.028745 | 0.046993 | 不提供 |
| RBDFAST / 线性 | 0.004553 | 不提供 | 不提供 |
| RBDFAST / 纯交互 | 0.000648 | 不提供 | 不提供 |
| RBDFAST / Ishigami | 0.014008 | 不提供 | 不提供 |

这些结果支持本轮配置下的数值正确性；FAST 在 Ishigami 上的误差明显高于线性模型，不能承诺统一精度。RBDFAST 对纯交互报告接近零的一阶效应符合定义，并不表示两个交互变量不重要。

其他方法的参照：

- Morris：单位范围线性模型五种子均恢复 `[1,2,0]`，sigma 仅有浮点舍入量；不等范围、带负系数的回归恢复真实斜率 `[3,-2,0]`。已有非线性轨迹 sigma 与极端输出量纲回归继续保留。
- RSA：两个各含四点的不交叠秩组，手算 CvM 统计量为 11/16；输入分布完全相同的另一个维度为 0。五种子单活跃变量实验均找对第一维。无关变量的有限样本 CvM 并不必须恰为零。
- DeltaTest：k=1/2/3 与独立逐对距离计算一致；重复输入不包含自身；小量纲归一化不变。五种子单活跃变量排名均正确；无关变量的删变量增量可能为负，未人为截断。
- MARS：五种子单活跃变量排名均正确，新增量纲一致性回归通过。复杂非线性/交互下的拟合误差与 GCV 重要性仍需另做更广泛验证。

## 指标含义与尚未关闭的问题

同名 `S1` 不意味着相同数学量。Sobol/FAST 是方差指标，RBDFAST 只估一阶；RSA 是区域分布差异，DeltaTest 是近邻误差增量，MARS 是删变量重拟合的绝对 GCV 变化。归一化后和为 1，也不意味着解释了相应比例的输出方差。参照 [SALib Sobol 源码](https://salib.readthedocs.io/en/latest/_modules/SALib/analyze/sobol.html) 与 [RBDFAST 源码](https://salib.readthedocs.io/en/latest/_modules/SALib/analyze/rbd_fast.html) 核对估计器结构；本轮没有宣称已运行完整跨库对照。

**Morris 存在需要明确选择的尺度语义。** 当前项目以真实 `ΔX` 为分母，结果有输出/输入单位；标准单位区间步长的基本效应则包含参数范围的影响。对于范围不同的参数，这两种排名可以不同，不能声称当前 `S1_norm` 自动实现了跨单位的“整体影响”比较。[SALib Morris 实现](https://salib.readthedocs.io/en/latest/_modules/SALib/analyze/morris.html) 区分固定网格步长和带标准差缩放的形式。当前实现已用不等范围测试明确锁定物理步长行为，本轮没有擅自更换已有语义。

中英文 [API 文档](../docs_v2/cn/api/analysis.md) 已补齐这些区别。需要后续重点处理：Morris 是否增加单位区间基本效应；MARS 的非线性/交互重要性验证；邻居法对输入量纲与距离选择的敏感性；高维、相关/非均匀输入、噪声模型及样本量收敛。极端输出缩放可能仍超过浮点可表示范围，本轮 `1e-6` 回归不是任意量级稳定性的证明。Delta 的组合搜索辅助入口也不在本轮独立公式全覆盖声明内。

## 回归与安装包验证

新增 [test_analysis_scientific_references.py](../tests/test_analysis_scientific_references.py) 共 15 项，不依赖外部参考库运行。首次 Delta 等专项 4 failed / 10 passed；另有 MARS 缩放单例失败，分别见 [首次日志](verification/0929-analysis-science-before.txt) 和 [MARS 复现](verification/0929-analysis-mars-scale-before.txt)。修复后连同 Delta 既有测试 **26 passed**，见 [专项结果](verification/0929-analysis-science-after.txt)。

全量及重新构建的 Python 3.14 wheel 结果见 [py312 日志](verification/0929-analysis-science-py312-full.txt)、[构建日志](verification/0929-analysis-science-py314-build.txt)、[安装包日志](verification/0929-analysis-science-py314-wheel.txt)。本轮不提交、不推送，远程 CI 状态不变。

最终结果：65 组实验全部满足脚本验收条件；py312 **2020 passed，59.17 秒，零警告**；新构建的 Python 3.14 独立 wheel **2020 passed，35.55 秒，零警告**。独立安装依赖检查及 10 个原生扩展导入通过。触达的两份生产源码、新测试和实验脚本通过 Ruff 静态/格式检查，`git diff --check` 通过。不同任务并行执行，耗时不作为算法性能结论。
