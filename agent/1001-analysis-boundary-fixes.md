# 2026-10-01 敏感性边界问题修复

上轮确认的 **SA01–SA07 已按报告范围处理**：修正错误数值，或在样本无法可靠估计时明确拒绝。本轮新增 **50 项回归**；conda py312 针对性 **63 passed，1.37 秒**，最终全量 **2187 passed，50.91 秒，`-W error` 零未捕获警告**。保持 Morris 标准单位区间唯一模式、MARS 默认二阶及 R²<0.8 warning 约定，未提交/推送。

“已处理”不表示新增离散 RBDFAST 估计器，也不表示消除了 MARS 的高阶贡献偏差。原问题和修复前数据见 [边界复核](1001-analysis-module-review.md)。

## 实现与结果

| 编号 | 当前处理 | 修复后证据 |
|---|---|---|
| SA01 Morris 原类型差分 | X/Y 的计算数组转换为浮点，原始样本/输出及类型继续记录；先检查有限性 | float/uint8/bool 阈值输出均为 mu=mu_star=1.5、sigma=0；真实 uint8 回调一致；整数输入例均为 mu=mu_star=2 |
| SA02 RBDFAST 固定/重复取值 | 固定样本列直接为零；非恒定重复取值明确 ValueError；排序只做一次并在多输出间复用 | 四种行顺序下固定参数 S1 都为零，活跃参数约 0.997629；原离散/重复反例全部明确拒绝 |
| SA03 RSA 极端值/非有限数据 | X/Y 有限性检查；线性分位数按端点符号采用安全插值，不统一缩放原数据 | 输出×1e308 的 S1 从错误零恢复为手算 0.375；NaN/inf 不再伪装零分；微小和巨大输出共存仍分区正确 |
| SA04 FAST 不完整块 | 元数据正整数 M/N、N>4M²、可选 blockSize=N；严格要求总行数=N×nInput | 缺一行、缺两行、多一行均拒绝，额外输出不再静默忽略；完整设计 S1 活跃项约 0.997707 |
| SA05 RBDFAST 无效谐波预算 | M 为正整数，布尔值不算整数；N>2M，否则 ValueError | 三组 N≤2M 不再返回 `[1,1]`；正常连续 LHS 保持解析排序与行重排一致性 |
| SA06 Sobol 基础方差零 | 在分离 A/B 后检查实际估计分母；混合输出变化时明确提示增加基础样本量 | 一阶/二阶设计的稀有事件小样本都明确诊断；N=4096 对照近似总体 S1/ST/S2 |
| SA07 Morris 单轨迹 sigma | 至少两条完整轨迹，否则 ValueError | 空样本/单轨迹不再返回 NaN sigma 或通用 NumPy warning；两轨迹非线性手算参照保持 |

### 数值修复

Morris 仍计算 `ΔY / (ΔX / input_range)`。改用带符号浮点计算，避免 [NumPy diff](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) 保留 uint8 或 bool 类型产生的回绕、方向丢失。计算数组不原地改动调用者数据，返回及保存的 X/Y 保留原始数值和类型。

RSA 保留线性分位数和区域比较公式。同符号端点采用靠近端点的插值；异符号端点采用加权和，避免 `upper-lower` 在 ±1e308 附近溢出。没有先把整个输出除以最大幅值，否则 ±1e-200 可能在 ±1e308 的数组中提前丢失区分。分位数每输出只计算一次，不再为各输入重复排序。原始 Y 仍保存，合法恒定输出返回零的约定保留。

### 样本与方法的适用范围

RBDFAST 的既有排序及偏差修正结构可见 [SALib 实现](https://salib.readthedocs.io/en/latest/_modules/SALib/analyze/rbd_fast.html)。本项目增加的重复值拒绝是明确的适用性限制：不使用抖动、任意并列次序或改换统计公式来制造一个看似稳定的分数。非恒定重复列，包括很多整数/离散样本，现在会报错；固定样本列按零贡献约定处理。现有连续随机样本、输出尺度和多输出公式保持。N>2M 是修正分母有效的最低条件，不保证刚超过该下限的小样本足够准确。

FAST 仍要求完整的原始采样块顺序，元数据现在必须含 N；`FASTDesign.sampleWithMeta()` 已完整提供这些字段，不需要用户手动构造。可选 blockSize 也会核对。M/N/块长度等元数据错误在真实模型评价前处理；行数不符同样提前拒绝。本轮没有加入任意行重排后的采样设计重建。

Sobol 保留真正全体恒定输出返回零。基础样本未观察到稀有事件而混合样本有变化时，不能当成总体恒定输出；现在明确 ValueError 并提示增加 N。正常模型仍存在抽样误差。Morris 的 sigma 使用样本标准差，两条轨迹是可计算的最低要求，也不等于足够完成可靠筛选。

用户要求 MARS 低 R² 使用 warning 的约定保持；该约定不被扩展为其他方法的无效结构、非有限数据也必须继续返回。

## 验证

修复前，四个触达测试文件 **40 failed、23 passed，1.83 秒**，确认新增断言确实捕获原错误；修复后同一批 **63 passed**。新增 50 项分布：Morris 8、RSA 10、RBDFAST 17、FAST/Sobol 设计与方差 15。既有用例没有删除或降精度；最终全量从 2137 增至 **2187**。

独立复跑上轮 **32 组记录**：16 组有效样本返回预期指标，16 组无效/不足/不支持样本明确 ValueError，所有记录无 warning；不存在裸 ZeroDivisionError。原检测 JSON 保留，修复后另存文件。

RSA 额外 **81 组**对照，包含普通正/负/正态/带大偏移/二值输出、输出单位缩放、±最大有限浮点与最小正次正规数。用 1100 位 Decimal 计算独立分位数，再按区域独立计算双样本 CvM：区域成员逐项一致，最终统计量最大绝对差 **0**。其中 **72 组**可安全运行 NumPy 原分位数的对照，区域成员也保持一致。这里比较的是所测区域及统计量，不宣称所有插值阈值都逐位相同。

触达 Python 文件 Ruff 格式/静态检查通过，差异检查通过。EN/CN API 和测试导航已同步。没有修改原生扩展或新增依赖，没有重建 Python 3.14 wheel；此前跨版本测试不作为本轮变更验证。

## 保留的问题

本轮不修改 DeltaTest 或 MARS 的生产实现。DeltaTest 前轮修复继续由全量回归覆盖，本轮没有发现新问题，不等于全面证明正确。

MARS 高阶贡献低估仍是已知局限：默认二阶可能稳定地遗漏三阶贡献，即使 R²≈0.960 且没有告警。提高阶数也不能普遍保证贡献准确。既有反例见 [三项处理记录](1001-analysis-followup-fixes.md)，本轮没有重新运行其独立实验。下一步若继续关注 MARS，应对实际模型做独立贡献参照、样本量和阶数对照，不能把本轮七项修复表述为敏感性模块已全面正确。

## 证据与复现

- [修复前失败日志](verification/1001-analysis-boundary-before.txt)
- [针对性通过日志](verification/1001-analysis-boundary-targeted.txt)
- [最终全量日志](verification/1001-analysis-boundary-full.txt)
- [原 32 组修复前数据](verification/1001-analysis-module-review.json)
- [32 组修复后数据](verification/1001-analysis-boundary-after.json)及[运行日志](verification/1001-analysis-boundary-after.txt)
- [RSA 81 组独立对照](verification/1001-analysis-boundary-quantiles.json)及[运行日志](verification/1001-analysis-boundary-quantiles.txt)
- [边界复现脚本](verification/review_analysis_module.py)、[Decimal 独立验证脚本](verification/verify_rsa_quantiles.py)

在项目根目录复现：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. /home/wmtsky/anaconda3/bin/conda run --no-capture-output -n py312 python agent/verification/review_analysis_module.py --output agent/verification/1001-analysis-boundary-after.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. /home/wmtsky/anaconda3/bin/conda run --no-capture-output -n py312 python agent/verification/verify_rsa_quantiles.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/wmtsky/anaconda3/bin/conda run --no-capture-output -n py312 pytest -q -W error
```
