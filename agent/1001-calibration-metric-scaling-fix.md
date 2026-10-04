# 2026-10-01 校准指标单位换算修复（OM01）

**OM01 已完成。** 修复校准模块 NSE/R²/PBIAS/Pearson/KGE/R-factor 的绝对阈值判零；有效小量纲观测不再被误判。普通例子的 NSE 为 0.98，换算单位后仍为 0.98。OM02–OM04 未在本轮修复。

## 实现

修改 [calibration/util.py](../UQPyL/calibration/util.py)：

- 使用 `frexp`/`ldexp` 将内部计算值按二进制幂缩放，避免变化平方或标准差在共同单位换算后下溢/溢出。NSE/PBIAS/区间宽度使用共同尺度；Pearson 的模拟行可独立缩放，KGE 恢复各行标准差/均值比中的尺度差。原始数据和输出指标定义保持。
- 取消把 `np.isclose(value, 0)` 当作零方差、零和或零均值的判断。有限常数序列直接按原始值相等识别，防止均值舍入产生伪变化；缩放后的零分母继续明确报错。
- 正负抵消严重的有限求和使用 `math.fsum` 补偿。其误差界只选择准确求和路径，不用于将小量判为零。`[1e20,1,-1e20]` 的非零和不再被朴素求和丢失；`[-7,3,4]` 的真零和仍正确识别。
- KGE 使用 `hypot` 合成误差，避免平方中间值溢出却使本可表示的最终分数变为 inf。

mask、批量形状、PBIAS 符号约定、真实常数/零和/零均值错误语义保持。没有修改 ES/IES 参数更新及协方差公式；没有修改 MSE/MAE/RMSE 的有量纲计算路径。正常浮点结果允许最后位差异，不承诺逐位相同。

## 验证

新增 [test_calibration_metric_scaling.py](../tests/test_calibration_metric_scaling.py) **65 项**，覆盖：

- 六个无量纲入口的独立固定参照，在共同尺度 `1e-200/1e-12/1e-6/1e6/1e200` 下保持分数；包含多个模拟行与原数组不变检查。
- mask 后的 NaN 排除、小量纲真正常数、真实零观测和/均值、小而非零的均值、正负抵消。
- 多个相同值求均值时的舍入影响、相关系数的独立模拟尺度、有限大 KGE、观测含大偏移时仍可表示的变化。
- GLUE/SUFI2 公开运行：相同候选、原单位与 `1e-12` 单位下的 NSE 分数、最优参数和 behavioral/elite 筛选保持；SUFI2 的 R-factor 也保持。

首批 55 项修复前 **36 failed / 19 passed**。后续扩展的常数检查揭示初步去阈值实现仍受均值舍入影响，12 项失败；已加原值常数检查，最终 65 项全部通过。未放宽独立参照精度或真退化预期。

最终校准专项 **259 passed，1.78 秒**；完整 py312 **2295 passed，55.13 秒，`-W error` 零未捕获警告**。新增 65 项使前轮 2230 项增加到 2295 项。Ruff 检查通过。

独立审查脚本保留原检测 JSON/日志，另写修复后文件：**169 组记录**，36 组无量纲尺度参照全部正确；原来的八组错误拒绝消失。其余 23 组异常来自尚未处理的 OM02–OM04，106 组跨模块正常控制通过。不是 169 项新增 pytest，也不把剩余异常记成已解决。

## 范围与剩余项

独立脚本初次扩展所有损失到 `1e-200` 时，还记录到原 RMSE 平方路径下溢为零：两行数学 RMSE 约 `1.58e-201/2e-201`，实际零，见保留的 initial 日志。这是另一个极端尺度的有量纲计算问题，**本轮没有修复**；最终脚本把极端尺度限定到此次修复的六个无量纲指标，原 MSE/MAE/RMSE 普通范围对照保持。不能将本次完成解释为全部损失的任意极端范围均安全。

OM02–OM04、敏感性 SA09/SA11、MARS 高阶局限与 C15/C17/C20 既有暂缓项保持。未重建 Python 3.14 wheel，未提交、推送或发布。API 中英文、测试导航、TODO/交接已同步。

## 证据

- [首批修复前](verification/1001-calibration-metric-scaling-before.txt)
- [初步实现的常数舍入反例](verification/1001-calibration-metric-constant-before.txt)
- [校准专项](verification/1001-calibration-metric-scaling-targeted.txt)
- [完整回归](verification/1001-calibration-metric-scaling-full.txt)
- [修复后独立记录](verification/1001-other-modules-after-om01.json) / [摘要](verification/1001-other-modules-after-om01.txt)
- [保留的 RMSE 极端范围观察](verification/1001-other-modules-after-om01-initial.txt)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/wmtsky/anaconda3/bin/conda run --no-capture-output -n py312 \
  pytest -q -W error tests/test_calibration*.py
```
