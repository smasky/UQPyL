# 2026-10-02 校准误差指标与 ESS 诊断修复

本轮完成上一轮剩余项复核确认的三个问题；不改变权重公式，不把告警当作精度修复。

## 数值修复

- MSE、MAE 与已修复的 RMSE 共用二进制残差尺度计算。先计算有界残差的矩，再恢复物理单位，避免中间相减、平方或求和溢出。
- `mse([0,0], [[1.4e154,0]])` 从错误的 `inf` 恢复为约 `9.8e307`。
- `mae([-1e308,1e308], [[1e308,1e308]])` 从错误的 `inf` 恢复为 `1e308`。
- GLUE/SUFI2 公共运行流程中的大单位评分、最佳成员及 GLUE 阈值筛选均加入回归验证。
- 最终结果真正超出双精度范围时，保留零或 infinity，并发出 `RuntimeWarning: MSE exceeds floating-point range; underflow is returned as zero and overflow as infinity.`；MAE/RMSE 使用对应名称。不伪造有限值。

## ESS 诊断

- GLUE 显式使用 `logLikelihood` 且有效样本量低于 20 时，发出 `RuntimeWarning: GLUE uncertainty effective sample size is below 20; weighted intervals may be unreliable.`。
- `diagnostics['uncertaintyStatus']` 标记 `low_effective_sample_size` 或 `estimated`，SQLite 往返保留；estimated 不代表已证明区间准确。
- 没有显式似然的等权阈值筛选记录该状态，但不发出似然权重退化警告。
- 原始权重和区间完全保留，ESS=1 时不自动改等权或人为扩大区间。
- GLUE/SUFI2 的 20 阈值加入浮点舍入容差，防止 20 个等权样本因计算得到 19.999999999999996 而误报。

## 验证

- 新增 `tests/test_calibration_range_and_ess.py`：32 项，覆盖独立 Decimal 参照、mask、只读输入、消减、次正规数、真实溢出/下溢、公共筛选、ESS 与存储。
- 原有两项显式低 ESS 测试补充预期 warning 捕获，原数学断言保留。首次日志保留其预期新增告警失败，不隐藏。
- 校准专项：**418 passed，2.80 秒**，日志 [targeted](verification/1002-calibration-range-ess-targeted.txt)。
- py312 全量 `python -m pytest -q -W error`：**2795 passed，67.20 秒**，日志 [full](verification/1002-calibration-range-ess-full.txt)。
- [独立脚本](verification/check_calibration_error_range.py)：3 种子 × 10 个数量级 × 3 指标 × 3 行，共 270 条 Decimal 120 位参照，最大相对误差 **3.3306690738754696e-16**；真实范围越界的 warning 一并核对。[结果](verification/1002-calibration-error-range.json)。
- Ruff 检查通过；新增测试中两处 lambda 赋值最后改为等价局部函数。

## 范围

这次修复实际数值计算和诊断缺口，不解决 ES/IES 非线性与多峰后验近似、边界裁剪、高维重要性采样退化等方法限制。SUFI2 搜索集合包络仍不能自动解释为参数可信区间。未提交、推送或重建 wheel。
