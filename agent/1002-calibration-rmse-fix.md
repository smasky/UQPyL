# 校准 RMSE 数值稳定性修复（2026-10-02）

CAL02 实现完成。旧 `sqrt(mse(...))` 要求中间平方可表示，导致最终本可表示的 RMSE 在小量纲为 0、大量纲为 inf，并破坏 GLUE/SUFI2 的排序和筛选。

新实现先求残差，按每条模拟残差的最大二进制指数缩放，再计算均方根并恢复指数。普通有限差值直接相减，保留“大观测完全相消、另一位置有极小误差”的有效信号；只有有限输入相减溢出的行采用半尺度相减，并在最后补回指数。未用绝对阈值判零，不修改观测、不标准化模型或改变指标定义。

最终 RMS 确实超范围、舍入为零或无穷大时继续返回并发 RuntimeWarning：

```text
RMSE exceeds floating-point range; underflow is returned as zero and overflow as infinity.
```

mask、单/多模拟行和输入保护保持。MSE/MAE 未修改；真正不能表示的平方量不在 RMSE 可恢复范围的承诺内。

## 证据

- 新增 `tests/test_calibration_rmse_range.py` **26 项**，使用 Decimal 精确浮点输入、100 位精度计算独立参照：7 档量纲×有/无 mask，有限相减溢出但均方根可表示、大数精确抵消后保留小误差、次正规值、大偏移的相邻浮点数、最终真正溢出/下溢告警，以及两个公共方法三种尺度的实际筛选。
- 校准专项 **285 passed，2.08 秒，`-W error`**，见 `verification/1002-rmse-range-targeted.txt`。
- 原 63 组审查重跑，保存 [after-rmse JSON](verification/1002-calibration-science-after-rmse.json) 和 [输出](verification/1002-calibration-science-after-rmse.txt)，历史审查和 after-sufi2 文件保留。
- 观测 [1,2]，候选 [[1.1,2.2],[1,2.1]]：整体乘 1e-200 后 RMSE 正确为约 1.58114e-201/7.07107e-202；乘 1e200 后约 1.58114e199/7.07107e198，均无告警。GLUE 三种尺度只保留第二项，SUFI2 三种尺度都选第二项，先前错误消失。CAL01 的合法采样复核保持。
- Ruff/格式检查通过，中英文 API 和测试导航同步。最终 py312 全量 **2671 passed，56.72 秒，`-W error` 零未捕获警告**，见 `verification/1002-rmse-range-full.txt`。

本轮没有改变 ES/IES 的确定性更新/后验语义，也不宣称校准模块全部科学适用场景已验证。未提交、推送或重建 wheel；其他既有待处理/暂缓状态保持。
