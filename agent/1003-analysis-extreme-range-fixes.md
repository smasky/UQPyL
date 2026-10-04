# 敏感性极端范围补齐（2026-10-03）

本轮完成 SA09（DeltaTest 输入范围中间溢出）和 SA11（Morris 派生统计量恢复范围诊断），未扩展到 Problem 模块。conda py312；未提交、推送或重建 wheel。

## DeltaTest

普通归一化 `(X-lower)/(upper-lower)` 保持原运算。先识别真实固定维，再仅对有限值相减发生溢出的列，使用 `(X/2-lower/2)/(upper/2-lower/2)`。分子和分母中间溢出均处理；不把无限分母归一化为零，不裁剪越界样本。输出仍无法用有限数表达时保留明确失败。

analyze、findCombVio、findCombEA 共用 `_scaleInputs`，离散参数使用真实选项范围。正常尺度不改变计算路径，真实固定列必须等于声明值。

独立 Decimal 参照覆盖跨零极大范围、不对称极大范围、有限范围而越界分子溢出、次正规数范围；输入数组未修改。四个种子0/3/17/41下，公共分析、穷举和真实GA搜索均与普通尺度一致，保留活跃第一个变量。真实GA使用相同seed和50次评价预算，并核对bestDecs/bestObjs。

重跑旧复现审计，20条Delta记录全部正常；×1、×1e100、×1e308的归一化分数最大差 **0**，无warning，原来三个种子误选无关变量的问题消除。

## Morris

已有有限EE检查保持。在mu、mu_star、sigma恢复输出单位后检查真正溢出与下溢归零。对受影响字段发出RuntimeWarning，指出输出行和输入列；`extra['morris_statistic_status']`按输出×输入记录available/overflow/underflow，随SQLite保存。原`morris_effects`字段保持。

理论sigma超过double范围时不能制造有限正确值，因此保留inf并明确标为overflow；不是通过warning掩盖数值错误。归一化指标在恢复前计算，保持可用。有限普通/大尺度恢复值和真零不告警；可表示范围以下的非零均值归零标underflow。

旧例Y尺度1/1e307/1e308，理论sigma为`1.5*sqrt(2)*scale`：前两组有限正确；最后一组基本效应仍有限，但sigma真实超过double上限，返回inf并有且只有一条字段明确的告警，S1_norm=1。新增SQLite往返检查确认inf与诊断同时保留。

```text
RuntimeWarning: Morris sigma overflow in output units for output 0, input indices [0]; see extra['morris_statistic_status']. S1_norm remains available.
```

范围：本次诊断针对统计量恢复，不声称已经支持基本效应本身溢出或Morris输入范围超出double的全部情形；既有非法输入和非有限EE仍停止。

## 验证

- [新增13项回归](../tests/test_analysis_extreme_input_and_statistics.py)。最初12项在修复前 **11失败/1通过**，修复后通过，再补充1项下溢/真零与真实GA断言。
- [专项](verification/1003-sa-range-targeted.txt)：**92 passed，1.78秒，-W error**。
- [旧审计重跑](verification/1003-sa-range-audit.json)：Delta20条/Morris3条，保留原1001证据，未覆盖旧文件。[汇总断言](verification/1003-sa-range-summary.json)。
- [全量日志](verification/1003-sa-range-full.txt)：**3042 passed，76.75秒，-W error**。预期warning由对应测试捕获。
- Ruff与差异空白检查；中英文API/测试导航同步。

复现旧审计可在 `PYTHONPATH=.:agent/verification` 下导入 `review_analysis_postfix.reviewDelta/reviewMorris`，再以sanitize保存为新文件。该脚本原main包含其他方法并写历史路径，本轮未调用它覆盖历史证据。

本轮不改变MARS高阶贡献限制、其他算法的统计局限和已暂缓功能。全量通过不等于对任意输入证明算法正确。
