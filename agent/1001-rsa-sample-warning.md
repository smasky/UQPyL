# 2026-10-01 RSA 样本不足告警补齐

按用户选择，SA08 已处理：**非恒定输出没有可比较区域时发出 `RuntimeWarning`，继续返回并保存结果**。原来静默返回的零值现在明确标记为占位，不表示参数不敏感。

## 当前行为

- 每个区域及其补集各至少两个样本，才参与双样本统计量计算。
- 非恒定输出的有效区域数为零时，每个输出告警一次，提示增加样本或减少 `nRegion`；`S1` / `S1_norm` 保留有限零占位，状态为 `insufficient_samples`。
- 有可比较区域时沿用区域统计量均值，状态为 `estimated`；二值输出的空区域本身不会告警，不自动修改区域数。
- 真正恒定输出返回零，状态为 `constant_output`，不告警。
- `nRegion` 在分析时校验，须为至少 2 的整数，排除布尔值；错误配置在模型评价前明确 `ValueError`。空样本矩阵也明确拒绝。

`result.extra["rsa_regions"]` 保存 `n_regions`、`n_samples` 及按所选输出行排列的 `outputs`。每个输出记录 `output_label`、`status`、`valid_region_count`、`region_sample_counts`；SQLite / `AnaReader` 往返保留这些字段。多输出中不足的列独立告警，其余列正常计算。

`estimated` 仅说明存在可比较区域，不是统计精度保证；warning 也不会凭空补充样本信息。调用者如自行将 warning 配置为异常，仍遵循 Python 的警告过滤机制。

## 验证

生产修改仅涉及 `UQPyL/analysis/methods/rsa.py`。新增 15 项到既有 `tests/test_analysis_rsa_statistics.py`，并扩展恒定输出测试的状态断言。

| 检查 | 结果 |
|---|---|
| 修改前新协议用例 | 16 failed、11 passed；包括旧恒定输出用例新增的诊断断言 |
| RSA / 普通分析 / 数值参照 / 运行生命周期相关测试 | 95 passed，18.19 秒 |
| 最终 py312 全量，`-W error` | **2202 passed，100.87 秒**；预期 warning 均捕获，无未捕获警告 |
| 独立 Decimal 分区与 CvM 对照 | 81 组通过；72 组普通 NumPy 分区保持，15 组不足场景准确告警 |
| Ruff 静态/格式与差异检查 | 通过 |

重点用例包括 `Y=X`、20 行默认 20 区域的不足诊断，只有一次事件的二值输出，多输出选择/内部模型评价、方法复用以及诊断持久化。20 行两区域的独立手算 `T=10000/2000-399/120=1.675` 保持。

证据：[修改前日志](verification/1001-rsa-sample-warning-before.txt)、[相关测试](verification/1001-rsa-sample-warning-targeted.txt)、[全量测试](verification/1001-rsa-sample-warning-full.txt)、[81 组独立参照](verification/1001-rsa-sample-warning-reference.json)、[参照日志](verification/1001-rsa-sample-warning-reference.txt)。参照脚本增加 `--output`，本轮证据另存，没有覆盖前轮结果。

API 中英文、测试导航、TODO 和交接已同步。SA09–SA11 保持待处理，本轮未修改 DeltaTest、Sobol、Morris 或 MARS；没有重建 3.14 wheel，未提交/推送。
