# 2026-10-04 删除 obsLabels

按用户明确范围，仅删除观测标签 `obsLabels`；输入、目标、约束标签保持原用途。

- 移除 ModelProblem 的构造参数、默认标签生成和长度检查。
- 移除 CalResult 的 obsLabels 字段及 result/reader 摘要的 obs_labels。
- 清理中英文示例、API、Changelog 和既有测试中的观测标签参数。
- obs、mask、模拟列仍按位置对应，一维观测/二维模拟协议和计算公式均不变。

四种校准方法、有/无 mask 共 8 组的 42 个结果数组与迁移前基线完全相同，见 [数值对照](verification/1004-remove-obslabels-parity.txt)。

测试调整：删除已无意义的 4 项标签长度验证，替换为 1 项接口/结果字段不存在检查；四方法存储往返补充问题对象、CalResult、汇总均不含观测标签的断言。测试总数因此由 3178 减少 3 项，数值和形状覆盖不减。

conda py312 完整回归 **3175 passed，109.94 秒，-W error --strict-markers**，结果见 [日志](verification/1004-remove-obslabels-full.txt)。生产代码、.github、新接口/文档测试的 Ruff 检查、生产格式检查和 git diff --check 通过。

未提交、推送、重建安装包或发布。此前观测迁移报告中 obsLabels 相关描述为该轮历史快照，以本记录和当前接口为准。
