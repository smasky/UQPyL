# Changelog

本页记录 UQPyL 及其文档中面向用户的变更。

## Unreleased

### 文档

- 将文档重构为面向用户的工作流指南。
- 将 API 参考拆分为 `problem`、`doe`、`analysis`、`optimization`、`inference`、`calibration`、`surrogate` 等模块页面。
- 扩展示例，加入可运行代码、期望输出、verbose 输出片段和常见错误。
- 增加对 `Problem` 评估、批量目标函数、标量/向量边界、结果对象和 sqlite reader 的说明。
- 增加中文用户文档，包括 Quick Start、模块指南、Examples、API Reference 和拆分 API 摘要页。

### 注意

- `Problem.evaluate()` 的正式返回协议是 `Eval` 对象。使用 `res.objs` 和 `res.cons`，不要使用字典式访问。

## Released Versions

本地仓库可见的 `v2.0.x` tag：

| Version | Date |
|---|---|
| `v2.0.11` | 2024-12-24 |
| `v2.0.10` | 2024-12-24 |
| `v2.0.9` | 2024-10-15 |
| `v2.0.8` | 2024-10-15 |
| `v2.0.7` | 2024-10-02 |
| `v2.0.6` | 2024-09-19 |
| `v2.0.5` | 2024-06-04 |
| `v2.0.4` | 2024-05-08 |
| `v2.0.3` | 2024-05-07 |
| `v2.0.2` | 2024-05-07 |
| `v2.0.1` | 2024-04-20 |
