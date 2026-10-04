# 2026-10-04 现有形状协议下的缺陷修复

本轮按“先修复现有协议、不实施暂定接口迁移”处理。没有将 obs/mask 改成一维，没有给 objFunc/conFunc 添加另一套公共入口；外部仍推荐 evaluate(X, target=...)。

## 修改与实际效果

- **SH01 已修复**：ModelProblem 的 masked NaN 检查接受与 obs 网格对应的模拟张量，或其显式 C 顺序展平的二维 `(N, obs.size)`。两者都按同一观测位置应用 mask。未屏蔽 NaN、错误列数或不匹配的其他布局仍拒绝，不因元素数量相同就任意重排 mask。
- **SH03 已修复**：中英文 Problem 指南中 5 处目标/评价器示例先展平模拟和观测，再按展平 mask 选列，最后沿列轴求均值。不再在二维误差上访问 axis=2；全部屏蔽时明确拒绝无有效观测的计算。
- **SH02 诊断和约定已补齐，未改变接口语义**：替代模型记录缩放/多项式展开前的输入维数，预测时先检查输入维度。把多个单参数值误当一个多参数样本时，明确提示一维代表单样本，以及 reshape(-1,1) 的写法，不再落到低层 matmul 维度错误。fit 的一维便捷输入仍表示多个单参数样本，predict 的一维仍表示一个样本；这个历史差别在中英文指南明确记录，推荐两个入口都传二维 X。没有宣称 fit/predict 一维语义已经统一。
- **SH04 明确约定**：观测、掩码、模拟和有效观测 R 必须采用同一顺序。数组单独转置/重排后即使形状相同，也不可能仅靠形状恢复真实含义；文档不再暗示可自动识别。

## 验证依据

- 新增 tests/test_model_mask_layouts.py：网格/展平模拟的 evaluate、simulate、flattenSim；未屏蔽/不匹配 NaN 的拒绝；四种校准方法 masked NaN 的同值对照；有/无缩放的替代模型预测提示与正常预测数值，共 16 项。
- 扩展 tests/test_documented_workflows.py：直接提取中英文 Markdown 的 5 个函数并执行；覆盖网格/展平、无 mask/部分 mask/全部 mask，共 12 项参数化案例。测试校验具体目标值，不仅检查能运行。
- 相关专项 **98 passed，1.87 秒**，见[日志](verification/1004-shape-fixes-targeted.txt)。
- [53 条审计重跑](verification/1004-shape-contracts-fixed.json)与[原记录](verification/1004-shape-contracts.json)比较：49 条完全不变，仅 4 条符合本轮预期变化；见[对照](verification/1004-shape-fix-parity.json)。四校准方法保持观测顺序时，展平与网格的 samples/simulations/scores 仍逐值相同。
- 完整 conda py312 回归：**3122 passed，118.56 秒，-W error**，见[日志](verification/1004-shape-fixes-full.txt)。

生产与触达测试的 Ruff lint/format、git diff --check 通过。没有放宽数值容差、屏蔽全局 warning 或改变算法公式。非法预测维度仍报错，因为无法猜测用户意图并返回可靠结果。

## 边界与发布状态

一维观测接口与统一 fit 一维语义仍是待定迁移事项，不属于已完成内容。此前的 nTime/nSeries/SQLite 协议保留。

本轮没有推送、触发 GitHub Actions 或上传 PyPI。发布流程已在上轮完成本地校验；Windows/Linux/macOS 的实际矩阵成功状态仍需推送后确认，不能用本地修改冒充远程验收。

本轮生产 Python 代码有变化，因此上轮 2.1.7 安装包的 3094 项通过是旧快照；没有再次重复本地打包，最终发布应使用最终提交的 GitHub CI 产物。
