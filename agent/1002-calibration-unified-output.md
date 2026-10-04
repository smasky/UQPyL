# 校准统一输出（2026-10-02）

用户认可统一为“最佳解、参数集合、对应模拟、评分、可选权重/区间，并标记集合类型”。本轮实现该协议，不改变算法更新、采样、评分或数值区间。

## 通用 CalResult 字段

| 字段 | 协议 |
|---|---|
| bestDecs / bestSim | 继续提供最佳参数与对应完整展平模拟 |
| best_score / best_index | 原始最佳评分，以及最佳行在通用主集合中的位置 |
| samples / simulations / scores | 主集合参数、完整展平模拟、逐行原始评分，严格同行对应 |
| sample_kind | GLUE=behavioral，SUFI2=sampling_ensemble，ES/IES=updated_ensemble |
| weights | 与主集合对应的显式权重；没有则 None |
| intervals | 带 kind/space/lower/upper/probability/sample_source/indices 的列表，没有则空列表 |
| uncertainty | 可选独立先验加权结果，没有则 None |

新对外字段按 AGENTS 约定采用 snake_case，内部状态沿用驼峰。旧方法特定字段保留；通用代码不再需要根据方法切换 posteriorDecs/behavioralDecs。SUFI2 的主集合明确是最后一轮搜索集合，不能解释为精确后验。

区间的 indices 将未屏蔽输出映射到完整 simulations 的列。sample_source=samples 表示主集合，uncertainty.samples 表示独立先验池。SUFI2 的独立先验权重仅位于 uncertainty 内，主集合 weights 保持 None，避免将不同样本数/分布的权重混用。probability 是分位水平，不是经过验证的覆盖率。

GLUE 保留原始候选 bestIdx 作方法特定诊断；best_index 单独映射到筛选后的通用集合。ES/IES 不自动生成未经估计的可信区间。构造结果只复制已有结果，不重新评价模型。

## 顺手修复的摘要缺陷

此前 GLUE/SUFI2 未设置 extra.bestIdx，summary 默认取 scores[0]，最佳参数不在首行时，摘要最佳评分错误。本轮两方法明确保存原始最佳索引，统一结果生成 best_score，摘要与 reader 同步使用它。新增独立例中首行误差非零、第三行精确匹配，摘要应为 0。

summary()/CalReader.get_run_summary() 还统一输出 sample_kind/n_samples/has_weights/interval_count/best_index。SQLite result 与小摘要都保留通用协议。

## 验证

新增 [13 项测试](../tests/test_calibration_unified_output.py)：四方法主集合形状/评分/最佳行、GLUE 筛选后索引转换、保存读取、reader 摘要、mask 区间列映射、先验权重独立来源、未估计区间为空、通用字段与状态/旧字段的数组隔离，以及两方法摘要非首行最佳评分。

专项 **386 passed，4.98 秒，-W error**。[专项日志](verification/1002-calibration-output-targeted.txt)。全量 py312 **2763 passed，86.21 秒，-W error 零未捕获警告**，见[全量日志](verification/1002-calibration-output-full.txt)。Ruff 与触达差异空白检查通过。

中文/英文 API、使用指南和测试导航同步。未新增模拟次数或依赖，未提交/推送/重建 wheel。上轮非线性精度、采样预算及不确定性解释的限制保持。
