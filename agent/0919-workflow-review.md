# 2026-09-19 端到端流程与独立安装包复核

## 范围与结论

本轮完成三条主流程的实际执行、持久化往返、文档示例执行及独立 wheel 验证。新增 11 项回归（4 项端到端、7 项文档），源码与安装包均为 **1917 passed**，使用 `-W error`。本轮未发现需要修改生产代码的新缺陷；修复文档与真实接口、随机数约定不一致的问题。

本轮未发布、未调整版本，也未提交或推送。C15 一般 R 优化、C17 约束引导选点、C20 精确断点续跑继续暂缓。

## 三条主流程

1. **DOE → 评价 → 敏感性分析 → Reader**：覆盖真实空间和单位空间的 Saltelli 采样、预评估 Y 复用、Sobol 理论值对照、变量标签及保存读回；确认分析没有额外调用真实模型。
2. **DOE → 代理调参 → 优化 → 真实复核 → 导出**：24 个真实样本，固定训练/验证划分，RBF 三组候选加最终拟合共 4 次 fit；代理优化 48 次评价不新增真实调用，最后仅真实复核最佳点一次。检查预测精度、调参报告、OptReader 与 NPZ 导出。
3. **ModelProblem → IES → MCMC → 诊断 → Reader/导出**：包括观测 mask、被屏蔽的 NaN、IES 评价批次、推断 FE 与真实调用计数、解码结果和目标一致性；确认按需诊断不调用真实模型，混合结果目录可正确读取校准与推断记录。部分结果仍明确标记 incomplete，不能用于精确续跑。

测试文件：`tests/test_user_workflows.py`。

## 发现与修复

- 中英文 quick start 使用旧 `context.sim` / `res.sim`，实际接口为 `.sims`，复制运行会报 AttributeError。两份示例已修复。
- 英文 problem API 的 `Eval` 字段、`hasSims`、构造参数和 `target="sims"` 说明存在旧写法，已按真实协议修正。删除构造器支持传入 `simulator` 的错误描述；当前构造器在内部创建 simulator。
- 中英文代理示例依赖 `np.random.seed`，但划分器与调参器使用局部 RNG，不能据此复现。改为向 `split` / `gridTune` 显式传入 `seed=123`；新增重复执行对照。
- 更新实际运行输出中的迭代次数、停止原因和数值，以及 quick start 的章节序号。
- 文档示例纳入 pytest；wheel 验证脚本复制文档到临时测试目录，并启用 `-W error`，避免安装包验证遗漏文档流程。

最初四页文档共 36 个 Python 代码块：34 成功、2 失败。修复后执行七页中选定的 39 个可运行代码块，全部成功。这里不包含所有 API 签名、伪代码或依赖上下文的片段，也不代表已执行全部文档页面。

证据：`verification/0919-workflows-docs-before.json`、`verification/0919-workflows-docs-after.json`；可复跑脚本 `verification/review_workflow_docs.py`，`--refresh-output` 仅更新所选示例已有的输出块。示例打印的耗时不作为跨机器一致性要求。

## 验证

| 项目 | 结果 |
|---|---|
| conda py312 源码全量，`pytest -q -W error` | 1917 passed，41.10 秒 |
| 独立 wheel 全量，`-W error` | 1917 passed，77.59 秒；Python 行覆盖率 93% |
| wheel 原生扩展 | 10 个扩展全部从安装目录成功导入 |
| wheel 环境 | 从 py312 创建全新 venv，仓库外执行，清理 PYTHONPATH/PYTHONHOME；pip check 通过 |
| wheel 依赖 | NumPy 2.5.3、SciPy 1.18.1；由安装日志记录解析结果 |
| 文档执行 | 7 页中选定的 39 个 Python 代码块通过 |
| 生产代码风格 | Ruff check 通过；168 个文件格式检查通过 |

构建包：`.cache/workflow-review-wheel/uqpyl-2.1.6-cp312-cp312-linux_x86_64.whl`。
日志：`verification/0919-workflows-wheel-build.txt`、`verification/0919-workflows-wheel-test.txt`、`verification/0919-workflows-pytest.txt`。

本轮仅验证本机 Linux / Python 3.12。其他操作系统与 Python 版本需由对应 CI 执行；不据此宣称跨平台验证完成。短链推断用来检查接口和持久化，不作为统计收敛或算法性能结论。
