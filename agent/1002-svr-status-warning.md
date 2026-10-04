# SVR 原生提示改为 Python warning（2026-10-02）

用户确认取消 C++ 提示后，完成状态透传，不静默隐藏迭代上限。

- Solver 保存迭代次数、是否用尽预算，经 decision_function → svm_model → pybind handle 传给 Python。达到预算只表示触及上限，未声称最终一步一定不满足容差。
- 删除原生 `fprintf(stderr, ...)`。保留原近似求解结果，没有修改优化步骤、参数、默认 maxIter 或评分方式。模型加载时迭代数标为 -1（历史文件没有诊断信息），本 Python 接口只暴露训练模型。
- `SVR.fitState['solver']` 提供 `iterations`、`maxIterations`、`iterationLimitReached`；上限触及时发 RuntimeWarning，默认继续预测。用户将 warning 升异常时仍使公共 fit 失效。
- AutoTuner 按候选与最终重拟合分别保存 solver，导出字段 `iterations`、`max_iterations`、`iteration_limit_reached`。调用完成仍是 finished，不冒充已收敛，也不自动剔除近似候选。

示例：

```text
RuntimeWarning: SVR reached maxIter=100000; returning an approximate model. Convergence is not established; inspect fitState['solver'].
```

## 验证

- 重建 py312 的 SVR 原生扩展，使用该 conda 环境编译器；最终构建成功、无编译警告。未重建其他扩展或 3.14 wheel。
- 新增 5 项测试，相关 **42 passed，0.81 秒**：两类 SVR 达上限、原生 stderr 为空、近似结果可预测、重拟合清理状态、两种调参模式分阶段诊断、warning 升异常失效。
- 原 4 个案例×两档预算，共 8 次调参/104 次拟合复核：**原生消息 0，Python warnings 4，上限状态 4**。8 次的选中候选、验证 R²、独立测试 R² 与修改前逐值相同。
- 修改前记录保留；新输出 [after JSON](verification/1002-svr-iteration-limit-after.json)、[摘要](verification/1002-svr-iteration-limit-after.txt)。复核脚本已捕获并保存预期 warning，没有屏蔽诊断。
- 最终 py312 **2636 passed，56.53 秒，`-W error` 零未捕获警告**，见 [全量日志](verification/1002-svr-status-full.txt)。4 个触达 Python 文件 Ruff/格式检查和 git diff --check 通过。
- 中英文 API、测试导航、TODO/交接同步。未提交/推送。其他精度局限及待处理/暂缓项保持。
