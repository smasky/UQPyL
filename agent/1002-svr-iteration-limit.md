# SVR 迭代上限核查（2026-10-02）

本轮核查扩展精度实验中的四次原生提示；未修改生产代码、默认迭代数或 pytest。

## 复现实验

[脚本](verification/check_svr_iteration_limit.py)、[逐拟合 JSON](verification/1002-svr-iteration-limit.json)、[摘要](verification/1002-svr-iteration-limit.txt)。py312、单 BLAS/OMP 线程、`-W error`。对同一组 192 点 LHS 数据、固定留出种子、12 个候选，分别使用 maxIter=100000/1000000；每次调参含 12 次候选拟合和 1 次全量重拟合，共 8 次调参、104 次底层拟合。用包装方法捕获每次拟合的原生 stderr、参数及样本数，没有改变求解或评分。独立 4096 测试点不参与选择。

| 案例 | 100000 上限提示位置（索引从 0 开始） | 最终候选 | 增加至 1000000 的结果 |
|---|---|---|---|
| local_peak seed=11 | 候选 7 | 1 | 无提示，最终预测逐值相同，R²=0.099908 |
| local_peak seed=23 | 候选 7 | 8 | 无提示，最终预测逐值相同，R²=0.937305 |
| local_peak seed=47 | 候选 7 | 8 | 无提示，最终预测逐值相同，R²=0.939867 |
| discontinuous seed=11 | 第 13 次拟合，即最终全量重拟合 | 7 | 无提示，R² 0.893278→0.893370，最大预测差约 0.004625 |

因此，尖峰 seed=11 的低精度不是被观察到的迭代上限造成；不能通过把 maxIter 一律增加十倍来宣称解决尖峰问题。第四例确实有最终模型触及上限，但本例精度影响小；不据此保证其他数据同样影响小。没有触发上限也不等于全局最优或普遍精度保证。

## 实现发现

- `svr/core/svm.cpp` 的 Solver 达到上限后重建必要梯度，打印 `WARNING: reaching max number of iterations`，继续计算 rho/系数并返回模型。返回近似模型本身合理。
- SolutionInfo 没有传递迭代次数/是否触及上限；pybind 的 SvmHandle 也未暴露该状态。
- `SVR.fitState` 仅保存 innerModel、symbol、kernel，没有收敛诊断。
- AutoTuner 报告把正常返回记录为 finished，没有候选/最终模型的收敛字段；这里 finished 表示拟合调用完成，不等于求解已收敛。
- 原生 fprintf 提示不是 Python warning，因此 `warnings.catch_warnings`、pytest `-W error` 无法识别。实验使用逐拟合文件描述符捕获才定位到具体阶段。

## 判断与建议

确认的是 **收敛诊断未跨原生接口传递**，不是本轮证明了 SVR 公式错误。后续若完善：由原生求解器返回迭代数及上限状态，SVR 记录状态并发 Python RuntimeWarning，AutoTuner 分别保存候选与最终重拟合诊断。默认可继续返回可用近似模型，不自动丢弃候选或强制报错，符合用户 warning 优先的约定。不能通过静默加大预算或屏蔽 stderr 替代诊断。

本轮只核查，以上诊断改进尚未实施。Ruff/格式检查通过；没有重复全量回归，上一轮 2631 passed 并非本轮新跑。未提交、推送或重建 wheel；原待处理/暂缓项保持。
