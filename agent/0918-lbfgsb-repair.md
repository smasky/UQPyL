# LBFGSB 返回协议修复与保留评估

用户要求先检查，可修复则保留，不可行则移除。结论：保留为显式可选优化器，默认继续使用 Boxmin。

## 复现与原因

从已保存的 D01 实验恢复 180 个 LBFGSB 起点与训练目标，分别测试默认 options、maxls=50、eps=1e-6。生产基线为 `32dc639`；使用 conda py312、单线程 BLAS。

原包装器直接返回 SciPy 的 `res.x/res.fun`，没有检查状态或保留已评价的最佳候选。原始默认配置 41/180 次异常停止，27 次返回分数与返回点复算目标不一致；均为异常停止。检查本机 SciPy `_minimize_lbfgsb` 可见退出后直接组装当前 x 和 f，未额外复算目标。这里确认的是实际返回协议问题，不宣称已经证明每次线搜索失败的数值根因。

[SciPy 官方文档](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html) 说明 `maxls` 控制每次迭代的最大线搜索步数，`eps` 控制数值梯度的绝对差分步长。增加 maxls 在本组案例有效，改变 eps 没有稳定消除失败，不将调参等同于根因修复。

| 配置 | 修复前异常停止 | 修复前点/分数不一致 | 修复后异常停止 | 修复后点/分数不一致 |
|---|---|---|---|---|
| SciPy 默认 | 41/180 | 27/180 | 41/180 | 0/180 |
| maxls=50 | 2/180 | 1/180 | 2/180 | 0/180 |
| eps=1e-6 | 39/180 | 10/180 | 39/180 | 0/180 |

## 修复范围

- 跟踪实际评价过的有限、界内候选，保存点的副本，返回最佳点与其对应评价值；包括差分探测点。复算有效的 SciPy 返回点，避免依赖其可能过时的 fun。额外复算至多增加一次目标调用，SciPy 自身 nfev 不包含这次调用。
- 暴露 `lastResult` 保留 SciPy 原始停止状态，不把异常停止伪装成成功。仅表示最后一次 run，不能代替多次重启的完整历史；其中 x/fun 不保证就是包装器选出的点/分数。
- 增加有限有序边界、起点维数和有限性检查；没有有限候选时报错。每次 run 清理旧状态，不吞掉用户回调异常。
- 要求确定性目标；没有为随机噪声目标提供一致性保证。保留原数值搜索选项，不强制增加 maxls，也不替换成另一种算法。
- 默认与两种调参配置合计 540 次修复后复跑，返回目标不一致为 0，劣于起点为 0。异常停止数保持不变是预期行为：修复保证可用候选的返回协议，不承诺收敛。

## 验证和复现

- 104 项专项通过，覆盖边界、随机隔离与复现、真实 GPR/KRG 拟合、真实提前停止、旧分数、可变数组复用、无有限候选、状态重置及非法输入。
- [复跑脚本](verification/lbfgsb_return_audit.py)、[修复前记录](verification/0918-lbfgsb-before.json)、[修复后记录](verification/0918-lbfgsb-after.json)。修复前记录由修改包装器前启动的独立进程生成；用当前代码重跑会得到修复后的语义。
- 运行命令：`conda run --no-capture-output -n py312 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python agent/verification/lbfgsb_return_audit.py /tmp/lbfgsb-audit.json`。
- 这是固定合成数据的诊断，不能推断所有核、真实数据与大样本上的收敛率；没有必要据此移除 LBFGSB。

全量验证：**1409 passed，4 条原有 FAST warnings，38.06 秒**。见 [日志](verification/0918-lbfgsb-pytest.txt)。本轮修改尚未提交推送或验证远程 CI。
