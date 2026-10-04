# 2026-09-29 第一批关键路径补测与修复

## 范围

落实 [测试有效性审查](0929-test-effectiveness-review.md) 的 T01–T03：新增 18 项回归，没有删除或迁移原有 1917 项测试。开发与源码验证使用 conda `py312`；Python 3.14 使用独立环境构建、在仓库外安装 wheel 验证。不扩展暂缓的 C15/C17/C20，不提交、推送或发布。

## 实测发现与处理

### T01：MOASMO 高级选点

当候选总数大于请求数量，但第一非支配层小于请求数量时，原循环仍按请求数量抽取。第一层取空后调用 `argmax`，触发 `ValueError: attempt to get argmax of an empty sequence`。

修复：循环次数限制为第一层可用数量，后续沿用已有 `_novelCandidates` 补充、规范化和去重逻辑；有限离散域耗尽仍以 `no_novel_candidates` 结束。没有引入约束引导选点或新的补点策略。

新增 [test_moasmo_advanced_infilling.py](../tests/test_moasmo_advanced_infilling.py) 共 6 项：

- 几何分布已知的候选，独立检查 maximin 选点顺序。
- 第一层不足、候选总量不足、重复候选，检查实际模型评价数、边界及去重。
- 已遍历的有限离散域，不重复调用真实模型，正确停止。
- 使用真实 RBF 代理和 NSGAII 的短流程，验证真实坐标及目标结果。

### T02：ABC 停滞重置

原流程在末尾标记待重置蜂，下一轮先执行 `setEmployedBees`。当 employed bees 全部被标记为待重置时，它会重新分配角色并覆盖部分重置标记。固定种子的恒定目标短流程复现了“停滞却未执行预期重置”。

修复：每轮先重置上一轮标记的食物源，再分配 employed 角色及执行更新。保留 `limitCount > limit` 的严格阈值和既有迭代边界预算约定。

新增 [test_abc_abandonment.py](../tests/test_abc_abandonment.py) 共 4 项：需补充/无需补充 employed 两个分支，严格阈值，以及实际短流程。验证重置行数对应 FEs、计数清零、未重置行保持不变、目标对应真实参数、参数在界内。短流程包装原方法记录调用，不替代实际重置计算。

### T03：Kriging 二次趋势

有效满秩设计的基底、均值和方差通过数值验证。补测也确认：二维二次趋势需要 6 个独立基函数，但 4 个样本或共线样本此前会经最小二乘静默产生结果，无法唯一确定趋势系数。

增加契约检查：训练趋势矩阵必须满列秩，否则抛出明确 `ValueError`。检查位于固定参数拟合与超参数优化共用的初始化入口，适用于所有 KRG 趋势；它会收紧此前对欠定/秩亏输入的接受行为，而不是继续返回无法识别的趋势结果。

新增 [test_kriging_quadratic_trend.py](../tests/test_kriging_quadratic_trend.py) 共 8 项：

- 显式二维基底项对照。
- 开启/关闭标准化的已知二次函数再现，以及近零残差的不确定度。
- 非零残差下，使用独立 GLS / universal kriging 稠密求解公式对照预测均值和方差，不复用生产 QR/Cholesky 因子。
- 欠定/共线设计 × 固定参数/超参数优化，明确拒绝并清空拟合状态。

## 验证

首次 15 项专项：4 failed、11 passed，分别为 MOASMO 候选耗尽、ABC 短流程未重置、KRG 两类无效设计未拒绝，见 [修复前日志](verification/0929-key-paths-before.txt)。首次修复后 15 项全部通过，见 [专项日志](verification/0929-key-paths-after.txt)。随后增加独立非零方差参照与优化入口验证，最终新增量为 18 项，纳入全量回归。

最终源码与 wheel 验证均完成：

| 验证 | 结果 |
|---|---|
| conda py312，全量 `pytest -q -W error --cov=UQPyL --cov-branch` | **1935 passed，64.16 秒，零警告** |
| Python 3.14.7，GCC/G++ 15.2，严格指针类型检查构建 wheel | 成功 |
| 仓库外全新 venv 安装、`pip check`、10 个原生扩展导入 | 全部通过 |
| Python 3.14 独立 wheel，全量 `pytest -q -W error`，含行覆盖 | **1935 passed，62.54 秒，零警告** |
| 修改/新增的 6 个 Python 文件 Ruff 静态与格式检查、`git diff --check` | 通过 |

源码全量分支基线：行覆盖 9828 / 10523（93.40%），分支覆盖 2423 / 2994（80.93%）。MOASMO 行/分支约 98.91% / 96.43%，ABC 均为 100%，Kriging 约 95.08% / 88.64%。这证明目标路径已执行，不替代数值断言或全组合正确性证明。pytest-cov 表中的 TOTAL 91% 合并了语句与分支，不能与旧的 93% 纯行覆盖直接比较。

报告分别保存在 `.cache/key-paths-py312/` 和 `.cache/key-paths-py314-wheel-test/`。两个环境使用不同覆盖工具/解释器，行数统计可能不同；本次并行构建、测试且源码额外启用分支追踪，耗时不作为性能变化结论。T07 已建立首次分支基线，但 CI 配置及后续缺口梳理未在本轮扩展。

- [py312 全量日志](verification/0929-key-paths-py312-full.txt)
- [Python 3.14 wheel 构建日志](verification/0929-key-paths-py314-build.txt)
- [Python 3.14 独立 wheel 全量日志](verification/0929-key-paths-py314-wheel.txt)

修改的 3 个生产文件与新增的 3 个测试文件通过 Ruff 静态检查和格式检查。跨平台 CI 仍待后续执行，本次不宣称 Windows/macOS 验证完成。

复跑源码验证使用 `conda run --no-capture-output -n py312 pytest -q -W error --cov=UQPyL --cov-branch`。wheel 构建沿用 [3.14 支持记录](0929-python314-support.md) 的编译器环境变量，将产物目录设为 `.cache/key-paths-py314-wheel`，再调用 `.github/scripts/test_wheel.py --wheel-dir .cache/key-paths-py314-wheel --report-dir .cache/key-paths-py314-wheel-test`。
