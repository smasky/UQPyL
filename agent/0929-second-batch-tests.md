# 2026-09-29 第二批测试增强：指标边界、Pareto 前沿与临时目录

## 范围与结果

落实测试审查的 T04–T06，基线为第一批完成后的 1935 项。新增 50 项校准边界测试；原来 2 个混合 Pareto 前沿测试展开为 22 项并增强断言，原检查职责保留；6 项公共运行测试改用标准临时目录。总量净增 70 项，共 2005 项。未迁移历史审查回归，不扩展 C15/C17/C20。

## T04：指标边界

[新增边界测试](../tests/test_calibration_metric_boundaries.py) 覆盖以下等价类：

- 8 个指标在空输入或全部 mask 后没有有效观测时，明确抛出 `ValueError`。
- 公共输入整理拒绝非法 simulation 维度、观测长度与 mask 长度。
- NSE/R2 的零观测方差、PBIAS 的零观测和、Pearson 的常量观测/模拟、KGE 的零均值及常量情况。
- mask 过滤后的计算与显式子集一致，掩蔽的 NaN/Inf 不参与计算，输入不被修改。
- 区间指标的空集合、长度不匹配、mask 不匹配和反向上下界；端点包含、零宽区间、常量观测及被 mask 的非法区间。
- 零均值并不影响有定义的损失、NSE 和 Pearson，避免将拒绝范围扩大到正常输入。

实测发现：`pfactor` / `rfactor` 在空集合（包含全部被 mask）上调用 NumPy 统计产生警告；反向上下界未被拒绝。首次专项共 6 项失败，全部来自这两个指标的三类无效输入。

修复在 [calibration/util.py](../UQPyL/calibration/util.py)：提取共享 `_prepareInterval`，对 mask 后的有效部分检查非空与 `lower <= upper`，无效时抛出明确 `ValueError`。保留原来对常量观测的区别：P-factor 可计算覆盖比例，R-factor 因标准差为零而报错；区间端点计入覆盖。正常输入公式未改变。

本轮没有更改所有指标的数值缩放策略，也不以该组测试证明任意浮点量级下的稳定性。KGE 常量观测会先由 Pearson 检查拒绝，不为后面的重复零标准差保护强造不自然输入。

## T05：Pareto 前沿

[前沿测试](../tests/test_problem_mop_pf.py) 从“tuple/长度/非空”提升为形状、有效数值、解析关系及前沿范围验证：

- ZDT1/4 的平方根曲线，ZDT2/6 的二次曲线，ZDT3 的带振荡项曲线。
- DTLZ1 非负平面（目标和为 0.5），DTLZ2/3/4 非负单位球面。
- DTLZ5/6 的退化曲线：单位球面、前两个目标相等及端点范围。
- DTLZ7 的第三目标关系，以及断开区域的掩蔽。
- ZDT3/DTLZ7 用独立逐对支配定义检查保留点与掩蔽点，不调用生产 `NDSort` 作为参照；分块计算避免一次生成完整三维比较数组。
- 保留原有 10 种不支持目标数的构造报错检查，独立参数化以便定位失败。

ZDT3/DTLZ7 的 NaN 用于断开绘图，不是应清除的错误值。连续前沿要求所有点有限；断开前沿明确检查有效点、NaN 位置及支配关系。此次所有前沿测试通过，没有修改问题函数或前沿生成代码，也没有降低网格密度来缩短耗时。

## T06：临时目录与 SQLite 连接

[公共运行测试](../tests/test_runtime_common.py) 删除自建 `D:/UQ/Result/_pytest_runtime_common` 加 UUID 且不清理的 fixture，实际文件改用 pytest `tmp_path`，由 pytest 按临时目录保留/清理策略管理。仅 `RunSession` 的字符串字段测试保留 Windows 路径示例，不访问该路径。

三处 SQLite 测试连接使用 `contextlib.closing`，保证断言或 SQL 失败时也关闭连接；原有写入提交语义保留。本轮未删除磁盘上的历史遗留目录。没有新增只镜像 fixture 实现的测试，沿用原有读写断言验证行为。

## 验证记录

- [首次专项](verification/0929-second-batch-before.txt)：6 failed、70 passed，5.32 秒。
- [修复后专项，含既有指标/SUFI2 回归](verification/0929-second-batch-after.txt)：82 passed，5.18 秒。随后补充零均值仍可计算与掩蔽反向区间两项，纳入最终全量验证。
- [py312 全量](verification/0929-second-batch-py312-full.txt)：启用行与分支覆盖，`-W error`。
- [Python 3.14 wheel 构建](verification/0929-second-batch-py314-build.txt) 与 [独立安装包验证](verification/0929-second-batch-py314-wheel.txt)。

py312 全量 **2005 passed，48.77 秒，零警告**。Python 行覆盖 9850 / 10518（93.65%），分支覆盖 2440 / 2992（81.55%）。`calibration/util.py` 行覆盖约 98.96%、分支约 97.22%，剩余未执行行为是上述 KGE 的重复保护；不为追求 100% 而绕过正常入口。

4 个触达 Python 文件通过 Ruff 静态和格式检查，`git diff --check` 通过。源码报告位于 `.cache/second-batch-py312/`，独立 wheel 报告位于 `.cache/second-batch-py314-wheel-test/`。

Python 3.14.7 独立 wheel **2005 passed，29.21 秒，零警告**；仓库外全新 venv 安装、`pip check` 与 10 个原生扩展导入均通过。构建使用已有 GCC/G++ 15.2，并启用 `-Werror=incompatible-pointer-types`。两次测试的解释器和覆盖选项不同，不据此比较性能。

跨平台 CI 仍需后续实际执行；不提交、推送或发布。
