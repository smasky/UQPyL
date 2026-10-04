# 2026-10-04 Python 3.13+ 与 macOS Intel CI 修复

失败运行：https://github.com/smasky/UQPyL/actions/runs/37205000544 （CI #44）。上一轮临时目录修复已生效，style/sdist 通过，20 个 wheel 中 9 个通过、11 个失败。

## 已确认原因

- `testReaderRejectsOldObservationSchema` 用 `with sqlite3.connect(...)` 提交事务，却未关闭连接。Python 3.13 起会报告 ResourceWarning，延后 GC 使错误栈落在优化等无关测试上。所有 3.13/3.14 任务受影响。改成 `closing(...)` 外层关闭、连接自身上下文内层提交。Python 3.14 stdlib 最小复现：旧写法 1 条 unclosed database 警告，修正后 0 条。保留 `-W error`。
- macOS Intel 的 Lasso 重复拟合预测误差：float64 最大 1.78e-15、float32 最大 9.54e-7；输入/父数组/训练数据的精确隔离检查均通过。仅预测比较改为 4×对应 dtype epsilon 的相对容差，不改变输入隔离或解析解测试。
- macOS Intel 的 KRG 两个测试把数值计算当成逐位相同：形状协议预测差约 1.14e-9，MultiSurrogate 重复预测差约 1.69e-9。形状协议中密集 Gaussian 相关矩阵在 py312 测得条件数 1.53e15，独立线性真值误差 3.62e-8。将这两个预测等价比较设为 1e-8，同时新增每次预测对独立线性/正弦余弦真值的绝对误差 <=1e-6 检查。随机流一致与相互独立仍精确比较，其它模型形状比较仍为 1e-12。

未修改算法、跳过测试或过滤警告。日志支持浮点舍入及病态放大解释，但不能单凭日志断言某个 BLAS 厂商实现存在缺陷；不因此修改构建后端。

## 验证

conda py312 专项 182 passed（2.68 秒，`-W error`）；Ruff 与差异检查通过。完整回归 **3175 passed，107.96 秒，`-W error --strict-markers`**，见 [全量日志](verification/1004-ci-portability-full.txt)。远程新矩阵待修复提交后验收；不能把本地通过当成 macOS/Windows 通过。

SQLite 官方说明：https://docs.python.org/3.13/library/sqlite3.html （Connection context manager 不自动关闭；3.13 新增未关闭连接 ResourceWarning）。

## CI #45 后续核查

运行：https://github.com/smasky/UQPyL/actions/runs/37206439214 ，提交 e3391de。此前三处浮点比较与 SQLite 清理已在多平台通过；Intel macOS Python 3.14 全部 3175 项通过。Intel macOS Python 3.11 新暴露 ASMO 测试末尾 KRG 预测逐位比较（差 1.64e-11），实际评价点与最优目标值精确一致。

后续只将该预测比较设为 rtol/atol=1e-8，保留选点与最优值精确检查。检查所有引用 KRG 的测试发现重启测试也对两次拟合预测作逐位比较，改为 rtol=1e-10/atol=1e-12；重启起点/种子精确检查保留。同一模型未重新拟合的状态隔离精确检查不变。专项 **45 passed，1.07 秒，-W error**，见[日志](verification/1004-ci-asmo-followup.txt)；Ruff 通过。未修改生产源码，不重复本地全量，修复提交后的远程完整矩阵仍须验收。

CI #45 最终 18/20 wheel 通过；Intel macOS 3.10 与 3.11 均仅 ASMO 同一预测断言失败（各 3174 passed、1 failed），candidate 因此跳过。没有其它失败。
