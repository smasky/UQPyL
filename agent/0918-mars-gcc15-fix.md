# 2026-09-18 MARS / GCC 15 编译兼容性修复

## 修复前基线

- 在修改生产代码前创建本地提交 `993d9f5`，分支 `baseline/mars-before-gcc15-fix` 指向该提交；保存了此前环境、六组 CI 和 wheel 验证记录。
- 该提交的生产源码仍与 `f07f058` 一致。基线独立 wheel 为 1345 passed、0 skipped、4 warnings。
- 修复前 wheel 保存在 `.cache/mars-baseline/uqpyl-2.1.6-cp312-cp312-linux_x86_64.whl`，SHA256：`9ca29bd8642f49d61f92e9acf7517f532db4c8db81b947ef11790cbae370a986`。
- 基线依赖列表与数值快照分别在 `.cache/mars-baseline/pip-freeze.txt`、`numerical.npz`。这些缓存不会随 Git 提交；源码基线由上述本地分支保留。
- 保存基线时仅本地提交；后续修复及基线分支均已推送，远程验证见文末。本轮没有正式发布。

## 根因与最小修复

在 Cython 3.3.0 的本次生成代码中，`@cython.final` 修饰的 `MissingnessBasisFunction` 覆写带默认参数的 `cpdef apply` 后，其 Python 包装函数通过基类 vtable 调用，却传入派生类实例指针和派生类默认参数结构指针，触发两处不兼容指针类型诊断。

同一 GCC 15.2 下，基线代码在默认诊断和 `-Werror=incompatible-pointer-types` 下均失败；仅移除该类的 `@cython.final` 后，两种编译检查均成功。[编译对照](verification/0918-mars-gcc15-compile-check.json)、[基线错误](verification/0918-mars-gcc15-before.txt)。删除 `.pxd` 中重复声明不能修复此错误，未纳入改动。

生产修复只移除该类的 final 标记并添加原因说明，使用正常虚分派。方法签名、默认参数、递归调用、缺失值计算及 MARS 拟合公式均未修改；没有编辑生成的 C 文件或添加强制指针转换。
该内部类不再禁止 Python 子类化，这是移除 final 标记的实际语义变化。未进行独立性能基准，不声称运行开销完全相同。

构建门禁同步：

- Linux CI 的 wheel 构建启用 `-Werror=incompatible-pointer-types`。
- Linux 发布构建从 `-Wno-error=incompatible-pointer-types` 改为严格检查。
- 不新增依赖，不改版本号，不触发发布。

## 验证

1. 新增 12 项回归，覆盖 Python 直接调用、默认/位置/关键字参数、`recurse=False`、非连续输出数组、互补 mask、`Basis.transform` 的 C 层分派和 pickle 后的递归调用。修复前 12 项全部通过，修复后随全量通过；编译回归由上述严格构建检查捕获。
2. conda `py312` 使用 GCC/G++ 15.2、Cython 3.3.0，隔离构建完整 wheel 成功，显式启用严格指针检查。[构建日志](verification/0918-mars-gcc15-build.txt)。
3. 独立 wheel 在仓库外临时 venv 安装；`pip check` 和全部 10 个原生扩展导入检查通过；完整测试 **1357 passed、0 skipped、4 warnings、35.59 秒**。4 条仍为 FAST 辅助频率复用警告。[完整日志](verification/0918-mars-gcc15-wheel-test.txt)。
4. 本机可编辑安装也使用 GCC 15.2 严格重建；依赖检查通过，本机 MARS 专项 **14 passed**。[安装日志](verification/0918-mars-gcc15-editable.txt)、[专项日志](verification/0918-mars-local-tests.txt)。
5. 8 组固定数据案例（两个 seed × 平滑开关 × 剪枝开关，双输入双输出），逐项比较预测、系数和基函数矩阵，共 24 个数组，最大绝对差 **0.0**。[比较结果](verification/0918-mars-numerical-comparison.json)、[可复现脚本](verification/mars_numerical_check.py)。这仅证明这些案例一致，不代替所有输入与平台的数学证明。

运行命令：

```bash
conda run -n py312 env CFLAGS=-Werror=incompatible-pointer-types python -m build --wheel --outdir .cache/mars-gcc15-wheel
conda run -n py312 python .github/scripts/test_wheel.py --wheel-dir .cache/mars-gcc15-wheel --report-dir .cache/mars-gcc15-wheel-test
conda run -n py312 env CFLAGS=-Werror=incompatible-pointer-types python -m pip install -e . --no-build-isolation --no-deps
conda run -n py312 python agent/verification/mars_numerical_check.py compare .cache/mars-baseline/numerical.npz
```

## 边界与接续

后续已将修复提交为 `c38078e` 并推送 `dev`，基线分支也已推送。新提交的 [CI run 35356942866](https://github.com/smasky/UQPyL/actions/runs/35356942866) 六组 Linux/Windows/macOS × Python 3.10/3.12 全部成功，[逐项证据](verification/0918-mars-fixed-ci-jobs.json)。MARS 修复已完成本机与远程验证，没有正式发布。
D01 已完成 [优化器实测对照](0918-d01-optimizer-comparison.md)，建议保留 Boxmin 默认；D02（预算规则）保持原状。
