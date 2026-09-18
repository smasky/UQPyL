# 2026-09-18 构建与 CI 验证

后续更新：本文件保留修复前的验证状态；GCC 15 问题已完成本地修复和验证，最新结论见 [MARS 修复记录](0918-mars-gcc15-fix.md)。

## 结论

基线 `f07f0588e6dce00ca34405c962572fc697e4578b`，本地及远程 `dev` 一致。
远程六组 CI 已通过；本机独立 wheel 在仓库外的全新环境中通过完整测试。
GCC 15 的编译兼容性问题已确认，未在本次验证中修复。

## 远程证据

[CI run 35344776057](https://github.com/smasky/UQPyL/actions/runs/35344776057)。
逐项检查以下六个任务的状态，以及 wheel 构建、干净环境测试步骤，均为 `success`：

| 平台 | Python 3.10 | Python 3.12 |
|---|---|---|
| ubuntu-latest | success | success |
| windows-latest | success | success |
| macos-latest | success | success |

[任务与步骤状态](verification/0918-ci-jobs.json)。本次未重新触发 CI、推送或发布。
初始连接器按提交查询返回空列表，公开 API 部分请求限流；随后通过直连取得准确 run ID，并通过连接器取得六个 job 的结果，最终结论以这些实际记录为准。

## 本机独立 wheel

使用 conda `py312`（Python 3.12.0）与 GCC/G++ 11.2：

```bash
conda run -n py312 python -m build --wheel --outdir .cache/handoff-wheel
conda run -n py312 python .github/scripts/test_wheel.py --wheel-dir .cache/handoff-wheel --report-dir .cache/wheel-test
```

- 产物：`.cache/handoff-wheel/uqpyl-2.1.6-cp312-cp312-linux_x86_64.whl`。
- 构建使用隔离依赖；测试脚本创建临时 venv，清除 PYTHONPATH/PYTHONHOME，在仓库外安装 wheel 并复制测试。
- 确认 UQPyL 从临时 venv 的 site-packages 导入，10 个原生扩展全部成功导入。
- `pip check` 通过。
- **1345 passed，0 skipped，4 warnings，34.08 秒**；JUnit 确认 errors/failures/skipped 全部为 0。
- 4 条警告均为既有 FAST 辅助频率复用警告；覆盖率汇总 91%。
- [构建日志](verification/0918-wheel-build.txt)、[安装及完整测试日志](verification/0918-wheel-test.txt)；JUnit 与 coverage XML 位于 `.cache/wheel-test/`。

## GCC 兼容性边界

环境配置时 GCC 15.2 构建失败，报错位置为 MARS `_basis.pyx` 中 `MissingnessBasisFunction.apply` 对应的 Cython 生成代码。
本次 GCC 11.2 构建同一代码仍有两处 `incompatible-pointer-types` 警告：派生类实例指针，以及带默认参数的参数结构指针，与基类函数签名不匹配。
以本次生成的 `_basis.c` 执行 GCC 11.2 `-fsyntax-only -Werror=incompatible-pointer-types`，同样失败，见[严格类型编译复现](verification/0918-mars-strict-compile.txt)。
因此问题不只是本机缺少工具链：当前生成代码无法通过严格指针类型诊断。`.github/workflows/build.yml` 的 Linux 发布构建已带 `-Wno-error=incompatible-pointer-types`；本机验证未添加该参数。
本次没有确定 Cython 上游与项目声明各自的责任，也没有完成源码修正。后续应单独处理生成代码的类型兼容性，并用 GCC 15 重新构建和测试；不应仅靠降级编译器就将此问题关闭。

这不否定本次已验证的六组 CI 和 GCC 11.2 wheel，但不代表 Python 3.11/3.13、所有编译器或正式发布产物均已验证。
