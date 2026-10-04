# 2026-09-29 CI 分支覆盖与必需组件导入门禁

## T07：配置与本地验证完成，远程新矩阵待运行

[test_wheel.py](../.github/scripts/test_wheel.py) 的独立安装包测试现在启用 `--cov-branch`。保留原有仓库外安装、`pip check`、原生模块路径/ABI 检查及 `-W error`。

新增 `coverage-summary.json` 与 `coverage-summary.md`，分别记录行与分支的覆盖量、总量和比率。JSON 包含解释器版本、平台和 GitHub 提交 SHA；本地无 GitHub SHA 时记为 null，不把工作区误归到旧提交。GitHub 环境下向 Job Summary 追加可读表格。

覆盖 XML 的文件位置继续重写到仓库目录，支持 Linux/macOS 和 Windows 分隔符。缺少实际行/分支统计会明确失败；不设置任意百分比门槛。每次执行前清除该脚本管理的旧 XML/汇总文件，避免安装失败后上传上次的成功报告。

[CI 工作流](../.github/workflows/ci.yml) 保存每组矩阵的 XML、JSON、Markdown；Codecov 用 OS flag 区分平台，上传名称含 OS/Python。可基于同一矩阵组合的历史产物观察趋势，行/分支分开比较，不把 pytest-cov 合并百分比误当纯行覆盖。没有增加新的运行时依赖，也没有调整发布工作流。

配置检查确认 Linux/Windows/macOS × Python 3.10–3.14 共 15 组，产物路径正确。这是 YAML/配置验证，不是 Windows/macOS 的执行结果。

## T08：必需组件缺失明确失败

清理 6 个测试文件中的 7 处跳过调用：MARS、SVR、Lasso 改为直接导入或正常运行检查，敏感性分析的 MARS 缺失会明确断言失败。正式 wheel 仍先逐一验证 10 个原生扩展，不能由 skip 代替成功。

GP 的 `c_kernel_` / `dot_kernel_` 是 Python 模块，其测试改为普通导入并检查公开类，不再称为可选原生扩展。两个带 `optional` / `if_available` 的测试名称随职责更新，总数不变。本轮没有修改生产包的可选导入策略，现有“可选模块缺失”和“非预期导入错误应传播”的回归仍保留。

在子进程中注入 MARS、SVR、Lasso 原生扩展缺失，三个代表文件均以 pytest 收集错误退出（exit code 2），没有跳过。不会删除或破坏开发环境中的实际二进制文件。

## 验证结果

| 检查 | 结果 |
|---|---|
| py312 源码全量，`-W error` | **2005 passed，70.01 秒，零警告** |
| Python 3.14.7 独立 wheel，使用本轮修改后的 CI 脚本 | **2005 passed，60.51 秒，零警告** |
| 独立安装依赖、10 个原生扩展来源/ABI | 通过 |
| 实际生成覆盖 XML、JSON、Markdown 与模拟 Job Summary 文件 | 通过；行覆盖 **93.64%**，分支覆盖 **81.55%** |
| 汇总函数的 4 种文件路径、精确比率、摘要追加和缺少分支拒绝 | 通过 |
| 原生组件缺失的三个负向验证 | 全部明确失败，零跳过 |
| 触达脚本/测试 Ruff 检查、格式整理，`git diff --check` | 通过 |

生产代码未改变，因此复用第二批构建的 cp314 wheel，仅将当前测试和最新 CI 脚本用于新的独立安装验证。两个全量任务并行执行，耗时不用于判断性能退化。

验证工具：[verify_wheel_quality_gates.py](verification/verify_wheel_quality_gates.py)。日志：[汇总与负向检查](verification/0929-wheel-quality-gates.txt)、[缺失组件结果](verification/0929-required-import-failures.json)、[py312](verification/0929-ci-quality-py312.txt)、[3.14 wheel](verification/0929-ci-quality-py314-wheel.txt)、[覆盖摘要](verification/0929-ci-quality-coverage.json)、[矩阵配置检查](verification/0929-ci-matrix-check.txt)。

## 远程状态核实

2026-09-29 查询 GitHub Actions：最新成功运行仍为 [CI #42](https://github.com/smasky/UQPyL/actions/runs/35411252001)，2026-09-19，提交 `e4a6aad258aada37a6956e374a7e2dd4d2e7e4be`。页面列出已完成的 6 组，即三个 OS × Python 3.10/3.12，及对应六份 wheel-test 产物。[API 摘要快照](verification/0929-ci-remote-runs.json) 保留查询结果。

当前工作区包含这些旧提交之后的改动；新的 15 组矩阵尚未推送执行，不能把旧运行的成功视为新改动通过。T07 的实现与本地验证已完成，远程矩阵验收仍待后续推送后执行；T08 已完成。本轮未提交、推送或发布。

科学/规模定期验证与重复准备代码精简仍为后续独立计划，C15/C17/C20 继续暂缓。历史迁移 AST/node-ID 校验是迁移时快照；本轮有意调整两个测试名称和导入职责后，不应将旧映射作为当前永久门禁。
