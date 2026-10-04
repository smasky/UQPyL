# 2026-10-04 构建与发布去重实施

按用户授权修改本地工作流，解决 CI 与标签发布重复构建的问题；没有推送、触发 Actions、创建标签或上传 PyPI。观测形状迁移未在此任务中实施。

## 最终流程

分支/手动 CI → 先格式与发布校验测试 → 20 个正式规格 wheel 的构建/修复/安装测试 + 1 个 sdist → 完整矩阵与元数据检查 → release-candidate artifact → 标签/手动发布只下载、校验、上传。

- CI 的 Python 范围为 3.10–3.14，每版 Linux manylinux x86_64、Windows AMD64、macOS x86_64 和 arm64。两个 macOS 架构使用 macos-15-intel / macos-15 原生 runner。
- Linux 保留 incompatible-pointer-types 编译检查，cibuildwheel 固定 3.2.1。移除 pyproject 中冗余 before-build pip 安装，构建依赖仍由 build-system.requires 的隔离环境提供。
- test_wheel.py 新增 --installed，复用 cibuildwheel 已安装的待测 wheel，不再次安装/构建 UQPyL。保留原 --wheel-dir 新建环境模式。
- 两种测试模式均在工作区外复制测试/文档并执行；明确检查版本、site-packages 位置和 10 个原生扩展，然后执行完整 pytest/覆盖率。
- build.yml 保留原路径与 pypi environment，便于继续使用现有 Trusted Publishing。它已没有 build/cibuildwheel 步骤。
- 标签发布自动寻找同一 SHA 的成功 CI；手动可指定 source_run_id。只接受本仓库 ci.yml 的 push/manual 成功运行，拒绝 PR、错误提交、其他工作流、缺失/过期产物。
- manifest 记录版本、SHA、仓库、run ID、文件名/大小/SHA256，并验证 20 个 wheel 目标和一个 sdist。发布前检查 tag 与两处包版本一致，下载后再次验证所有文件及元数据。
- 手动发布 publish 默认 false，仅验证；真正试构建应手动运行 CI。无候选产物时明确停止，不偷偷再编译。
- CI 对同仓库 PR 使用分支 push 的结果，fork PR 仍执行检查；不额外执行同仓库 PR 的 merge-ref 测试。依赖 merge-ref 检查的分支保护需要保留对应独立检查，不能将此去重规则当作 merge-ref 验证。
- 新提交取消同分支旧 CI；发布 concurrency 不取消上传。产物保留 14 天，同 job 重跑可替换自身旧产物；失败后可只重跑失败 job。

本地旧流程在“分支 CI 后发布同一提交”时为 15+20 个 wheel 构建；新流程一次完整 CI 为 20 个，发布新增 0 个。日常 CI 从 15 个目标增加到 20 个，是补齐 macOS 双架构原生测试，不是承诺每次开发 push 更快。不同提交、不同 ABI/架构及过期产物依然需要新构建。

## 验证

- 发布文件/协议/测试驱动专项：**25 passed，0.18 秒**；[日志](verification/1004-release-gate-tests.txt)。包括完整矩阵、身份不符、缺包、摘要变化、嵌入版本错误、普通 Linux wheel 拒绝、重复目标、标签不符、两个测试入口不二次安装/构建。
- CI 运行选择：**13 passed**；[日志](verification/1004-release-run-tests.txt)。测试错误 SHA/状态/事件/工作流/仓库、非法 ID、过期产物、旧同 SHA 有效产物回退及无候选停止。
- actionlint 1.7.7 验证两个 workflow 通过，见[记录](verification/1004-release-actionlint.txt)；未调用 shellcheck/pyflakes。Ruff lint/format 和 git diff --check 通过。
- 在独立源码副本构建本地 **2.1.7 cp312 linux_x86_64** wheel，避免覆盖工作区扩展；[构建日志](verification/1004-release-wheel-build.txt)。归档包含 LICENSE.md。
- 新 venv 安装该 wheel 后，实际运行 --installed：**3094 passed，160.91 秒，-W error**，10 个原生扩展导入成功；[日志](verification/1004-release-installed-test.txt)。这轮安装包带 coverage，不能与此前无 coverage 的源码约 100 秒直接作性能比较。

本地机器没有 Docker、Windows/macOS runner，因此未执行完整 cibuildwheel 跨平台矩阵。此处本地 linux wheel 只用于验证新测试驱动，不是可直接替代 manylinux 的发布候选，完整矩阵校验会拒绝它。源码包做构建与元数据检查；没有为了测试 sdist 再编译另一份 wheel。

## 操作说明与剩余事项

详见 [.github/RELEASING.md](../.github/RELEASING.md)。所有发布脚本测试置于 .github/tests，在 CI style 前置关卡执行，不计入库算法测试的 3094 项。

首次需要推送这些工作流改动，等整个 CI 成功，再将版本标签指向同一提交。新的 workflow job 名称及同仓库 PR 去重方式应与实际分支保护规则对应；本轮未读取/修改远程仓库保护设置。已有 PyPI 文件不能覆盖，未配置 skip-existing；部分上传失败时需先核对远程已有文件。

参考官方说明：
- [cibuildwheel 测试和构建选项](https://cibuildwheel.pypa.io/en/stable/options/)
- [GitHub 原生 runner 架构](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
- [PyPA 产物与发布分离](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
