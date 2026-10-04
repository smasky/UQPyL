# 2026-10-04 GitHub / PyPI 流程检查（未修改工作流）

已读取 GitHub main/dev 的原始 build.yml、ci.yml，与本地工作区比较。远程快照存于 verification/1004-remote-{main,dev}-{build,ci}.yml。GitHub API 返回匿名请求限流，因此未验证 Actions 历次运行耗时/状态；以下是配置确定的行为，不是对历史账单的估计。

## 发现

- 远程 main/dev 发布矩阵仍为 Python 3.10–3.13；CI 仅 3.10/3.12 × 三系统。本地未推送版本已扩展至 3.14，CI 为 3.10–3.14 × 三系统。
- ci.yml 通过 python -m build 生成 wheel 并交给 test_wheel.py 测试；build.yml 在 tag 或手动触发时通过 cibuildwheel 重新构建，没有从 CI 下载复用产物。
- 按本地矩阵，先推分支再推同一提交的 tag，通常是 15 个 CI wheel 构建加 20 个发布目标 wheel（Linux x86_64、Windows AMD64、macOS x86_64/arm64，各 5 个 Python），另有一次 sdist 构建。这里的 35 是目标构建次数，不意味着有 35 个完全可互换的文件。
- Linux 普通 CI 的 native linux wheel 与发布流程 auditwheel 修复后的 manylinux wheel 不等价。不能直接删除发布构建，再把旧 CI 产物全部原样上传。
- 发布任务仅 needs build_wheels/build_sdist；没有测试依赖，未配置 cibuildwheel test-command。测试通过的 CI wheel 并非最终发布产物。
- workflow_dispatch 同样进入 publish，并没有 build-only 开关。若为试构建手动执行，构建成功后会尝试上传。
- 没有发布前的 tag/包版本一致性及已发布版本预检。没有 concurrency 管理；同仓库分支 push 与 PR 事件也可能分别触发 CI。
- test_wheel.py 本身没有二次构建 UQPyL：它安装指定 wheel，在仓库外环境运行测试。重复来自两个工作流的构建职责。

## 建议流程

同一发布候选 SHA → 构建可发布 wheel + sdist → 安装并测试实际 wheel → 保存产物和验证记录 → 发布任务仅下载验收过的产物并上传。

1. 将发布规格构建与测试合并为一条产物流水线。Python 3.10–3.14，Linux manylinux、Windows、macOS 两种架构；测试应在可运行对应架构的环境完成，不把交叉编译成功当作运行验证。
2. 普通 PR 可以保留适当的开发验证矩阵；发布候选使用完整矩阵。矩阵中平台/ABI不同是必要构建，不是应删除的重复。
3. 发布按显式 run ID + commit SHA 复用完整成功的发布候选产物；校验版本、SHA、矩阵完整性、测试状态和文件摘要。产物过期或提交改变才重建。仅抽出 reusable workflow 并不能阻止同一提交被执行两遍。
4. 手动构建默认只构建/测试。发布入口不执行编译；保留 pypi environment 和 Trusted Publishing，并检查 tag 与 2.1.7 元数据一致。
5. 开发 CI 可取消同一分支过时运行、避免同仓库 PR 与 push 双跑；发布不要中途取消上传。失败后只重跑失败 job，不重新执行成功矩阵。

官方依据：
- https://cibuildwheel.pypa.io/en/stable/options/ （test-command、构建选择与架构）
- https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/ （构建产物与发布 job 分离）

本轮完成检查并给出具体去重方案，未编辑 workflow、未触发 GitHub Actions、未推送或上传 PyPI。形状协议迁移仍为之前暂定方案，没有在此任务中实施。
