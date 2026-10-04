# 构建、验证与发布

CI 生成可发布的包；发布工作流只复用产物，不执行编译。生产发布继续使用 `build.yml` 和 `pypi` environment，保留已有 Trusted Publishing 的工作流路径。

## 日常与候选构建

- 分支 push 自动运行 CI；同仓库 PR 的重复运行跳过，外部 fork PR 仍运行验证。
- 手动执行 **CI** 只构建/测试，不上传 PyPI。
- Python 3.10–3.14 × Linux x86_64、Windows AMD64、macOS Intel、macOS arm64，共 20 个 wheel；macOS 两种架构各用原生 runner。
- cibuildwheel 构建、修复并安装每个 wheel，然后在仓库外执行完整 pytest，校验包版本和 10 个原生扩展。测试命令没有再次构建或安装 UQPyL。
- sdist 单独生成一次并检查元数据。它本身没有再编译一次 wheel；wheel 的完整安装验证由上述矩阵完成。
- 格式、发布校验测试、20 个 wheel 和 sdist 全部通过后，生成 `release-candidate` artifact：`dist/` 与 `manifest.json`。
- manifest 记录项目版本、提交 SHA、仓库、CI run ID、全部文件名/大小/SHA256。只允许完整的 20 个 wheel 和一个源码包。
- 产物保留 14 天。报告独立上传；失败任务可以重跑，不必重新运行已成功的矩阵。重复运行同一个 job 时允许覆盖该 job 的旧 artifact。

同一分支的新提交会取消过时的 CI。候选必须等待整个 CI 完成且成功；不能把正在运行、失败或 PR 来源的产物用于发布。

## 发布步骤

1. 确认代码、版本与变更说明已提交；`pyproject.toml` 与 `UQPyL.__version__` 必须相同。
2. 推送候选提交并等待 **CI** 完整成功，记下 Actions 页面 URL 中的 run ID。
3. 将版本标签（例如 `v2.1.7`）指向该同一提交。推送版本标签会执行发布工作流，自动寻找这个 SHA 的成功 CI 产物。
4. 发布工作流核对标签/版本、来源运行、完整矩阵和文件摘要。通过 `pypi` environment 后下载同一来源产物、再次校验，再上传。

发布不回退到重新构建。如果产物过期、缺失或候选 SHA 不同，流程会明确失败。只有此时才应为对应提交重新运行 CI。不要为了发布而修改候选提交，否则新 SHA 就需要新的验收。

## 手动验证或恢复发布

在 **Publish tested distributions to PyPI** 中：

- `release_tag`：已经存在的版本标签。
- `source_run_id`：可指定对应成功 CI 的 ID；留空则自动选择同 SHA 的成功且仍有候选产物的运行。
- `publish=false`（默认）：仅验证现有产物，不上传。
- `publish=true`：验证后上传，用于明确需要手动发布的场景。

已上传的同名 PyPI 文件不会被覆盖，也未启用静默 `skip-existing`。若上传中途失败，应先检查 PyPI 已收到的文件，再决定后续操作；不要用修改版本不一致的产物冒充原候选。

## 本地检查

```bash
conda run --no-capture-output -n py312 python -m pip install pytest packaging PyYAML
conda run --no-capture-output -n py312 python -m pytest -q .github/tests -W error --basetemp=.cache/pytest/release-gates
node --test .github/tests/test_release_run.cjs
```

本地已有 wheel 仍可用 `.github/scripts/test_wheel.py --wheel-dir ...` 在新环境安装验证。`--installed` 专用于已装入 wheel 的隔离测试环境；它会拒绝导入工作区源码或版本不匹配的包。

修改 workflow 后还应运行 actionlint。跨平台构建/测试是否成功，以 GitHub Actions 的实际结果为准。
