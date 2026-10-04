# 2026-09-29 Python 3.14 支持

## 改动与范围

按用户要求将开发分支支持范围补到标准 CPython 3.14。版本号保持 2.1.6，`requires-python = ">=3.10"` 保持开放下界；未提交、推送或发布。

- `pyproject.toml` 增加 Python 3.14 classifier 与 `cp314-*` wheel 选择器。
- 构建依赖设为 `Cython>=3.1`、`pybind11>=3.0`；发布工具设为 `cibuildwheel>=3.2.1`。运行时依赖不变。
- 常规 CI 补齐 Linux/Windows/macOS × Python 3.10、3.11、3.12、3.13、3.14，共 15 组安装包测试。
- 发布 wheel 矩阵增加 cp314，沿用 Linux x86_64、Windows AMD64、macOS x86_64/arm64。
- 中英文 README 明确开发分支兼容目标。当前不声明自由线程 Python 3.14t 支持，选择器也不包含该构建。

工具链依据：[Cython 变更记录](https://docs.cython.org/en/latest/src/changes.html)、[pybind11 3.0 的 Python 3.14 支持](https://pybind11.readthedocs.io/en/latest/changelog.html#version-3-0-0-july-10-2025)、[cibuildwheel 变更记录](https://cibuildwheel.pypa.io/en/stable/changelog/)。本次隔离构建验证实际解析的依赖组合，不等同于所有最低构建依赖组合均已验证。

## 实测发现与修复

首次 3.14 全量安装包测试：5 failed、1911 passed、1 error，均涉及未关闭 SQLite 连接产生的资源警告。由于连接垃圾回收时点不固定，报告失败所在测试未必就是创建连接的测试。

根因为三个测试文件中的 8 处 `with sqlite3.connect(...)` 仅管理事务，未关闭连接。读取路径改为 `contextlib.closing`；写入路径使用 `with closing(...) as conn, conn:`，先提交/回滚，再关闭。未屏蔽警告、未放宽断言、未修改生产算法或资源清理代码。

- `tests/test_runtime_failures.py`：6 处只读检查连接。
- `tests/test_remaining_review.py`：1 处只读检查连接。
- `tests/test_review_a07_a15.py`：1 处建表/写入连接，保留事务语义。

## 验证结果

| 检查 | 结果 |
|---|---|
| 独立 conda Python 3.14.7，Linux x86_64 | 环境位于 `/tmp/uqpyl-py314` |
| 隔离 wheel 构建，GCC/G++ 15.2，`-Werror=incompatible-pointer-types` | 成功，生成 cp314 wheel |
| 仓库外全新 venv 安装 wheel，`pip check` | 通过 |
| 原生扩展从安装目录加载 | 10 个全部通过，wheel 不含旧 Python ABI 扩展 |
| 安装包全量 `pytest -q -W error`，含覆盖率与文档流程 | **1917 passed，28.35 秒，零警告；Python 行覆盖率 93%** |
| conda py312，三个修改测试文件，`-W error` | **136 passed，7.82 秒** |
| 版本分类、CI/发布 YAML 矩阵、构建选择器一致性 | 通过；cibuildwheel 4.2.1 dry-run 列出预期四种 cp314 平台/架构标识 |
| 包源码 `ruff check UQPyL` / `ruff format --check UQPyL`、`git diff --check` | 通过 |

最初默认 shell 找不到 gcc；随后显式使用已有 py312 环境的 GCC/G++ 15.2，Python 解释器与扩展 ABI 仍为独立的 3.14.7。没有更换或覆盖 py312 的 Python/已安装扩展。

额外对三个测试文件运行 Ruff，发现 11 条既有 E701/E702/E731 风格问题；它们不属于当前包源码风格门禁，本次仅修复资源关闭，未扩大到测试文件格式整理。

日志：[构建](verification/0929-py314-wheel-build.txt)、[首次失败](verification/0929-py314-wheel-before.txt)、[最终安装包测试](verification/0929-py314-wheel-test.txt)、[py312 回归](verification/0929-py312-compat-tests.txt)。构建产物位于 `.cache/py314-wheel/`，XML 报告位于 `.cache/py314-wheel-test/`。

复跑构建和安装包验证：

```bash
CC=/home/wmtsky/anaconda3/envs/py312/bin/x86_64-conda-linux-gnu-cc \
CXX=/home/wmtsky/anaconda3/envs/py312/bin/x86_64-conda-linux-gnu-c++ \
CFLAGS=-Werror=incompatible-pointer-types \
/tmp/uqpyl-py314/bin/python -m build --wheel --outdir .cache/py314-wheel
/tmp/uqpyl-py314/bin/python .github/scripts/test_wheel.py \
  --wheel-dir .cache/py314-wheel --report-dir .cache/py314-wheel-test
```

**边界：** 本次实际执行的是 Linux / Python 3.14.7 安装包全量与 py312 相关回归。其他操作系统以及新增 3.11/3.13 CI 组合只完成配置，待开发提交推送后执行；不能将构建标识 dry-run 计作这些平台的真实构建。发布流程与测试门禁的独立整理不在本次改动范围。
