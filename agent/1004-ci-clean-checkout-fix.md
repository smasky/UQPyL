# 2026-10-04 CI 全新工作目录修复

失败运行：https://github.com/smasky/UQPyL/actions/runs/37203470382

style job 的 Ruff 格式/静态检查已通过；真正失败发生在 `pytest .github/tests --basetemp=.cache/pytest/release-gates`。新 checkout 没有 `.cache/pytest`，pytest 仅创建 basetemp 本身，不递归创建父目录。原 `tests/conftest.py` 已处理此问题，但 `.github/tests` 不在它的作用域内。实际 6 passed、19 errors，矩阵尚未开始。

修复为 `.github/tests/conftest.py` 在 pytest_configure 阶段创建 basetemp 的父目录。保留 pytest 对临时目录自身的管理、全部检查和测试断言，没有改算法、跳过测试或放宽校验。

在 git archive 导出的无缓存副本中复现同样 19 errors；只加入新 conftest 后，从另一个无缓存副本运行与 CI 相同的 Ruff 检查、发布校验测试和 Node 测试，得到 25 Python + 13 JavaScript 全部通过。见 [修复前](verification/1004-ci-clean-before.txt)、[修复后](verification/1004-ci-clean-after.txt)。使用 conda py312 解释器。生产源码未变，因此未重复 3175 项源码全量。

之前本地使用已存在的目录或 `/tmp` 父目录，未覆盖该干净 checkout 场景；前轮本地验证不能代替远程 CI。修复提交推送后仍须等待完整矩阵，不创建发布标签。
