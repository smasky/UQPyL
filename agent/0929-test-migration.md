# 2026-09-29 历史回归按模块归类

## 改动

将 10 个 `test_review_*` / `test_remaining_review.py` 中的 **123 个测试函数、488 个参数化后用例**，迁移到 34 个按模块和行为命名的文件。保留当前平铺目录，不将整个测试树改为包或改变 pytest 导入模式。文件增多用于分开原来混在一起的模块职责，不代表新增测试或重复执行。

完整文件导航见 [tests/README.md](../tests/README.md)，其中列出每个迁移文件的用例数和历史来源。每个测试函数上方也保留原文件名、函数名，A/B/C 问题批次仍可追溯。原有其他专项测试及其位置不变；本轮没有修改生产代码、删除断言、合并用例或改变参数矩阵。

两处公共准备代码从测试文件中独立出来：

- [optimization_test_support.py](../tests/optimization_test_support.py)：`METHODS`、`MULTI`、`QUIET` 和 `makeMethod` 从停止语义测试提取。停止语义与配置恢复都从此处导入；工厂实现保持不变，每次创建新的算法实例。
- [algorithm_capability_test_support.py](../tests/algorithm_capability_test_support.py)：共用的问题工厂 `makeProblem`，供优化能力/停止原因和推断停止原因持久化使用，避免拆分后复制辅助函数或让测试文件互相导入。

配置恢复测试函数内只有一处有意的代码变化：导入来源从 `test_optimization_stopping_semantics` 改为 `optimization_test_support`。没有改动该测试的断言和参数。

## 等价性证据

迁移前以当前工作区的 2005 项收集结果为基线，保留了此前未提交的修复。没有从 Git HEAD 取旧版测试覆盖当前文件。

[机器映射](verification/0929-test-migration-map.json) 记录旧文件 SHA256、123 个函数的旧/新位置与 AST 指纹、依赖声明与辅助工厂指纹、488 个完整参数化 node ID 对应关系，以及迁移后的全量预期 node ID。不是仅凭总数相同判断没有遗漏。

[校验脚本](verification/verify_test_migration.py) 检查：

1. 所有迁移函数及停止语义文件中的原测试函数，包含装饰器、默认参数和断言的 AST 保持一致；只对上述一处导入来源做明确归一化。
2. 所需模块级声明、辅助函数和提取出的工厂 AST 保留。
3. 全套实际收集结果与映射后的预期 node ID 多重集合完全相同，共 2005 项；488 项迁移用例逐项对应，没有重复收集。
4. 检查涉及文件没有继续从 `test_*` 模块导入。

AST 一致不等于运行语义的充分证明，因此还执行了独立进程验证和全量回归。此次 Ruff 格式整理只改变布局，纳入上述 AST 校验。

复跑校验：

```bash
conda run --no-capture-output -n py312 python agent/verification/verify_test_migration.py
conda run --no-capture-output -n py312 pytest -q -W error
```

该校验脚本用于本次迁移快照；将来有意修改断言或新增用例后需要更新映射，不能把它当成永久禁止测试演进的门禁。

## 验证记录

- [AST、依赖与收集等价性](verification/0929-test-migration-equivalence.txt)。
- [34 个迁移文件各自独立进程执行](verification/0929-test-migration-individual.txt)，[逐文件退出码](verification/0929-test-migration-individual.json)。每个文件使用单独临时目录和 pytest cache，检查不依赖其他测试文件的导入/执行顺序。
- [py312 全量回归](verification/0929-test-migration-full.txt)。
- [Python 3.14 独立 wheel 回归](verification/0929-test-migration-wheel.txt)：生产代码未变，复用第二批构建的 wheel，在新 venv 中复制当前测试（含两个辅助模块）后运行，没有重复构建相同生产代码。
- [Ruff 检查记录](verification/0929-test-migration-style.txt)。

最终验证：

| 检查 | 结果 |
|---|---|
| AST / 依赖 / 参数化 node ID 对应 | 通过；123 个迁移函数、488 项映射，全量 2005 项精确对应 |
| 34 个迁移文件分别在独立进程运行，`-W error` | 全部成功，共 488 项通过 |
| py312 全量，`-W error` | **2005 passed，34.28 秒，零警告** |
| Python 3.14 独立 wheel，全新 venv，`-W error` | **2005 passed，46.33 秒，零警告**；依赖检查与 10 个原生扩展导入通过 |
| Ruff 格式检查、未定义名称/语法检查 | 38 个文件通过 |
| `git diff --check` | 通过 |

完整 Ruff 规则检查仍报告一条既有 E731：`testInferencePublicObjectiveDirectionsAndSamplingUnchanged` 中的 `objective = lambda ...`，原属 `test_review_c01_c05.py`，现位于 `test_inference_result_isolation_and_directions.py`。为保持本轮函数 AST 等价，未改写为 `def`、未新增忽略规则。此项是已有风格提示，不计为完整 Ruff 检查通过。

本轮没有运行新的科学规模实验或增加 CI 组合，不提交、推送或发布。T07 的 CI 集成与 T08 导入职责整理仍是后续独立任务，C15/C17/C20 继续暂缓。
