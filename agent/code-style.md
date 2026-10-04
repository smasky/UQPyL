# 代码与接口约定

本轮 C22—C24 的规则；继续遵循根目录 AGENTS.md。格式工具只用于 Python 包装层，不把原生/移植核心改造成另一套实现。

- Python：四空格缩进，Ruff 0.16.8，目标 Python 3.10，行宽 120，双引号。长字符串、公式和链接不为凑行宽强行拆分。配置在 `pyproject.toml`，CI 检查 `UQPyL/`。
- 内部新增字段、方法、局部对象用 camelCase；私有方法用单下划线。X/Y、矩阵符号保留；Python 协议方法、NumPy/SciPy/scikit-learn 风格接口、原生核心参数等保留约定名字。不要通过全仓正则重命名数学变量或第三方回调。
- 结果对象属性沿用 camelCase；正式 `summary()`、`toDict()` 的固定字段用 snake_case，分析的 X/Y 数学符号例外。`settings`、`meta`、`extra` 以及用户自定义诊断内容不递归改写键名；其键可能是公开配置或用户数据。
- 现有底层 SQLite 表/快照字段属于存储协议，不能仅为风格直接改表名或列名。reader 汇总及新增 `load_partial_result()` 使用 snake_case，原始快照读取仍按数据库字段返回。
- 正式采样预算统一 `maxIters`，包含初始正式样本；不包括 warm-up。本次移除 AMH/DEMC 的 `maxIterTimes` 名称，不留别名。EGO 支持 `surrogate`、`optimizer` 构造注入。
- 自有 Python 文档统一 Google `Args` / `Returns` / `Raises` / `Notes` 分节。旧的 99 处 Sphinx 字段已转换；移植核心按原文档规范保留。

公共方法的说明模板：

```python
def run(self, problem, X, seed=None):
    """Describe the operation and mathematical scope.

    Args:
        problem: Required problem type and supported constraints.
        X: Input shape, coordinate space, units, and finite-value requirements.
        seed: Local random seed, if used.

    Returns:
        Result: Result type, array shapes, objective direction, and units.

    Raises:
        ValueError: Configurations rejected before model evaluation.

    Notes:
        Explain state mutation, preprocessing/refitting, persistence, and
        important algorithm limitations. Do not promise exact recovery from
        snapshots or convergence from a single diagnostic.
    """
```

验证命令（conda py312）：

```bash
python -m ruff format --check UQPyL
python -m ruff check UQPyL
pytest -q -W error
```

格式规则不启用自动删除导入、算法重写或全仓命名强制规则；已有科学计算符号、扩展接口和运行时动态入口需人工判断。测试文件暂不纳入包格式门禁，避免把大量既有测试的机械重排混入这次源码整理。
