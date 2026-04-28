# Problem 设计草图

## 结论

当前 `Problem` 先收敛成**静态、定长输出、可选约束**的问题抽象。

不要在这一层提前混入：

- 同化
- 动态系统
- 诊断量
- 时序状态

当前方案只解决 6 件事：

| 点 | 方案 |
|---|---|
| 名字 | 保留 `Problem` 作为公开名字 |
| 语义 | 明确成静态评估问题，不再承担更宽语义 |
| 返回值 | 不再返回 `dict`，改成固定结果对象 `EvalResult` |
| 约束 | 允许不存在 |
| 计算方式 | 默认全算，支持目标和约束分开算 |
| 输入空间 | 从 `Problem` 拆出，形成 `SpaceBase + Space` 两层 |

---

## 适用范围

这份草图只覆盖当前静态 `Problem` 架构，服务于：

| 场景 | 是否纳入 |
|---|---|
| 优化 | 是 |
| 敏感性分析 | 是 |
| 代理模型 | 是 |
| 固定维度输出的推断 | 是 |
| EnKF / IES / ES-MDA | 否 |
| 动态状态空间问题 | 否 |
| diagnostics 主通道 | 否 |

结论：  
**先把静态问题抽象理顺，不提前做动态扩张。**

---

## 主接口方案

主入口统一为：

```python
problem.evaluate(X, target=None)
```

语义如下：

| 调用 | 含义 |
|---|---|
| `evaluate(X)` | 默认全算 |
| `evaluate(X, target="objs")` | 只算目标 |
| `evaluate(X, target="cons")` | 只算约束 |

不再使用 `require`。

原因：

| 方案 | 判断 |
|---|---|
| `require=("objs", "cons")` | 太底层，接口难看 |
| `target=None/"objs"/"cons"` | 更自然，语义直接 |

---

## 为什么这样定

| 问题 | 处理方式 |
|---|---|
| 有些问题没有约束 | `cons` 允许为空 |
| 有些约束计算很贵 | 支持只算 `objs` |
| 当前 `dict` 返回太松 | 改成固定结果对象 |
| 当前 `Problem` 职责过多 | 输入空间单独拆出去 |
| 后面可能有特殊变量空间 | 留 `SpaceBase` 外部扩展口 |

这 5 个点是现在最该先解决的，而不是继续扩展功能面。

---

## 命名约定

本轮草图统一采用以下命名：

| 类别 | 命名 |
|---|---|
| 输入个数 | `nInput` |
| 目标个数 | `nObj` |
| 约束个数 | `nCon` |
| 输入名 | `xLabels` |
| 目标名 | `objLabels` |
| 约束名 | `conLabels` |
| 优化方向 | `optType` |
| 结果类 | `EvalResult` |
| 空间基类 | `SpaceBase` |
| 默认空间实现 | `Space` |

结论：  
**全部收短，尽量贴近现有代码风格，同时保证内部一致。**

---

## 核心抽象

### 1. `SpaceBase`

职责：定义输入空间协议，允许外部自定义。

| 必须承担的内容 |
|---|
| 输入维度 |
| 输入标签 |
| 输入校验 |
| 输入变换 |

草图：

```python
class SpaceBase:
    nInput: int
    xLabels: list[str]

    def validate(self, X): ...
    def transform(self, X): ...
```

设计意图：

| 点 | 说明 |
|---|---|
| 为什么要有 `SpaceBase` | 让外部用户可以自定义空间逻辑 |
| 为什么不直接只留 `Space` | 否则后面特殊变量空间又会重新塞回 `Problem` |

---

### 2. `Space`

职责：默认输入空间实现，处理当前项目里最常见的边界和变量类型问题。

| 包含内容 |
|---|
| `ub / lb` |
| `varType / varSet` |
| 连续、整数、离散变量变换 |
| 输入空间校验 |

明确不属于 `Space` 的内容：

| 项 | 原因 |
|---|---|
| `optType` | 它描述的是目标方向，不是输入空间属性 |

草图：

```python
class Space(SpaceBase):
    ub: np.ndarray
    lb: np.ndarray
    varType: np.ndarray | None
    varSet: dict | None

    def validate(self, X): ...
    def transform(self, X): ...
```

说明：

| 项 | 结论 |
|---|---|
| `Space` 是否是默认实现 | 是 |
| 是否允许用户不用它 | 是，只要遵守 `SpaceBase` 协议 |

---

### 3. `EvalResult`

职责：作为 `evaluate` 的固定返回对象。

第一版只保留最小字段，不放 diagnostics。

| 字段 | 含义 |
|---|---|
| `objs` | 目标值，没有则为 `None` |
| `cons` | 约束值，没有则为 `None` |

草图：

```python
class EvalResult:
    objs: np.ndarray | None = None
    cons: np.ndarray | None = None

    @property
    def hasObjs(self) -> bool: ...

    @property
    def hasCons(self) -> bool: ...
```

当前明确不纳入：

| 项 | 当前是否纳入 |
|---|---|
| diagnostics | 否 |
| raw 输出 | 否 |
| meta 信息 | 否 |

结论：  
**先做最小结果对象，不把结果层搞胖。**

---

### 4. `ProblemBase`

职责：只定义协议和元数据，不塞变量空间细节。

草图：

```python
class ProblemBase:
    name: str
    space: SpaceBase
    nInput: int
    nObj: int
    nCon: int
    xLabels: list[str]
    objLabels: list[str]
    conLabels: list[str] | None
    optType: str | list[str] | None

    def evaluate(self, X, target=None) -> EvalResult:
        raise NotImplementedError
```

设计要点：

| 点 | 说明 |
|---|---|
| `nInput` 是否保留在 `Problem` 上 | 保留，方便算法侧直接取用 |
| `xLabels` 是否也保留在 `Problem` 上 | 保留，避免外部代码总去绕 `space` |
| `optType` 放在哪 | 放在 `Problem`，不放在 `Space` |
| `optType` 是否强制必填 | 否，作为可选元数据 |

---

### 5. `Problem`

职责：静态问题的具体实现，适配用户提供的：

- `evaluator`
- `objFunc`
- `conFunc`

草图：

```python
class Problem(ProblemBase):
    def __init__(
        self,
        space,
        evaluator=None,
        objFunc=None,
        conFunc=None,
        nObj=1,
        nCon=0,
        name=None,
        objLabels=None,
        conLabels=None,
        optType=None,
    ): ...

    def evaluate(self, X, target=None) -> EvalResult: ...

    def objFunc(self, X):
        return self.evaluate(X, target="objs").objs

    def conFunc(self, X):
        return self.evaluate(X, target="cons").cons
```

设计说明：

| 点 | 结论 |
|---|---|
| `Problem` 是否继续保留 `objFunc/conFunc` | 保留，但降级成兼容接口 |
| `evaluate` 是否是主协议 | 是 |

---

## 评估协议

### 调用协议

```python
problem.evaluate(X)
problem.evaluate(X, target="objs")
problem.evaluate(X, target="cons")
```

### 返回规则

| 调用 | 返回 |
|---|---|
| `evaluate(X)` | 若有约束则返回 `objs + cons`，否则 `cons=None` |
| `evaluate(X, target="objs")` | 只返回 `objs` |
| `evaluate(X, target="cons")` | 只返回 `cons`；若无约束则 `cons=None` |

### 错误规则

| 情况 | 处理 |
|---|---|
| `target` 非法 | 直接报错 |
| 问题无约束但请求 `cons` | 返回 `EvalResult(cons=None)` |
| 算法强依赖约束但拿到 `None` | 由算法层报错，不在 `Problem` 层兜底 |

---

## 兼容策略

当前代码里很多地方直接依赖：

- `objFunc`
- `conFunc`
- `nInput`
- `ub / lb`

所以迁移不能一步切断。

兼容策略如下：

| 旧接口 | 新角色 |
|---|---|
| `evaluate()` | 主协议 |
| `objFunc()` | `evaluate(..., target="objs")` 的包装器 |
| `conFunc()` | `evaluate(..., target="cons")` 的包装器 |
| `ub / lb` | 初期可在 `Problem` 上做透传到 `space` |
| `varType / varSet` | 初期可在 `Problem` 上做透传到 `space` |

结论：  
**先收主协议，再逐步弱化旧接口，不做硬切。**

---

## `optType` 定位

`optType` 保留，但只属于 `Problem`，不属于 `Space`。

原因：

| 判断 | 结论 |
|---|---|
| `optType` 是不是输入空间属性 | 不是 |
| `optType` 是不是问题元数据 | 是 |
| 是否所有 `Problem` 都必须强制有 `optType` | 否 |

建议规则：

| 场景 | 规则 |
|---|---|
| 单目标优化 | `optType="min"` 或 `"max"` |
| 多目标优化 | `optType=["min", "max", ...]` |
| 非优化场景 | `optType=None` 允许存在 |
| 优化算法运行时发现 `optType=None` | 由算法层报错 |

结论：  
**`optType` 是 `Problem` 的可选元数据，不是 `Space` 的组成部分。**

---

## 应该从当前 `Problem` 里拆出去的东西

这些内容不该继续留在 `Problem` 核心抽象里：

| 内容 | 去向 |
|---|---|
| 上下界设置 | `Space` |
| 整数变量取整 | `Space` |
| 离散变量映射 | `Space` |
| 混合变量编码细节 | `Space` |

结论：  
**问题定义和输入空间处理必须拆开。**

---

## 当前明确不纳入的内容

这几项先排除：

| 项 | 原因 |
|---|---|
| diagnostics | 现在会把 `Problem` 继续做胖 |
| time axis | 属于动态问题，不属于当前静态抽象 |
| state | 同上 |
| observations | 同上 |
| assimilation 结果字段 | 应该留给后续动态 / 同化抽象 |

结论：  
**现在先克制，不提前设计未来层。**

---

## 包结构草图

```text
UQPyL/problem/
  __init__.py
  base.py
  space.py
  result.py
  static.py
  adapters.py
  benchmark/
    __init__.py
    sop/
    mop/
```

含义如下：

| 文件 | 职责 |
|---|---|
| `base.py` | 基础协议 |
| `space.py` | 输入空间定义 |
| `result.py` | 评估结果对象 |
| `static.py` | 静态 `Problem` 实现 |
| `adapters.py` | `singleFunc` 等兼容包装 |
| `benchmark/` | 标准测试问题 |

---

## 迁移顺序

| 阶段 | 动作 |
|---|---|
| 第一阶段 | 引入 `EvalResult`，替换 `dict` 返回 |
| 第二阶段 | 引入 `target=None/"objs"/"cons"` 协议 |
| 第三阶段 | 抽出 `SpaceBase + Space` |
| 第四阶段 | 在 `Problem` 上保留必要透传字段，平滑迁移算法层 |
| 第五阶段 | 整理 `problem` 包结构，分离 benchmark 和工具 |

---

## 最终定位

当前规划下，`Problem` 的定义应该明确成：

> 一个静态、定长输出、支持可选约束、支持目标与约束分开评估、并基于可扩展输入空间协议的问题抽象。

而不是：

> 一个统一承载动态系统、同化、时序状态、诊断量的总抽象。

最终一句话：

**先把 `Problem` 做窄、做稳、做干净，并把 `Space` 做成可扩展协议，再考虑后续动态扩展。**
