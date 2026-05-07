# Analysis / Design 接口方案

## 结论

`analysis` 保留面向对象，但只保留“分析器对象”这一层。  
采样设计统一抽成 `design` 对象，并在分析器初始化时注入。

主关系定为：

| 层 | 职责 |
|---|---|
| `doe` | 通用设计，生成普通样本 |
| `analysis/designs` | 专用设计，生成结构化样本 |
| `analysis` | 分析计算、结果整理、设计元数据消费 |

分析器与设计器的关系定为：

```python
method = SomeAnalysis(design=SomeDesign(...))
X = method.sample(problem, ...)
res = method.analyze(problem, X, Y)
```

这里的 `method.sample()` 只是对 `self.design.sample()` 的薄代理。

---

## 这套方案解决什么问题

当前 `analysis` 存在 4 个核心问题：

| 问题 | 现象 |
|---|---|
| 采样和分析耦合 | `Sobol/FAST/Morris` 自己内嵌样本构造 |
| 超参数归属混乱 | 有些参数属于分析，有些属于设计，但都堆在 analysis 上 |
| 参数签名不一致 | 不同采样方法根本不是一个语义，没法硬统一 |
| 通用 DOE 无法自然接入 analysis | `LHS/Random/FFD/Sobol` 和分析专用设计目前是两条裂开的路径 |

本方案的目标只有一条：

> 统一使用路径，但不强行统一不同设计的参数语义。

---

## 总体结构

推荐保留并完善如下目录结构：

```text
UQPyL/
  doe/
    base.py
    random.py
    lhs.py
    full_fact.py
    sobol.py

  analysis/
    base.py
    designs/
      base.py
      sobol.py
      fast.py
      morris.py
    sobol.py
    fast.py
    morris.py
    rsa.py
    delta.py
    mars.py
    rbd_fast.py
```

语义如下：

| 位置 | 语义 |
|---|---|
| `doe` | 通用 design |
| `analysis/designs` | 专用 design |
| `analysis` | analyzer |

这里不新增顶层 `sampling` 模块。

---

## 为什么保留面向对象

不建议把 `analysis` 全改成纯函数，原因如下：

| 点 | 结论 |
|---|---|
| 分析器有配置 | 适合对象持有 |
| 分析器要绑定 design | 适合对象组合 |
| 结果记录 / verbose / save / log | 对象组织更顺 |
| 多个分析方法统一外观 | 对象接口更自然 |

但当前重 OO 的问题也确实存在，所以要收窄：

| 保留 | 去掉 |
|---|---|
| 分析器对象 | 分析器内部的采样公式 |
| 参数配置 | 设计器和分析器混写参数 |
| 结果对象 | 采样器状态乱挂在分析器里 |

最终原则：

> 保留 OO 外壳，去掉 OO 过载。

---

## 设计器注入方案

### 主方案

分析器初始化时接收 `design`：

```python
method = RSA(design=LHS("classic"))
method = Sobol(design=SobolDesign(secondOrder=True, scramble=False))
method = FAST(design=FASTDesign(M=4))
method = Morris(design=MorrisDesign(numLevels=4))
```

这样带来三个直接收益：

| 收益 | 说明 |
|---|---|
| 通用 DOE 和专用 design 接到同一路径 | 用户心智统一 |
| 采样参数归 design 所有 | 参数边界清楚 |
| `sample()` 和 `analyze()` 使用的是同一个 design | 避免参数错配 |

---

### 不采用的方案

#### 方案 1：在 `sample()` 时临时传 design

```python
method = Sobol()
X = method.sample(problem, design=SobolDesign(...))
```

问题：

| 问题 | 原因 |
|---|---|
| `sample()` 与 `analyze()` 可能用到不同 design | 容易错配 |
| 分析器对象本身不完整 | 生命周期不稳 |

#### 方案 2：analysis 自己直接收一堆采样参数

```python
Sobol(secondOrder=True, scramble=False, skipValue=0, ...)
```

问题：

| 问题 | 原因 |
|---|---|
| 参数归属错误 | 这些是 design 参数，不是分析参数 |
| 后续无法复用 design | 结构还是没拆开 |

---

## 参数归属规则

### 1. 分析参数

只属于 analyzer。

例子：

| 方法 | 参数 |
|---|---|
| `RSA` | `nRegion` |
| `DeltaTest` | `nNeighbors` |
| 所有 analysis | `target`、`index` |

这些参数影响的是“怎么分析”，不是“怎么采样”。

---

### 2. 设计参数

只属于 design。

例子：

| design | 参数 |
|---|---|
| `LHS` | `criterion`、`iterations` |
| `FFD` | `levels` |
| `SobolDesign` | `secondOrder`、`skipValue`、`scramble` |
| `FASTDesign` | `M` |
| `MorrisDesign` | `numLevels` |

这些参数影响的是“样本怎么构造”，不是“指标怎么计算”。

---

## 参数不一致时怎么办

结论：**不要统一不同 design 的 `sample()` 参数签名。**

不同 design 的主参数本来就不一样：

| design | 主参数语义 |
|---|---|
| `Random/LHS/Sobol` | `nt` |
| `FFD` | `levels` |
| `SobolDesign` | `N` |
| `FASTDesign` | `N` |
| `MorrisDesign` | `numTrajectory` |

所以正确方案不是：

```python
sample(problem, nt, seed=None)
```

而是：

| 层 | 策略 |
|---|---|
| `design.sample()` | 各自保留强语义签名 |
| `analysis.sample()` | 只做代理，接受 `**kwargs` |

即：

```python
class AnalysisABC:
    def sample(self, problem, **kwargs):
        if self.design is None:
            raise ValueError("design is required for sampling.")
        return self.design.sample(problem, **kwargs)
```

一句话：

> 统一调用路径，不统一参数形状。

---

## 为什么分析器仍然需要 design 信息

虽然设计参数归 design，但分析器在 `analyze()` 时仍可能需要设计信息。

例子：

| analyzer | analyze 时需要知道什么 |
|---|---|
| `Sobol` | `secondOrder`，否则没法拆 `Y` |
| `FAST` | `M`，否则频谱截取不对 |
| `Morris` | `numLevels`，否则解释和部分计算边界不完整 |

所以设计器不能只负责 `sample()`，还必须负责提供样本结构元数据。

---

## Design 最小协议

所有可注入 analyzer 的 design，至少应满足：

| 方法 / 属性 | 作用 |
|---|---|
| `sample(problem, ..., seed=None)` | 生成样本 |
| `get_meta()` | 返回 analyzer 可能需要的设计元数据 |

可选但推荐：

| 方法 / 属性 | 作用 |
|---|---|
| `name` | 结果记录、调试、日志 |
| `validate_for(problem)` | 特定 design 的问题维度或参数合法性校验 |

---

## `get_meta()` 约定

设计器返回的 metadata 只包含分析器真正需要的结构信息。

例子：

| design | `get_meta()` 示例 |
|---|---|
| `LHS()` | `{}` |
| `Random()` | `{}` |
| `FFD()` | `{}` |
| `SobolDesign(secondOrder=True, skipValue=0, scramble=False)` | `{"secondOrder": True, "skipValue": 0, "scramble": False}` |
| `FASTDesign(M=4)` | `{"M": 4}` |
| `MorrisDesign(numLevels=4)` | `{"numLevels": 4}` |

分析器只读取自己关心的字段。

---

## 命名方案

### analyzer 命名

保持短名：

| 文件 | 类名 |
|---|---|
| `analysis/sobol.py` | `Sobol` |
| `analysis/fast.py` | `FAST` |
| `analysis/morris.py` | `Morris` |
| `analysis/rsa.py` | `RSA` |
| `analysis/delta.py` | `DeltaTest` |

---

### design 命名

考虑到：

| 候选 | 问题 |
|---|---|
| `SaltelliDesign` | 太长，而且更像实现细节名 |
| `Saltelli` | 会和 analyzer 命名策略脱节，也不够用户视角 |
| `FAST` / `Morris` | 会与 analyzer 撞名 |

最终建议统一面向“用户要做哪种分析”来命名：

| 文件 | 类名 |
|---|---|
| `analysis/designs/sobol.py` | `SobolDesign` |
| `analysis/designs/fast.py` | `FASTDesign` |
| `analysis/designs/morris.py` | `MorrisDesign` |

理由：

| 点 | 说明 |
|---|---|
| 不撞名 | 与 analyzer 分层清楚 |
| 面向用户 | 用户心智是 Sobol/FAST/Morris 分析，不是 Saltelli 论文内部命名 |
| 对称 | 三个 design 风格一致 |

---

## AnalysisABC 建议职责

`analysis/base.py` 后续建议只保留：

| 职责 | 保留与否 |
|---|---|
| `design` 挂载 | 保留 |
| `problem` 挂载 | 保留 |
| `sample()` 薄代理 | 保留 |
| `analyze()` 抽象方法 | 保留 |
| 结果记录 / 设置 / verbose | 保留 |
| 具体采样公式 | 不保留 |
| 具体指标计算细节 | 尽量下放到各分析器私有函数 |

建议新增：

| 方法 | 作用 |
|---|---|
| `setDesign(design)` | 显式替换 design |
| `getDesignMeta()` | 统一取 `self.design.get_meta()` |

---

## `sample()` 的角色

后续 `analysis.sample()` 不应该被当作统一参数协议，而只是一个便利入口。

它的定位是：

| 角色 | 含义 |
|---|---|
| 主逻辑 | `self.design.sample(problem, **kwargs)` |
| 责任 | 代理，不解释不同参数 |
| 文档重点 | 参数由具体 design 决定 |

换句话说：

> 真正稳定的 API 是 `design.sample()`，不是 `analysis.sample()` 的参数表。

---

## 通用 DOE 如何接进 analysis

这套方案下，通用 DOE 和专用 design 都能成为 analyzer 的 `design`。

示例：

| analyzer | design |
|---|---|
| `RSA` | `LHS("classic")` / `Random()` / `Sobol()` |
| `DeltaTest` | `LHS("classic")` / `Random()` |
| `MARS` | `LHS("classic")` / `Random()` |
| `Sobol` | `SobolDesign(...)` |
| `FAST` | `FASTDesign(...)` |
| `Morris` | `MorrisDesign(...)` |

这样 `analysis` 的外部使用路径就统一了。

---

## 后续实施顺序

### 阶段 1：先建 design 接口

| 动作 | 说明 |
|---|---|
| 新建 `analysis/designs/base.py` | 定义 design 最小协议 |
| 新建 `SobolDesign/FASTDesign/MorrisDesign` | 迁入专用样本构造 |

---

### 阶段 2：改 `AnalysisABC`

| 动作 | 说明 |
|---|---|
| 加 `design` 字段 | 初始化注入 |
| 加 `sample(problem, **kwargs)` | 薄代理到 design |
| 加 `getDesignMeta()` | 统一取 design metadata |

---

### 阶段 3：改 analyzer

| analyzer | 改法 |
|---|---|
| `Sobol` | 不再内嵌 Saltelli 构造，改读 `SobolDesign` |
| `FAST` | 不再内嵌 FAST 采样，改读 `FASTDesign` |
| `Morris` | 不再内嵌轨迹采样，改读 `MorrisDesign` |
| `RSA/DeltaTest/MARS/RBDFAST` | 接受通用 `doe` design |

---

## 最终原则

最终接口原则定为：

1. `analysis` 保留对象。
2. `design` 在初始化时注入 analyzer。
3. 采样参数只属于 design。
4. 分析器通过 design metadata 获取所需结构信息。
5. 不统一不同 design 的参数签名。
6. 统一的是“设计器可注入 analyzer”这条使用路径。

一句话总结：

> `analysis` 负责算，`design` 负责采；analysis 组合 design，但不吞掉 design 的参数语义。
