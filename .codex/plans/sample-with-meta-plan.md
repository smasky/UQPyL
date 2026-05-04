# `sampleWithMeta` 方案草案

## 结论

建议采用“双入口采样”方案：

- `sample(...)`：返回普通 `np.ndarray`
- `sampleWithMeta(...)`：返回带元数据的样本对象

这个方案的目标不是把整个 `DoE` 层都改成“返回复杂对象”，而是：

1. 保留当前默认用法的轻量体验
2. 给 `Sobol` / `FAST` / `Morris` 这类需要采样协议上下文的分析方法提供稳定输入
3. 避免把设计协议参数在 `sample()` 和 `analyze()` 两边重复维护

---

## 这个方案要解决什么

当前 `analysis` 的核心问题不是“拿不到样本”，而是“拿到了样本，但丢了样本的生成协议”。

典型例子：

- `Sobol.analyze()` 需要知道 `secondOrder`
- `FAST.analyze()` 需要知道 `M`
- `Morris.analyze()` 需要知道 `numLevels`

如果只传裸 `X`，这些信息只能：

- 再从 `analyze()` 参数里传一遍
- 从 `X.shape` 做不可靠推断
- 或者让 `analysis` 自己持有一份与采样阶段重复的参数

这三种都不理想。

---

## 核心设计

### 1. 每个设计对象提供两个采样入口

统一约定：

```python
X = design.sample(problem, ...)
sample = design.sampleWithMeta(problem, ...)
```

语义：

- `sample()` 面向默认使用场景，返回普通样本矩阵
- `sampleWithMeta()` 面向需要保留设计协议的场景，返回结构化样本对象

这样可以避免：

- 强制所有调用方都处理 metadata
- 为少数 analysis 场景抬高整个 `DoE` 层使用门槛

---

### 2. `sampleWithMeta()` 返回轻量对象

建议新增一个轻量容器，例如：

```python
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

@dataclass
class SampleWithMeta:
    X: np.ndarray
    meta: Dict[str, Any]
```

最小要求只保留两个字段：

- `X`：样本矩阵
- `meta`：设计协议元数据

是否增加 `designName`、`problemInfo`、`seed` 等字段，可以后续再扩，不必第一版做重。

---

## 元数据内容建议

`meta` 不应该变成“把整个设计对象序列化一遍”，而应只放分析器真正需要的协议字段。

### 通用 `DoE`

对 `LHS` / `Random` / `FFD` 这类通用设计，建议 `meta` 尽量轻：

```python
{
    "designType": "lhs",
    "criterion": "classic",
    "iterations": 5,
}
```

说明：

- 这些字段更多是记录用途
- 大多数 analysis 不会依赖它们做解释

### 专用设计

对结构化设计，`meta` 需要明确描述协议：

`SobolDesign`:

```python
{
    "designType": "sobol",
    "secondOrder": True,
    "skipValue": 0,
    "scramble": False,
    "N": 512,
}
```

`FASTDesign`:

```python
{
    "designType": "fast",
    "M": 4,
    "N": 513,
}
```

`MorrisDesign`:

```python
{
    "designType": "morris",
    "numLevels": 4,
    "numTrajectory": 100,
}
```

原则：

1. 只放分析期会消费的协议字段
2. 允许记录本次采样规模参数，如 `N`
3. 不要求不同方法的 `meta` 结构完全统一

---

## 为什么是 `sampleWithMeta`，不是只保留一种返回

### 不建议让 `sample()` 一律返回复杂对象

原因：

1. 会直接改变当前大量默认用法
2. 用户原本习惯 `X = sampler.sample(...)`
3. `problem.evaluate(X)`、`objFunc(X)`、现有分析代码大量默认输入是 `np.ndarray`

如果直接强推统一返回对象，短期收益未必抵得上侵入性。

### `sampleWithMeta` 的好处

1. 默认路径不变
2. 高级路径明确
3. 命名直接表达“这不是普通样本”
4. 比 `returnMeta=True` 更清楚，也更稳定

---

## `analysis` 侧如何消费

建议 `analysis.analyze()` 支持两种输入：

- 裸 `X`
- `SampleWithMeta`

例如：

```python
X = lhs.sample(problem, 100)
res = RSA().analyze(problem, X, Y)
```

```python
sample = sobolDesign.sampleWithMeta(problem, N=512, seed=1)
res = Sobol().analyze(problem, sample, Y)
```

分析器内部统一先做一次解包：

```python
def _unwrap_sample(X):
    if isinstance(X, SampleWithMeta):
        return X.X, X.meta
    return X, None
```

然后按方法分类处理。

### 1. 对协议不敏感的方法

如：

- `RSA`
- `DeltaTest`
- `MARS`

它们通常只需要 `X`：

- 如果传入 `SampleWithMeta`，直接取 `.X`
- `meta` 可以忽略

### 2. 对协议敏感的方法

如：

- `Sobol`
- `FAST`
- `Morris`

它们优先使用 `meta`：

- 如果有 `meta`，从中读取协议字段
- 如果没有 `meta`，允许回退到显式参数
- 如果两边都没有，报错并提示用户使用 `sampleWithMeta()`

例如 `Sobol.analyze()`：

```python
sample = design.sampleWithMeta(problem, N=512)
Y = problem.objFunc(sample.X)
res = Sobol().analyze(problem, sample, Y)
```

或者兼容旧用法：

```python
X = design.sample(problem, N=512)
Y = problem.objFunc(X)
res = Sobol().analyze(problem, X, Y, secondOrder=True)
```

---

## 与当前模块边界的关系

这个方案不强制你立刻决定“design 放在 `DoE` 还是 `analysis`”。

它先解决的是“协议随样本走”的问题。

无论最终设计放在哪一侧，都可以共用这套返回语义：

- 通用 `DoE` 可以提供 `sample()` / `sampleWithMeta()`
- analysis 专用 design 也可以提供 `sample()` / `sampleWithMeta()`

所以它是一个比目录归属更稳定的接口层设计。

---

## 推荐的最小落地范围

第一轮不要全仓一起推，建议只覆盖最需要 metadata 的路径。

### 阶段 1

先新增基础能力：

1. 增加 `SampleWithMeta`
2. 在相关基类中增加 `sampleWithMeta()`
3. 默认 `sample()` 内部调用统一生成逻辑，只返回 `X`

建议模式：

```python
def sample(self, problem, ...):
    return self.sampleWithMeta(problem, ...).X
```

这样普通样本和带元数据样本共享同一份生成实现。

### 阶段 2

优先接入最依赖协议的三个方法：

1. `Sobol`
2. `FAST`
3. `Morris`

使其 `analyze()` 能消费 `SampleWithMeta`。

### 阶段 3

再决定是否给通用设计也补齐 `sampleWithMeta()`，用于：

- 调试
- 日志
- 统一外部接口

这一步不是第一优先级。

---

## 基类建议

### `DoE.base.Sampler`

建议新增：

```python
class Sampler(...):
    def sample(self, problem, nt, seed=None):
        return self.sampleWithMeta(problem, nt, seed=seed).X

    def sampleWithMeta(self, problem, nt, seed=None):
        ...
```

并提供一个可覆盖的钩子：

```python
def _build_meta(self, ...):
    return {}
```

通用设计默认返回轻量 `meta` 即可。

### analysis 专用 design

如果后面保留专用 design 层，也建议遵循同样接口：

```python
sample(problem, ...)
sampleWithMeta(problem, ...)
```

这样用户心智一致。

---

## 命名建议

推荐直接用：

- `sample`
- `sampleWithMeta`

不建议：

- `sample_meta`
- `sampleMetaData`
- `sampleInfo`
- `returnMeta=True`

原因：

1. `sampleWithMeta` 可读性最好
2. 与当前 API 风格兼容
3. 比布尔参数更显式

---

## 风险与取舍

### 优点

1. 保住默认易用性
2. 给协议敏感 analysis 提供稳定上下文
3. 避免在采样与分析两边重复维护参数
4. 迁移可以渐进进行

### 代价

1. API 面上会多一个方法
2. `analysis` 需要兼容两种输入类型
3. 文档里需要明确区分“普通样本”和“结构化样本”

### 当前阶段可接受性

这个代价是可接受的，因为相比“一刀切全返回对象”，它对现有用户干扰更小。

---

## 最终建议

推荐采用下面这条主线：

1. 保留 `sample()` 返回 `np.ndarray`
2. 新增 `sampleWithMeta()` 返回 `SampleWithMeta`
3. `Sobol/FAST/Morris` 优先消费 `SampleWithMeta`
4. `RSA/DeltaTest/MARS` 只做兼容，不强依赖 `meta`
5. 先把 metadata 当成 analysis 协议桥接层，不急着把整个库都改成“对象样本流”

一句话总结：

> `sampleWithMeta` 不是为了替代普通 `sample`，而是为了给需要协议上下文的分析方法增加一条稳定、显式、低侵入的样本通路。
