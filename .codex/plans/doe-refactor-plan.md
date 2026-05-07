# DoE 改造计划

## 目标

本轮只完成 `UQPyL/DoE` 的内部改造，不同时推进 `analysis` 侧重构。

本轮改造目标有 5 个：

1. 统一 `DoE` 采样接口
2. 引入 `SampleSet`
3. 增加 `sampleWithMeta()`
4. 收口 `problem.unit_to_space()` 映射路径
5. 为后续 `analysis` 消费 protocol metadata 做准备

本轮不做：

1. 不拆 `DoE` 子目录
2. 不引入第二个基类
3. 不改 `analysis` 侧 API
4. 不强推全库都消费 `SampleSet`

---

## 最终口径

`DoE` 先统一使用**单一基类**：

- `Sampler`

所有采样器都提供两个方法：

- `sample(...)`：返回 `np.ndarray`
- `sampleWithMeta(...)`：返回 `SampleSet`

所有采样器先继续放在 `UQPyL/DoE/` 同一层目录下，不分文件夹。

语义区分暂时只体现在子类 metadata 上，不体现在目录层级或基类层级上。

---

## 核心设计

### 1. 新增 `SampleSet`

建议放在：

- `UQPyL/DoE/base.py`

第一版最小结构：

```python
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

@dataclass
class SampleSet:
    X: np.ndarray
    meta: Dict[str, Any]
```

约定：

1. `X` 为 problem space 样本
2. `meta` 为采样协议元数据
3. `SampleSet` 只表示一次采样设计输出
4. `SampleSet` 不是结果对象，不包含 `Y`

可选增强，但不是第一轮必需：

```python
@property
def shape(self):
    return self.X.shape
```

---

### 2. 统一 `Sampler` 基类

目标接口：

```python
class Sampler(...):
    def sample(self, problem, nt, seed=None):
        return self.sampleWithMeta(problem, nt, seed=seed).X

    def sampleWithMeta(self, problem, nt, seed=None):
        ...

    def _generate(self, nt, nx):
        ...

    def _build_meta(self, ...):
        return {}
```

这里的职责划分如下：

| 方法 | 职责 |
|---|---|
| `sample()` | 默认普通采样入口 |
| `sampleWithMeta()` | 返回 `SampleSet` 的正式入口 |
| `_generate()` | 只负责生成 unit-space 样本 |
| `_build_meta()` | 构造 metadata |

统一约定：

1. `_generate()` 返回 unit-space 样本
2. `sampleWithMeta()` 负责调用 `problem.unit_to_space()`
3. `sample()` 永远返回 problem-space `np.ndarray`
4. `SampleSet.X` 永远是 problem-space 样本

---

## metadata 设计原则

### 总原则

`meta` 不是日志容器，而是**采样协议描述**。

只有 analysis 后续真的可能消费的信息，才放进去。

### 通用采样器

对 `Random` / `LHS` / `FFD` / `SobolSequence`，metadata 可以保持轻量。

示例：

`Random`

```python
{
    "designType": "random",
}
```

`LHS`

```python
{
    "designType": "lhs",
    "criterion": "classic",
    "iterations": 5,
}
```

`FFD`

```python
{
    "designType": "full_fact",
    "levels": levels,
}
```

`SobolSequence`

```python
{
    "designType": "sobol_sequence",
    "scramble": True,
    "skipValue": 0,
}
```

这些 `meta` 主要用于记录和统一接口，不作为分析协议强依赖。

### 结构化采样器

对 `SaltelliSequence` / `FASTSequence` / `MorrisSequence`，metadata 要明确描述协议。

`SaltelliSequence`

```python
{
    "designType": "saltelli",
    "N": 512,
    "secondOrder": True,
    "skipValue": 0,
    "scramble": False,
    "blockSize": 2 * nInput + 2,
}
```

`FASTSequence`

```python
{
    "designType": "fast",
    "N": 513,
    "M": 4,
    "blockSize": 513,
}
```

`MorrisSequence`

```python
{
    "designType": "morris",
    "numTrajectory": 100,
    "numLevels": 4,
    "trajectorySize": nInput + 1,
}
```

第一轮先不强行统一全部字段名，只要求：

1. `designType` 无歧义
2. protocol 关键参数完整
3. 采样规模参数可追溯

---

## 现有采样器改造范围

### 1. `UQPyL/DoE/base.py`

需要完成：

1. 新增 `SampleSet`
2. 给 `Sampler` 增加 `sampleWithMeta()`
3. 保留并重写 `sample()` 为 `sampleWithMeta(...).X`
4. 增加 `_build_meta()` 默认实现
5. 继续保留基础校验逻辑

注意：

- `sampleWithMeta()` 不应假设所有子类参数都叫 `nt`
- 第一轮先优先覆盖现有确实走 `nt` 风格的采样器
- `FFD` 因为 `levels` 语义特殊，可继续单独覆写

### 2. `UQPyL/DoE/random.py`

需要完成：

1. 保持 `_generate(nt, nx)` 不变
2. 覆盖 `_build_meta()`

### 3. `UQPyL/DoE/lhs.py`

需要完成：

1. 保持 `_generate(nt, nx)` 不变
2. 覆盖 `_build_meta()`，记录 `criterion` 和 `iterations`

### 4. `UQPyL/DoE/sobol.py`

需要完成：

1. 保持普通 Sobol sequence 语义
2. 明确 `designType = "sobol_sequence"`
3. 覆盖 `_build_meta()`，记录 `scramble` 与 `skipValue`

### 5. `UQPyL/DoE/full_fact.py`

这是第一轮里的特例。

原因：

- 它不使用 `nt`
- 它使用 `levels`

建议：

1. 继续保留独立 `sample()`
2. 手动新增 `sampleWithMeta()`
3. 内部仍共享同一套生成逻辑

不要为了统一而硬把 `FFD` 塞进 `nt` 接口。

### 6. `UQPyL/DoE/saltelli.py`

需要完成：

1. 逐步去掉自定义 `sample()` 与基类重复的部分
2. 尽量复用基类的 `sampleWithMeta()`
3. 覆盖 `_build_meta()`
4. `designType` 明确用 `saltelli`

### 7. `UQPyL/DoE/fast.py`

需要完成：

1. 逐步去掉自定义 `sample()` 与基类重复的部分
2. 复用基类的 `sampleWithMeta()`
3. 覆盖 `_build_meta()`

### 8. `UQPyL/DoE/morris.py`

需要完成：

1. 逐步去掉自定义 `sample()` 与基类重复的部分
2. 复用基类的 `sampleWithMeta()`
3. 覆盖 `_build_meta()`

---

## 兼容性策略

### 对外行为

本轮目标是不破坏默认用法：

```python
X = sampler.sample(problem, ...)
```

仍然成立。

新增能力：

```python
sample = sampler.sampleWithMeta(problem, ...)
```

### 命名

保留：

- `sample`
- `sampleWithMeta`

不引入：

- `sampleMeta`
- `returnMeta=True`

### 类名

保留：

- `SaltelliSequence`
- `FASTSequence`
- `MorrisSequence`

是否后续改为 `SaltelliDesign`、`FASTDesign`、`MorrisDesign`，本轮不做。

---

## 风险点

### 1. `FFD` 的参数语义不同

风险：

- 基类默认 `nt` 模式不适合 `FFD`

处理：

- 第一轮允许 `FFD` 特化

### 2. metadata 过早膨胀

风险：

- 把调试信息、problem 信息、运行日志全塞进 `meta`

处理：

- 第一轮只放 protocol 字段和少量描述字段

### 3. `designType` 歧义

风险：

- 把普通 Sobol sequence 和 Saltelli protocol 都叫 `"sobol"`

处理：

- 普通序列使用 `sobol_sequence`
- Sobol 敏感性专用设计使用 `saltelli`

### 4. 双重映射问题继续扩散

风险：

- 某些调用方继续把 `sample()` 返回值当 unit-space 再做 `unit_to_space`

处理：

- 在计划和实现中明确：
  - `sample()` 返回 problem space
  - `sampleWithMeta().X` 返回 problem space

---

## 实施顺序

### 阶段 1：改基类

1. 更新 `UQPyL/DoE/base.py`
2. 新增 `SampleSet`
3. 新增 `sampleWithMeta()`
4. 保持 `sample()` 兼容

验收标准：

- 普通采样器不改调用方式也能工作
- 新接口可返回 `SampleSet`

### 阶段 2：改通用采样器

1. `Random`
2. `LHS`
3. `SobolSequence`
4. `FFD`

验收标准：

- 全部支持 `sampleWithMeta()`
- metadata 最小可用

### 阶段 3：改结构化采样器

1. `SaltelliSequence`
2. `FASTSequence`
3. `MorrisSequence`

验收标准：

- 三者都能通过 `sampleWithMeta()` 返回 protocol metadata
- protocol 字段足够支撑后续 analysis 消费

### 阶段 4：整理导出与文档

1. 更新 `UQPyL/DoE/__init__.py`
2. 导出 `SampleSet`
3. 文档中区分 `sample()` 和 `sampleWithMeta()`

---

## 本轮完成后的状态

完成后，`DoE` 应满足：

1. 所有采样器使用统一双入口
2. 默认用法保持不变
3. `SampleSet` 成为标准结构化输出
4. protocol metadata 可由结构化采样器稳定提供
5. 后续 `analysis` 改造可以直接基于 `SampleSet` 展开

一句话总结：

> 本轮先把 `DoE` 变成一个统一、可输出 protocol-aware 样本的采样层，再在下一轮把 `analysis` 改造成 `SampleSet` 的消费者。
