# DOE / Analysis 目录重构计划

## 结论

当前重构方向定为：

| 模块 | 职责 |
|---|---|
| `UQPyL.doe` | 只保留通用 DOE / 通用样本生成 |
| `UQPyL.analysis.designs` | 承接分析专用样本设计 |
| `UQPyL.analysis` | 只负责分析计算、结果组织、参数校验 |

本轮不新增顶层 `sampling` 模块。

---

## 目标

这轮计划只解决 4 件事：

| 目标 | 说明 |
|---|---|
| 收紧 `doe` 边界 | 不再混入分析专用结构化采样 |
| 给 `analysis` 增加 `designs` 子层 | 专用设计有明确落点 |
| 建立两套基类职责 | `doe.base` 和 `analysis.designs.base` 分开 |
| 清理 `analysis` 的采样职责 | 分析器不再自己内嵌样本构造细节 |

---

## 目录方案

目标目录结构如下：

```text
UQPyL/
  doe/
    __init__.py
    base.py
    random.py
    lhs.py
    full_fact.py
    sobol.py

  analysis/
    __init__.py
    base.py
    designs/
      __init__.py
      base.py
      saltelli.py
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

说明如下：

| 路径 | 定位 |
|---|---|
| `doe/` | 普通样本采样器 |
| `analysis/designs/` | 分析方法专用设计器 |
| `analysis/*.py` | 分析指标计算器 |

---

## 模块边界

### 1. `doe`

保留对象：

| 对象 | 处理 |
|---|---|
| `Random` | 保留 |
| `LHS` | 保留 |
| `FFD` | 保留 |
| `SobolSequence` | 保留 |

移出对象：

| 对象 | 去向 |
|---|---|
| `SaltelliSequence` | `analysis/designs/saltelli.py` |
| `FASTSequence` | `analysis/designs/fast.py` |
| `MorrisSequence` | `analysis/designs/morris.py` |

结论：

> `doe` 只负责返回普通样本点，不再负责为特定分析公式构造结构化样本。

---

### 2. `analysis/designs`

这里放“分析专用设计器”，不是通用采样器。

收纳范围：

| 设计器 | 服务对象 |
|---|---|
| `SaltelliDesign` | `analysis.Sobol` |
| `FASTDesign` | `analysis.FAST` |
| `MorrisDesign` | `analysis.Morris` |

这些对象的共同特征：

| 特征 | 含义 |
|---|---|
| 输出不是普通独立样本 | 样本之间带结构关系 |
| 样本数有强公式约束 | 不是简单 `nSamples` 语义 |
| 离开对应分析方法意义不完整 | 本质属于分析设计 |

---

### 3. `analysis`

分析器职责统一收成三类：

| 职责 | 是否保留 |
|---|---|
| 指标计算 | 是 |
| 结果记录 / 输出整理 | 是 |
| 参数校验 | 是 |
| 专用样本构造公式 | 否，移到 `analysis.designs` |
| 通用 DOE 逻辑 | 否，交给 `doe` |

---

## 两套基类职责

### `UQPyL.doe.base`

这个基类只服务通用采样器。

建议职责：

| 职责 | 说明 |
|---|---|
| 初始化随机数生成器 | 统一 `seed -> rng` |
| 统一输入校验 | `problem`、样本数、返回 shape |
| 统一单位空间到物理空间映射 | 调 `problem.unit_to_space` |
| 定义通用 `_generate` 协议 | 子类只生成单位空间样本 |

明确不承担：

| 不承担 | 原因 |
|---|---|
| 结构化样本协议 | 会把通用基类拉歪 |
| 分析方法专用参数语义 | 不属于通用 DOE |

建议语义：

```python
sampler.sample(problem, nSamples, seed=None) -> ndarray[nSamples, nInput]
```

---

### `UQPyL.analysis.designs.base`

这个基类只服务分析专用设计器。

建议职责：

| 职责 | 说明 |
|---|---|
| 初始化随机数生成器 | 与 DOE 一致 |
| 统一 `problem` 映射 | 统一调 `problem.unit_to_space` |
| 定义结构化设计协议 | 输出结构样本而非普通样本 |
| 暴露样本结构约束 | 例如总样本数、分块规则、合法参数范围 |

建议它的语义不要强装成通用 `nSamples`，而应允许各自保留自己的主参数：

| 设计器 | 主参数语义 |
|---|---|
| `SaltelliDesign` | `N`, `secondOrder`, `skipValue`, `scramble` |
| `FASTDesign` | `N`, `M` |
| `MorrisDesign` | `numTrajectory`, `numLevels` |

结论：

> `analysis.designs.base` 是“结构化实验设计基类”，不是 `doe.Sampler` 的子类别名。

---

## `analysis` 内部分类

### 1. 通用采样驱动分析器

这类方法只需要普通样本，不依赖特定结构。

| 类 | 处理方式 |
|---|---|
| `DeltaTest` | 保留 `sampler` 注入 |
| `RSA` | 保留 `sampler` 注入 |
| `MARS` | 保留 `sampler` 注入 |
| `RBDFAST` | 先保留 `sampler` 注入，后续再看是否需要专用设计层 |

规则：

> 这类分析器只依赖 `doe.Sampler`，不自己做单位空间构造。

---

### 2. 专用设计驱动分析器

这类方法要求输入样本具备固定结构。

| 类 | 设计器依赖 |
|---|---|
| `Sobol` | `SaltelliDesign` |
| `FAST` | `FASTDesign` |
| `Morris` | `MorrisDesign` |

规则：

> 这类分析器只调用 design 对象生成样本，不内嵌设计公式。

---

## 需要修掉的现存问题

### 1. 重复映射

当前 `DeltaTest.sample`、`RSA.sample` 等逻辑是：

1. `sampler.sample(problem, ...)`
2. 再次 `problem._transform_unit_X(X)`

这个流程边界是错的。

统一规则改成：

| 规则 | 说明 |
|---|---|
| `doe` / `designs` 的 `sample()` 返回值 | 默认已经是 problem 空间样本 |
| `analysis` 内部再拿到样本 | 不再二次映射 |

---

### 2. 旧接口继续扩散

当前很多地方仍直接使用：

```python
problem._transform_unit_X(X)
```

后续触达时统一改成：

```python
problem.unit_to_space(X)
```

本轮重点是先在 `doe` 和 `analysis/designs` 收口，不做全仓库扫描。

---

### 3. 采样实现重复

当前重复关系：

| 重复逻辑 | 位置 |
|---|---|
| Saltelli 样本构造 | `doe/saltelli.py` 与 `analysis/sobol.py` |
| FAST 样本构造 | `doe/fast.py` 与 `analysis/fast.py` |
| Morris 样本构造 | `doe/morris.py` 与 `analysis/morris.py` |

目标状态：

| 逻辑 | 唯一实现位置 |
|---|---|
| Saltelli design | `analysis/designs/saltelli.py` |
| FAST design | `analysis/designs/fast.py` |
| Morris design | `analysis/designs/morris.py` |

---

## 分阶段实施顺序

### 阶段 1：先收 `doe`

| 动作 | 说明 |
|---|---|
| 重写 `doe/base.py` | 建立通用采样基类 |
| 适配 `Random/LHS/FFD/SobolSequence` | 全部走统一基类 |
| 从 `doe/__init__.py` 降出专用采样器 | 不再公开 `Saltelli/FAST/Morris` |

验收：

> `doe` 内部只剩通用采样器，语义纯净。

---

### 阶段 2：建立 `analysis/designs`

| 动作 | 说明 |
|---|---|
| 新建 `analysis/designs/base.py` | 建立结构化设计基类 |
| 新建 `saltelli.py` | 承接 Sobol 专用设计 |
| 新建 `fast.py` | 承接 FAST 专用设计 |
| 新建 `morris.py` | 承接 Morris 专用设计 |

验收：

> 三种专用设计均有唯一实现位置。

---

### 阶段 3：收 `analysis`

| 动作 | 说明 |
|---|---|
| `Sobol.sample()` 改调 `SaltelliDesign` | 删除内嵌 Saltelli 构造 |
| `FAST.sample()` 改调 `FASTDesign` | 删除内嵌 FAST 构造 |
| `Morris.sample()` 改调 `MorrisDesign` | 删除内嵌轨迹构造 |
| `DeltaTest/RSA/MARS/RBDFAST` 校正 sample 边界 | 去掉二次映射和重复职责 |

验收：

> `analysis` 只保留分析逻辑，不再持有专用采样公式。

---

### 阶段 4：整理导出与测试

| 动作 | 说明 |
|---|---|
| 更新 `doe/__init__.py` | 只导出通用采样器 |
| 更新 `analysis/__init__.py` | 不必默认导出 designs，视对外 API 决定 |
| 补 `doe` 测试 | 基类协议、shape、seed、一致性 |
| 补 `analysis/designs` 测试 | 结构、样本数、参数校验 |
| 补 `analysis` 定向回归测试 | 分析器与 design/sampler 连接正确 |

---

## 本轮不做

| 不做项 | 原因 |
|---|---|
| 新增顶层 `sampling` 模块 | 当前不需要扩大概念层级 |
| 全仓库 import 全量迁移 | 范围太大，先收主链 |
| 全部分析器统一成一种 sample 签名 | 语义不同，先别强行抹平 |
| 对外兼容层一次性做满 | 先把内部结构理顺 |

---

## 验收标准

本轮结构重构完成时，应满足：

| 标准 | 含义 |
|---|---|
| `doe` 只包含通用采样 | 没有专用分析设计残留 |
| `analysis/designs` 成为专用设计唯一入口 | 不再多处重复实现 |
| `analysis` 分析器不内嵌样本构造公式 | 职责单一 |
| `problem.unit_to_space` 成为触达路径上的正式映射接口 | 不继续扩散旧名字 |
| `DeltaTest/RSA/MARS` 这类通用分析器不再二次映射样本 | 边界正确 |

---

## 下一步执行建议

推荐执行顺序固定为：

1. 先改 `doe/base.py`
2. 再新建 `analysis/designs/`
3. 然后迁 `Sobol/FAST/Morris`
4. 最后修 `DeltaTest/RSA/MARS/RBDFAST` 和测试

最终原则：

> `doe` 负责普通样本，`analysis.designs` 负责结构化设计，`analysis` 负责分析计算。
