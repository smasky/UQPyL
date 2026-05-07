# Surrogate 架构重构计划

## 背景

`surrogate` 模块目前已经积累了较多历史实现，单看每个模型大多还能工作，但从整体上看，已经出现了比较明显的架构漂移：

- 公共协议没有完全收敛，`fit / predict` 的输入输出语义并不统一。
- 预处理逻辑散落在基类和各模型内部，`shape / scaler / polyFeature` 处理重复且不一致。
- 参数定义、参数当前值、参数搜索空间三种职责还混在同一套机制里。
- `AutoTuner`、`MultiSurrogate`、`KRG / GPR` 这类高阶能力和基础模型层耦合较深。
- 对 `problem / optimization` 的历史接口兼容痕迹较多，导致 surrogate 本体不够“内聚”。

如果继续按“遇到一个模型修一个模型”的方式推进，后续在对接 `optimization.expensive`、`analysis`、`inference` 时会持续返工。因此本轮建议先从 `surrogate` 总体架构入手，先收敛内部协议，再逐步迁移具体模型，最后再处理外部模块对接。

## 目标

本轮重构的核心目标不是立刻重写所有 surrogate 模型，而是先把 `surrogate` 模块整理成一个边界清晰、协议稳定、便于继续演进的子系统。

目标可以概括为以下几点：

1. 统一 surrogate 公共协议，明确训练、预测、不确定性返回和多输出行为。
2. 把数据预处理流程从模型算法实现中抽离，收敛到公共层。
3. 把超参数当前值与可调搜索空间解耦，减少调参逻辑对模型核心的侵入。
4. 区分“基础模型层”和“组合/调参/适配层”，降低模型之间的横向耦合。
5. 为后续对接 `optimization / analysis / inference` 提供稳定的内部接口，而不是让外部模块直接依赖具体模型细节。

## 非目标

为了控制本轮范围，以下内容不作为第一阶段目标：

- 不追求一次性重写所有 surrogate 模型实现。
- 不要求本轮解决所有数值稳定性和算法性能问题。
- 不要求本轮完成所有外部模块对接。
- 不专门做一次全仓库清理旧接口，只在触达文件中顺手收敛。

## 总体分层

建议把 `surrogate` 按职责拆成五层，从内到外逐步稳定。

### 1. Core Protocol

这是最内层，也是必须先稳定的一层。所有 surrogate 模型都应遵守同一套公共协议。

建议统一约束：

- `fit(xTrain, yTrain) -> self`
- `predict(xPred, ...) -> np.ndarray | tuple[...]`
- 输入统一接受 `np.ndarray`
- 单输出统一返回 `(n, 1)`
- 多输出统一返回 `(n, m)`
- 如果模型支持不确定性输出，额外返回值的形状也必须稳定

这一层的核心载体是：

- `SurrogateABC`
- 公共输入输出校验逻辑
- 统一的数据变换入口

目标是让“一个 surrogate 是怎么被外部使用的”先稳定下来。

### 2. Data Pipeline

这一层专门负责数据预处理，不负责算法本身。

建议从基类中明确抽出以下职责：

- `validate_x_y`
- `prepare_train_x`
- `prepare_train_y`
- `prepare_predict_x`
- `inverse_transform_y`

应统一处理：

- `x / y` shape 规范
- scaler 的 fit/transform/inverse_transform
- polynomial feature 变换
- 单样本输入与批量输入兼容

目标是让各模型只关心“算法如何拟合/预测”，而不再各自维护一套输入清洗逻辑。

### 3. Hyperparameter System

这一层负责统一参数系统，重点是把“模型当前参数”和“可调参数空间”从概念上分开。

当前 `Setting` 可以继续保留，但建议逐步明确成两类职责：

- 参数当前值：模型训练和预测时真实读取的配置
- 参数搜索空间：供 `AutoTuner` 或其他优化器使用的可调边界、类型、log 标记、离散集合

建议后续把参数系统收敛为：

- 模型定义参数
- tunable 参数元信息
- 参数读写工具

这样模型内部就不用知道调参器如何搜索，调参器也不用知道模型具体如何存储参数。

在当前阶段，还需要额外满足 `kernel` 的约束：

- `Setting` 不能只服务 surrogate 本体，也要能容纳 kernel 参数
- `Setting` 应被视为 surrogate 与 kernel 共享的参数注册表
- `Setting` 需要支持参数归属信息，至少在设计上能区分 `model` 与 `kernel`
- `Setting` 需要支持延迟初始化参数，例如 `theta / l` 这类在 `initialize(nInput)` 后按维度展开的参数

因此，本轮不先强行统一三套 kernel，而是把 kernel 当作 `Setting` 设计的输入约束来处理。

### 4. Model Families

当公共协议和数据流程稳定后，再整理模型族结构。

建议逻辑上分为以下几类：

- `deterministic`
  - `LinearRegression`
  - `PolynomialRegression`
  - `RBF`
  - `MARS`
  - `FNN`
- `probabilistic`
  - `GPR`
  - `KRG`
- `kernel_methods`
  - `SVR`
- `ensemble`
  - `MultiSurrogate`
  - 未来真正的 bagging / boosting / weighting 组合模型

这里的重点不是一定要立即改目录结构，而是先让代码职责上体现这种分组。

尤其要注意：

- `KRG` 和 `GPR` 都属于“可输出不确定性”的 surrogate，应尽量共用一套上层协议。
- `MultiSurrogate` 应被视为组合层能力，而不是基础模型本体。
- `AutoTuner` 应该依赖公共参数协议，而不是依赖某几个模型的实现细节。

### 5. Integration Adapters

最外层才是和其他模块的对接层。

建议把对接逻辑视为 adapter，而不是继续塞回 surrogate 核心层。后续可以逐步整理为：

- 面向 `optimization.expensive` 的 surrogate adapter
- 面向 `analysis` 的 refit adapter
- 面向 `inference` 的 emulator / likelihood adapter

原则是：

- `surrogate` 本体尽量不直接感知外部模块的运行时细节。
- 外部模块通过稳定协议消费 surrogate。
- 外部模块需要的特殊语义通过 adapter 解决，而不是让每个 surrogate 模型单独兼容。

## 推荐重构顺序

### 第一阶段：基础协议收敛

优先整理：

- `base.py`
- `setting.py`
- `auto_tuner.py`
- `MultiSurrogate`

第一阶段的目标：

- 明确单输出/多输出 shape 协议
- 明确 scaler 和 feature transform 的统一入口
- 明确参数读取和参数空间定义的边界
- 明确调参器与 surrogate 基类的交互方式

这一阶段完成后，模型实现还可以有历史差异，但外围协议必须尽量稳定。

### 第二阶段：核心模型优先迁移

建议优先处理：

- `RBF`
- `LinearRegression`
- `GPR`
- `KRG`

原因：

- 这几个模型覆盖了最核心的 surrogate 使用路径。
- 它们分别代表了 deterministic、probabilistic、kernel-style 数值实现的几种典型模式。
- `KRG / GPR` 还可以提前推动不确定性输出协议统一。

这一阶段重点不是性能优化，而是让这些模型完全落到新公共协议上。

### 第三阶段：边缘与复杂模型迁移

后续再处理：

- `SVR`
- `MARS`
- `FNN`
- 其余回归模型
- 未完成的 ensemble 实现

这一阶段允许逐个推进，不需要一步到位。

### 第四阶段：外部模块对接

当 surrogate 内部协议稳定后，再统一处理：

- `optimization.expensive`
- `analysis`
- `inference`

目标是让这些模块依赖的是 surrogate 的稳定协议，而不是某个模型的历史返回格式。

## 关键设计建议

### 关于 `predict` 协议

这是后续最容易失控的点之一，建议尽早统一。

可选方向有两种：

方案 A：保留当前轻量风格

- 默认 `predict(xPred)` 只返回预测值
- 概率模型用额外参数请求不确定性，如 `returnStd`、`returnVar`

方案 B：统一返回预测结果对象

- 比如统一返回 `Prediction`，其中包含 `mean / std / mse / extra`
- 比如统一返回 `Prediction`，其中包含 `mean / std / var / extra`

如果短期目标是最小改造成本，建议先走方案 A；如果后面打算让 `surrogate` 深度服务 `optimization.expensive` 和 `inference`，则方案 B 长期更稳。

当前建议的第一阶段收口方式如下：

- 正式协议使用 `predict(xPred, returnStd=False, returnVar=False)`
- 默认只返回 `mean`
- `returnStd=True` 时返回 `(mean, std)`
- `returnVar=True` 时返回 `(mean, var)`
- `returnStd` 与 `returnVar` 互斥
- 不支持不确定性的模型，如果请求 `std / var`，应显式报错

- 旧代码中名为 `mse` 的不确定性返回，在架构层统一理解为 `var`
- 新协议中不再把不确定性正式命名为 `mse`

### 关于多输出

短期建议仍然允许：

- 大多数基础模型先保持单输出实现
- 多输出通过 `MultiSurrogate` 组合完成

这样更符合当前模块的实际状态，也能降低第一轮重构风险。

长期再考虑是否为部分模型增加原生多输出支持。

### 关于参数系统

建议保持 `Setting` 作为过渡期实现，但逐步限制它的职责边界：

- 可以继续保存参数值和调参元信息
- 不建议继续承担过多模型外部协议职责
- 后续如果需要，可以在不破坏外部接口的前提下把内部结构再整理

### 关于目录结构

本轮不要求立即重排目录，但建议后续迁移时遵循“先职责收敛，再文件搬迁”的原则。

如果先搬目录而协议未稳定，反而更容易把问题扩散到更多文件。

## 风险点

### 1. 历史模型实现差异较大

不同 surrogate 模型的历史实现风格差异明显，尤其是：

- 参数命名
- `predict` 返回值
- 调参流程
- 内部预处理逻辑

因此不适合一次性强行抽象过度，应先收最小公共协议。

### 2. `KRG / GPR` 的不确定性协议不一致

这两类模型后续会直接影响 `EGO / ASMO` 一类算法的可用性，因此必须尽早统一返回语义。

### 3. `AutoTuner` 容易反向污染模型层

如果调参器直接依赖模型内部细节，后面每改一个模型都可能波及调参逻辑。应尽量让 `AutoTuner` 只依赖公共参数协议和 `fit / predict` 协议。

### 4. 外部模块可能依赖旧行为

即便本轮先不处理对接，也要预期后续 `optimization.expensive`、`analysis` 等模块会暴露出一些隐式依赖。应把这些问题留到 adapter 层统一处理，而不是继续回灌到 surrogate 核心层。

## 建议的阶段性验收标准

### 第一阶段验收

- `SurrogateABC` 输入输出协议明确
- `MultiSurrogate` 行为稳定
- `AutoTuner` 只依赖新协议
- 代表性模型能完成基础 `fit / predict` 烟雾测试

### 第二阶段验收

- `RBF / LinearRegression / GPR / KRG` 全部迁移到统一协议
- 概率模型不确定性输出形状一致
- 单输出场景的行为稳定可预期

### 第三阶段验收

- 其余 surrogate 模型逐步迁移完成
- 模块内不再出现大范围新旧协议混用

### 第四阶段验收

- `optimization.expensive` 等外部模块通过稳定接口使用 surrogate
- 旧兼容逻辑尽量收缩到 adapter 层

## 当前建议的下一步

如果按本计划推进，建议下一步直接进入第一阶段，优先整理以下文件：

- `UQPyL/surrogate/base.py`
- `UQPyL/surrogate/setting.py`
- `UQPyL/surrogate/auto_tuner.py`

同时把 `MultiSurrogate` 明确定位为“组合层过渡实现”，先保证协议稳定，再决定后续是否继续扩展 ensemble 体系。

这一步完成后，再开始迁移第一批核心模型，而不是一开始就全面铺开到所有 surrogate 实现。
