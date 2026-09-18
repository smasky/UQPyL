# UQPyL `problem` 改造计划

## 目标

本次改造有两个核心目标：

### 1. 收口 `evaluate`

对外主接口中，不再显式开放：

```python
Problem(..., evaluate=...)
ModelProblem(..., evaluate=...)
```

后续规则改为：

- 普通用户不能通过构造参数传入 `evaluate`
- 高级用户如需完全接管评估流程，只能通过继承覆写 `evaluate()`

### 2. 明确底层演进方向

`problem` 模块后续的底层组织，按下面这条链收敛：

```text
X -> Space -> Simulator? -> Evaluator -> Eval
```

其中：

- `Space` 必选
- `Evaluator` 必选
- `Simulator` 可选

---

## 最终规则

### `Problem`

主链：

```text
X -> objs/cons
```

允许用户定义：

- `objFunc(X)`
- `conFunc(X)`

不再允许：

- 构造参数 `evaluate=...`

如确实需要自定义完整评估流程：

- 通过继承覆写 `evaluate()`

### `ModelProblem`

主链：

```text
X -> sim -> objs/cons
```

允许用户定义：

- `simFunc(X)`
- `objFunc(X, context)`
- `conFunc(X, context)`

不再允许：

- 构造参数 `evaluate=...`

如确实需要自定义完整评估流程：

- 通过继承覆写 `evaluate()`

---

## 为什么这样定

### 1. 主接口更清楚

去掉构造参数 `evaluate=...` 之后，用户层只剩两种正式模式：

- `Problem`
- `ModelProblem`

而不是把组合式 `evaluate` 继续作为第三种平级入口。

### 2. `target` 语义更容易收口

当前只要显式传入 `evaluate`，`target` 基本就退化成结果过滤器。

收掉构造式 `evaluate` 后：

- `Problem` 的 `target` 可以更稳定地理解为目标/约束分支选择
- `ModelProblem` 的 `target` 可以明确为在已有 `sim` 前提下选择输出块

### 3. 高级扩展能力仍然保留

通过继承覆写 `evaluate()`，仍然可以支持：

- 特殊黑箱问题
- 高级集成适配器
- 非标准评估流程

但这条能力不再出现在主接口上。

### 4. 更适合未来与 hydroPilot 联动

因为 hydroPilot 的本质结构本来就更接近：

```text
Space + Simulator + Evaluator
```

UQPyL 把主接口规则收紧后，底层再引入这一分层会更自然。

---

## 技术路线

当前正式确定的技术路线是：

### 初级接口：各类 `Func`

面向大多数用户，继续保留函数式入口。

#### `Problem`

- `objFunc(X)`
- `conFunc(X)`

#### `ModelProblem`

- `simFunc(X)`
- `objFunc(X, context)`
- `conFunc(X, context)`

这一层强调：

- 简单
- 直接
- 学习成本低

### 高级接口：`evaluator`

面向复杂问题和集成场景，优先提供 evaluator 扩展能力。

需要时再进一步自定义 simulator。

这一层强调：

- 更强控制力
- 更清楚的职责边界
- 更适合与 hydroPilot 这类外部系统联动

### `evaluate()` 的定位

`evaluate()` 不再作为主接口显式开放构造参数。

后续定位为：

- 保留类方法覆写能力
- 不作为推荐扩展路线
- 仅作为最后的 escape hatch

一句话概括：

**初级用户使用各类 `Func`，高级用户优先使用 `evaluator`，只有极特殊场景才直接覆写 `evaluate()`。**

---

## 总体架构方向

本次不是立刻把所有类推翻重写，而是按下面的目标架构逐步迁移。

## 目标架构

```text
Problem
  - space
  - evaluator
  - simulator | None

ModelProblem
  - Problem 的语义包装类

Space
SimulatorBase
EvaluatorBase
SimulatorResult
Eval
```

### `Space`

负责：

- 输入维度
- 上下界
- 变量类型
- 输入规整

### `Simulator`

负责：

- 批量运行
- 仿真产物提取
- 中间结果组织

### `Evaluator`

负责：

- 从 `X` 或 `SimulatorResult` 计算 `objs`
- 计算 `cons`
- 必要时组织 `sim`
- 返回 `Eval`

---

## 改造原则

### 原则一

**外部规则先收紧，内部结构再统一。**

即先把用户接口规则定清楚，再做底层抽象演进。

### 原则二

**普通问题继续保持简单。**

不能因为要支持 simulator/evaluator 分层，就让普通 `Problem` 用户也被迫理解这些概念。

### 原则三

**高级扩展通过继承暴露，不通过主构造参数暴露。**

### 原则四

**所有自定义 `evaluate()` 都必须受统一协议校验。**

也就是说，允许覆写不等于放弃约束。

---

## 分阶段计划

## 阶段一：规则收口

目标：

- 从主接口中移除显式 `evaluate=...`
- 保留 `evaluate()` 方法本身
- 明确两种模式的用户函数签名

具体动作：

1. 修改 `Problem` 构造逻辑
   - 删除 `evaluate` 构造参数
   - 删除与 `_eval_fn` 相关的主路径逻辑

2. 修改 `ModelProblem` 构造逻辑
   - 删除 `evaluate` 构造参数
   - 删除 `_eval_fn` 主路径逻辑

3. 保留实例方法 `evaluate()`
   - 作为正式公共方法继续存在
   - 作为子类可覆写扩展点存在

4. 更新构造校验规则
   - `Problem`：只接受 `objFunc` / `objFunc + conFunc`
   - `ModelProblem`：只接受 `simFunc + objFunc` 或 `simFunc + objFunc + conFunc`

---

## 阶段二：统一 `evaluate()` 校验出口

目标：

- 无论 `evaluate()` 来自基类默认实现还是子类覆写，都统一校验

具体动作：

1. 增加统一结果校验逻辑
   - 返回值必须是 `Eval`
   - `objs` / `cons` / `sim` 必须满足 shape 规则
   - 第一维必须等于 `n_samples`

2. 对 `Problem` 增加额外约束
   - `sim` 必须为 `None`
   - `objs` / `cons` 列数必须匹配 `nObj` / `nCon`

3. 对 `ModelProblem` 增加额外约束
   - `sim` 必须存在于正式仿真路径中
   - `objFunc(X, context)` / `conFunc(X, context)` 的结果维度必须合法

4. 对 `target` 过滤后的结果继续校验

---

## 阶段三：文档与测试同步收口

目标：

- 仓库内不再把 `evaluate=...` 当成正式推荐接口

具体动作：

1. 修改问题模块文档
   - 删除 `evaluate=...` 的主用法描述
   - 只保留“继承覆写 `evaluate()`”作为高级扩展说明

2. 修改 API 文档
   - 从构造签名中移除 `evaluate`

3. 修改 README 与示例
   - 普通问题只展示 `objFunc` / `conFunc`
   - 仿真问题只展示 `simFunc + objFunc/conFunc`

4. 修改测试
   - 将所有 `Problem(..., evaluate=...)` 测试改为：
     - `objFunc/conFunc` 版本
     - 或子类覆写 `evaluate()` 版本

---

## 阶段四：底层引入 `Space + Simulator + Evaluator`

目标：

- 在不破坏用户层规则的前提下，把底层逐步演进到组件式结构

具体动作：

1. 新增基础协议对象
   - `simulator_base.py`
   - `evaluator_base.py`
   - `simulator_result.py`

2. 为普通问题提供默认 evaluator
   - `dir_evaluator.py`
   - 类名：`DirEvaluator`

3. 为仿真问题提供默认 evaluator
   - `sim_evaluator.py`
   - 类名：`SimEvaluator`

4. 让 `Problem` 内部逐步转成：
   - `space`
   - `simulator`
   - `evaluator`

5. 让 `ModelProblem` 逐步转成包装类
   - 对外保留现有语义
   - 对内转成 `Problem + simulator + evaluator`

---

## 阶段五：为外部集成预留适配位

目标：

- 支持像 hydroPilot 这样的外部系统接入
- 但不让 `problem` 核心层依赖 hydroPilot

具体动作：

1. UQPyL 核心只认：
   - `Space`
   - `SimulatorBase`
   - `EvaluatorBase`
   - 默认实现可逐步收敛为 `DirEvaluator` / `SimEvaluator`

2. 外部系统在适配层实现：
   - 自己的 simulator
   - 自己的 evaluator

3. 不在 `problem` 核心里写死 hydroPilot 逻辑

---

## 影响评估

### 对算法主体的影响

影响很小。

原因：

- 优化、分析、推断、率定模块主要依赖 `problem.evaluate()`、`objFunc()`、`conFunc()`、`simFunc()`
- 它们不直接依赖构造参数 `evaluate=...`

### 对问题定义层的影响

影响明确。

需要改：

- `problem` 构造签名
- 文档
- 测试
- 少量依赖 `evaluate=...` 的适配层

### 对外部扩展的影响

影响可控。

因为：

- 特殊用户仍然可以通过继承覆写 `evaluate()`
- 并没有把能力彻底删掉

---

## 需要特别明确的规则

### 规则 1

`Problem` 不再显式接受 `evaluate=...`

### 规则 2

`ModelProblem` 不再显式接受 `evaluate=...`

### 规则 3

如需完整自定义评估流程，只能继承后覆写 `evaluate()`

### 规则 4

所有 `evaluate()` 的最终返回都必须是合法 `Eval`

### 规则 5

`ModelProblem.evaluate()` 的正式仿真路径中，必须先有 `sim`

---

## 最终建议

建议把本次改造定成：

### 对外

- 收掉构造参数 `evaluate`
- 保留两种正式模式：
  - `Problem`
  - `ModelProblem`

### 对内

- 逐步向 `Space + Simulator + Evaluator` 收敛
- 组件命名采用：
  - `SimulatorBase`
  - `DirEvaluator`
  - `SimEvaluator`

### 对高级扩展

- 只通过继承覆写 `evaluate()` 开放

一句话总结：

**先把 `evaluate` 从主接口中收掉，把用户层规则定清楚；再把底层逐步演进到 `Space + Simulator + Evaluator`，这样既能保持 UQPyL 的简洁，也能为仿真型和外部集成问题预留稳定结构。**
