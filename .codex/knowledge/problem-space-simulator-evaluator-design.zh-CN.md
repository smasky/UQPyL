# UQPyL Problem 改造方案

## 目标

将当前 `Problem` / `ModelProblem` 的二分结构，收敛为一个更统一的装配式问题抽象：

```text
X -> Space -> Simulator? -> Evaluator -> Eval
```

设计目标有三个：

- 保持普通问题的易用性
- 让仿真问题有正式位置
- 为 hydroPilot 联动预留接口，但不和 hydroPilot 深度耦合

## 总体结论

建议把 `Problem` 从“问题类型类”改成“问题装配器”。

核心思想：

- `Space` 必选
- `Evaluator` 必选
- `Simulator` 可选

这样：

- 普通问题：只有 `Space + Evaluator`
- 仿真问题：使用 `Space + Simulator + Evaluator`

`ModelProblem` 不建议继续作为根抽象，建议保留为便捷包装类。

## 三层职责

### 1. `Space`

负责输入空间定义与输入规整：

- `nInput`
- `lb` / `ub`
- `varType` / `varSet`
- `xLabels`
- `validate`
- `unit_to_space`
- `apply_var_type`

这一层基本沿用现有 `Space` 即可。

### 2. `Simulator`

负责把批量输入 `X` 变成原始运行结果。

只负责：

- 参数样本执行
- 模型运行
- 中间结果提取
- 原始产物组织

不负责：

- 目标值定义
- 约束定义
- 优化方向
- 指标解释

### 3. `Evaluator`

负责把 `X` 或 `SimulatorResult` 解释成 `Eval`。

只负责：

- 计算 `objs`
- 计算 `cons`
- 需要时将 `sim` 放入 `Eval`
- 统一输出语义

## 建议数据协议

### `SimulatorResult`

建议新增中间结果协议：

```python
@dataclass
class SimulatorResult:
    sim: np.ndarray | None = None
    obs: np.ndarray | None = None
    mask: np.ndarray | None = None
    extras: dict | None = None
```

说明：

- `sim`：标准仿真结果
- `obs` / `mask`：供仿真型 evaluator 使用
- `extras`：留给运行期附加结果，不进入 UQPyL 核心语义

### `Eval`

建议保留当前主字段，并补一个扩展位：

```python
@dataclass
class Eval:
    objs: np.ndarray | None = None
    cons: np.ndarray | None = None
    sim: np.ndarray | None = None
    extras: dict | None = None
```

其中：

- `objs` / `cons` / `sim` 仍是正式协议主字段
- `extras` 只作为附加位，不让下游核心模块依赖

## 建议接口

### `simulator_base.py`

```python
class SimulatorBase:
    def run(self, X) -> SimulatorResult:
        raise NotImplementedError
```

### `evaluator_base.py`

```python
class EvaluatorBase:
    def evaluate(self, X, simulatorResult=None) -> Eval:
        raise NotImplementedError
```

### 新 `Problem`

```python
class Problem:
    def __init__(
        self,
        *,
        space,
        evaluator,
        simulator=None,
        nObj=None,
        nCon=0,
        optType="min",
        name=None,
        objLabels=None,
        conLabels=None,
    ):
        ...
```

核心规则：

- `space` 必填
- `evaluator` 必填
- `simulator` 可空

### `evaluate()`

```python
def evaluate(self, X, target=None):
    X = self.space.validate(X)

    if self.simulator is None:
        result = self.evaluator.evaluate(X, simulatorResult=None)
    else:
        simulatorResult = self.simulator.run(X)
        result = self.evaluator.evaluate(X, simulatorResult=simulatorResult)

    return self._filter_eval_by_target(result, target)
```

## 两类问题如何映射

### 普通问题

普通问题是这个新结构的最简单特例：

```text
X -> Evaluator -> Eval(objs, cons)
```

即：

- `simulator = None`
- `evaluator = DirectEvaluator`

这类问题不需要显式 simulator。

### 仿真问题

仿真问题使用完整结构：

```text
X -> Simulator -> SimulatorResult -> Evaluator -> Eval
```

即：

- `simulator = ExternalSimulator`
- `evaluator = SimulationEvaluator`

## `ModelProblem` 的定位

建议保留 `ModelProblem`，但降级为语义包装类，不再作为根抽象。

它的作用变成：

- 为水文/率定场景提供便捷构造
- 自动组织 `obs` / `mask` / `seriesLabels`
- 内部转成 `Problem + Simulator + Evaluator`

也就是说：

```text
ModelProblem = Problem 的便捷工厂 / 包装类
```

## 与 hydroPilot 的联动方式

原则：

- UQPyL 不依赖 hydroPilot 内部实现
- UQPyL 只依赖 `SimulatorBase` / `EvaluatorBase` 协议
- hydroPilot 在适配层实现对应对象

最自然的关系是：

- hydroPilot 参数空间 -> `Space`
- hydroPilot 执行链 -> `Simulator`
- hydroPilot evaluator -> `Evaluator`

这样 UQPyL 只管装配，不管 hydroPilot 内部怎么跑模型。

## 为什么这个设计适合 hydroPilot

hydroPilot 本身已经天然分层：

- 参数空间
- 参数写入与执行
- 序列提取
- evaluator

所以 UQPyL 如果继续坚持“直接问题 / 模型问题”二分，会显得偏结果导向；
如果换成 `Space + Simulator + Evaluator`，就和 hydroPilot 的真实结构同构。

这不是为了耦合 hydroPilot，而是让 UQPyL 的问题抽象更贴近真实仿真工作流。

## 兼容策略

建议分三步迁移。

### 第一步

新增：

- `simulator_base.py`
- `evaluator_base.py`
- `simulator_result.py`

同时给 `Eval` 增加 `extras`。

### 第二步

让 `Problem` 同时支持两种构造方式：

- 旧式：`objFunc` / `conFunc` / `evaluate`
- 新式：`space + simulator + evaluator`

旧接口内部自动转成默认 evaluator。

### 第三步

把 `ModelProblem` 调整为包装类：

- 外部 API 尽量不变
- 内部不再作为独立根协议

## 建议默认实现

为了不破坏普通用户体验，建议内置两个默认 evaluator：

### `direct_evaluator.py`

用于普通问题：

- `objFunc`
- `conFunc`
- `evaluate`

### `simulation_evaluator.py`

用于仿真问题：

- 接收 `SimulatorResult`
- 从 `sim` / `obs` / `mask` 计算结果

## 最终建议

建议最终形成下面这套结构：

```text
Problem
  - space
  - simulator | None
  - evaluator

ModelProblem
  - Problem 的包装类

Space
SimulatorBase
EvaluatorBase
SimulatorResult
Eval
```

一句话总结：

**把 `Problem` 设计成统一装配入口，把普通问题视为“无 simulator 的特例”，把仿真问题视为“有 simulator 的问题”，这样既能保持 UQPyL 简洁，也能自然接住 hydroPilot。**
