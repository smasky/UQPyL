# Surrogate 训练/调参/数据分层落地清单

## 目的

这份清单用于把 `surrogate` 模块中与训练、内部超参数优化、外部调参和训练数据管理相关的职责边界正式收拢。

本轮不追求一次性完成所有 surrogate 模型迁移，而是优先把以下四个核心问题落地：

1. `fit()`、模型内部训练、内部超参数优化、`AutoTuner` 的职责边界明确。
2. `xTrain / yTrain` 的数据语义明确，不再混用 raw data 与 prepared data。
3. `AutoTuner` 不再依赖模型私有实现细节。
4. `GPR / KRG` 的内部优化器与 `AutoTuner` 的外层调参器形成清晰分层。

## 设计结论

后续实现以以下约定为准：

- `fit(xTrain, yTrain)`：用户正式入口，输入 raw data。
- `fitModel(xTrain, yTrain)`：模型本体训练入口，输入 prepared data。
- `fitHyper(xTrain, yTrain)`：模型内部超参数求解入口，仅部分模型需要，输入 prepared data。
- `predict(xPred, returnStd=False, returnVar=False)`：预测入口，输入 raw `xPred`。
- `self.xTrain / self.yTrain`：统一约定为 prepared training data。

数据语义按三层区分：

- raw data：用户输入原始数据
- prepared data：完成公共预处理后的训练数据
- split data：为 tuner / validation 划分出来的 prepared 子集

关于 `GPR / KRG` 一类模型的内部优化器与 `AutoTuner` 的关系，采用尽量简单的两种模式：

- `separate`
  - 保持原来的分层逻辑
  - 模型内部默认超参数仍由内部优化器处理
  - `AutoTuner` 只调外层明确交给它的参数
- `joint`
  - 用户决定联合调参时，这一批参数全部交给外部调参器
  - 模型内部完全停掉超参数优化
  - 本轮训练直接执行 `fitModel(...)`

关于 `GPR / KRG` 内部优化器的分类，当前保留 `MP / EA` 两个家族名，作为后续扩展口；但要明确：

- 当前 `MP` 家族的实际落地实现是 `Boxmin`
- 当前 `EA` 家族的实际落地实现是 evolutionary algorithms
- 因此本轮代码和文档都不再暗示“任意 MP 算法已经被正式支持”

因此实现约束是：

- `joint` 模式下，不进入 `fitHyper(...)`
- `joint` 模式下，模型内部不再修改超参数
- 所有联合调参数统一由外部 `AutoTuner` 负责

关于 `kernel` 与 kernel 相关超参数，本轮进一步统一为“统一参数表 + 条件生效参数”设计：

- `kernel` 也进入统一参数系统，视为一个离散可枚举参数
- `Setting` 可以维护一份比当前 kernel 实际需求更大的参数全集
- 当前 `kernel` 选定后，只有其中一部分参数真正生效
- 对当前 `kernel` 不适用的参数，训练落参时直接跳过，不视为错误
- 候选参数落地时，必须先切换 `kernel`，再应用其余参数

这意味着本轮不把 `kernel` 单独抽成一套结构搜索框架，而是采用更直接的约定：

- 用户可以把 `kernel` 和其他参数一起交给 `AutoTuner`
- `AutoTuner` 不需要理解每个 kernel 的内部细节
- 模型本体负责判断“当前 kernel 下哪些参数有效”
- `joint` 模式下，只要外部已经调 `kernel` 或 kernel 超参数，内部优化就必须完全关闭

## 范围

优先触达文件：

- `UQPyL/surrogate/base.py`
- `UQPyL/surrogate/auto_tuner.py`
- `UQPyL/surrogate/gp/gaussian_process.py`
- `UQPyL/surrogate/kriging/kriging.py`

第二批跟随适配文件：

- `UQPyL/surrogate/regression/linear_regression.py`
- `UQPyL/surrogate/regression/polynomial_regression.py`
- `UQPyL/surrogate/rbf/radial_basis_function.py`

## 落地步骤

### Step 1：在基类中正式引入训练分层接口

目标：

- 不再让内部能力依赖各模型私有命名
- 给训练和调参提供稳定框架接口

需要做的事：

1. 在 `SurrogateABC` 中新增 `prepareTrainingData(xTrain, yTrain)`。
2. 将当前 `__check_and_scale__()` 的职责逐步转移到该接口上。
3. 在 `SurrogateABC` 中新增抽象方法 `fitModel(xTrain, yTrain)`。
4. 在 `SurrogateABC` 中提供默认 `fitHyper(xTrain, yTrain)`，默认直接调用 `fitModel(...)`。
5. 在 `SurrogateABC.fit(...)` 默认流程中统一为：
   - raw data -> `prepareTrainingData(...)`
   - prepared data -> `fitHyper(...)`
   - 返回 `self`

验收标准：

- 基类层存在明确的训练分层接口。
- 后续模型不再需要通过 `_fitPure()` 暴露内部训练入口。

### Step 2：明确模型内部保存的数据语义

目标：

- 统一 `self.xTrain / self.yTrain` 的含义

需要做的事：

1. 明确所有核心模型内部的 `self.xTrain / self.yTrain` 都保存 prepared data。
2. 检查 `predict()` 是否统一接受 raw `xPred`，并在内部自行走公共预处理。
3. 检查 `AutoTuner` 是否不再混淆 raw data 与 prepared data。

验收标准：

- 所有核心模型中 `self.xTrain / self.yTrain` 语义一致。
- 不再出现同一模型里一会儿把 `self.xTrain` 当 raw data、一会儿当 scaled data 的情况。

### Step 3：将核心模型的私有训练入口替换为正式接口

目标：

- 用 `fitModel()` 取代当前分散的 `_fitPure()` 私有入口

需要做的事：

1. 在 `regression`、`rbf`、`gp`、`kriging` 中引入 `fitModel(...)`。
2. 将当前 `_fitPure()` 的核心逻辑迁移到 `fitModel(...)`。
3. 过渡期内可以先保留 `_fitPure()`，但内部实现改为直接调用 `fitModel(...)`。
4. 新代码一律不再直接依赖 `_fitPure()`。

验收标准：

- 四类核心模型都提供统一命名的内部训练入口。
- `AutoTuner` 不再直接调用 `_fitPure()`。

### Step 4：为 GPR / KRG 引入正式的 `fitHyper()`

目标：

- 把模型内部超参数优化与外部调参器逻辑正式区分

需要做的事：

1. 在 `GPR` 中将当前 likelihood 超参数搜索流程收进 `fitHyper(...)`。
2. 在 `KRG` 中将当前 theta 优化流程收进 `fitHyper(...)`。
3. `fitModel(...)` 只负责“给定当前参数后的拟合”。
4. `fit()` 统一走：
   - `prepareTrainingData(...)`
   - `fitHyper(...)`
5. `joint` 模式下，模型训练需要绕过 `fitHyper(...)`，直接进入 `fitModel(...)`。
6. 为 kernel 类模型补充“激活参数”语义，至少在实现上满足：
   - 先切 `kernel`
   - 再决定当前哪些参数有效
   - 最后执行 `fitModel(...)`

验收标准：

- `GPR / KRG` 中能清楚区分：
  - 模型本体训练
  - 模型内部超参数优化
- `optimizer` 在 `GPR / KRG` 中的角色明确为“模型内部超参数求解器”。
- `joint` 模式下，内部优化器不会参与训练流程。
- kernel 切换后不会继续错误沿用上一套 kernel 的内部状态。

### Step 5：重构 AutoTuner 的数据流与调用链

目标：

- 让 `AutoTuner` 成为外层调参器，而不是模型内部训练细节的消费者

需要做的事：

1. `AutoTuner` 入口统一接收 raw data。
2. 在 `AutoTuner` 内部先调用 `model.prepareTrainingData(...)`。
3. 后续 train/test split 统一在 prepared data 上进行。
4. 调参循环里调用正式内部接口：
   - 对普通模型：`fitModel(...)`
   - 对需要时也可以统一走框架层约定，而不是私有方法
5. 最终用最佳参数在全量 prepared data 上重新训练。
6. `AutoTuner` 需要支持至少两种调参模式：
   - `separate`
   - `joint`
7. `joint` 模式下，`AutoTuner` 需要显式要求模型跳过内部超参数优化。
8. 当候选参数中包含 `kernel` 时，`AutoTuner`` 需要按“先 kernel、后其余参数”的顺序落参。
9. `AutoTuner` 不要求候选参数在所有 kernel 下都有效；对当前 kernel 不适用的参数允许被模型安全忽略。

验收标准：

- `AutoTuner` 不再依赖 `_fitPure()`。
- `AutoTuner` 的数据流中 raw / prepared / split 三层语义清晰。
- `AutoTuner` 与 `GPR / KRG` 的内部优化器职责边界清晰。
- kernel 变化时，不需要外层同步重建一套完全不同的参数表。

### Step 6：统一文档与命名说明

目标：

- 降低后续维护时的理解成本

需要做的事：

1. 在 `base.py` 中为训练分层方法补简短说明。
2. 在 `AutoTuner` 中明确：
   - 它是外层调参器
   - 它不等价于 `GPR / KRG` 的内部超参数优化器
3. 在核心模型中补足 `fitModel / fitHyper` 的说明。

验收标准：

- 读代码时能直观看出每层职责。
- 不再需要通过上下文猜测 `fit()` 内到底有没有做参数优化。

## 文件级任务清单

### `UQPyL/surrogate/base.py`

- [x] 新增 `prepareTrainingData(...)`
- [x] 新增 `fitModel(...)` 抽象接口
- [x] 新增默认 `fitHyper(...)`
- [x] 调整 `fit()` 的统一默认流程
- [x] 明确 `self.xTrain / self.yTrain` 保存 prepared data
- [x] 新增 `fitState`、`storeTrainingData(...)`、`requireFitted(...)`

### `UQPyL/surrogate/auto_tuner.py`

- [x] 改为先准备数据，再做 split
- [x] 停止依赖 `_fitPure()`
- [x] 改为依赖正式内部训练接口
- [x] 明确 raw / prepared / split 三层数据流
- [x] 支持 `kernel` 作为统一参数表中的离散参数参与调参
- [x] 候选解落参时遵循“先 kernel，后其余参数”
- [x] 支持 `joint / separate` 两种调参模式

### `UQPyL/surrogate/gp/gaussian_process.py`

- [x] 引入 `fitModel(...)`
- [x] 引入 `fitHyper(...)`
- [x] 将当前内部优化器逻辑收进 `fitHyper(...)`
- [x] 保持 `predict(mean/std/var)` 协议不变
- [x] 明确当前 kernel 下参数激活与忽略规则
- [x] 使用 `fitState` 承载预测态中间结果
- [x] 支持 `kernel` choice 注册与切换

### `UQPyL/surrogate/kriging/kriging.py`

- [x] 引入 `fitModel(...)`
- [x] 引入 `fitHyper(...)`
- [x] 将当前 theta 优化逻辑收进 `fitHyper(...)`
- [x] 保持 `predict(mean/std/var)` 协议不变
- [x] 明确当前 kernel 下参数激活与忽略规则
- [x] 使用 `fitState` 承载预测态中间结果
- [x] 支持 `kernel` choice 注册与切换

### `UQPyL/surrogate/regression/*`

- [x] 将 `_fitPure()` 迁移为 `fitModel(...)`
- [x] 继续保持 deterministic 模型无不确定性输出

### `UQPyL/surrogate/rbf/radial_basis_function.py`

- [x] 将 `_fitPure()` 迁移为 `fitModel(...)`
- [x] 保持 kernel-setting 共享关系
- [x] 明确当前 kernel 下参数激活与忽略规则
- [x] 使用 `fitState` 承载预测态中间结果
- [x] 支持 `kernel` choice 注册与切换

## 风险点

### 1. 过渡期双接口并存

在从 `_fitPure()` 迁移到 `fitModel()` 的过程中，短期内可能会出现两个入口并存。需要明确：

- 新代码只用 `fitModel()`
- `_fitPure()` 只作为短期兼容包装，不再继续扩展

### 2. GPR / KRG 的优化器职责混淆

如果 `fitHyper()` 与 `AutoTuner` 边界不清，后面仍会重复做两层近似相同的搜索逻辑。需要严格区分：

- 模型内部超参数优化
- 用户外层调参

尤其要避免：

- 外层已经联合调参
- 内层又重新优化同一批参数

因此 `joint` 模式下必须直接停掉内部优化。

### 3. kernel 切换带来的条件参数行为

如果把 `kernel` 也纳入统一参数表，就会自然出现“同一份参数表里，部分参数只在特定 kernel 下生效”的情况。这里要明确这不是异常，而是正式设计的一部分：

- `kernel` 可以像普通参数一样被调
- 其他参数允许保留在参数全集中
- 当前 `kernel` 不支持的参数直接忽略

真正需要避免的是：

- 切换 kernel 后，旧 kernel 的内部状态残留
- 参数落参顺序错误，导致后续参数按错误的 kernel 语义解释

因此实现时要优先保证：

- 先切 `kernel`
- 再刷新 kernel 相关状态
- 再应用其余有效参数

### 4. prepared data 的语义不彻底统一

如果某些模型仍在内部偷偷重复做 scaler / feature transform，就会继续破坏数据分层。需要在改动时重点检查。

## 完成标志

满足以下条件即可认为这一专题初步完成：

1. 核心四类模型都具备统一的内部训练接口。
2. `AutoTuner` 不再依赖 `_fitPure()`。
3. `GPR / KRG` 中模型训练与内部超参数优化正式分层。
4. `self.xTrain / self.yTrain` 在核心模型中语义一致。
5. `kernel` 可以作为统一参数表中的一员参与调参，且条件参数行为清晰。
6. 核心测试通过，且训练/预测/调参流程没有新增接口混乱。

## 推荐实施顺序

建议按如下顺序推进：

1. `base.py`
2. `gp/gaussian_process.py`
3. `kriging/kriging.py`
4. `auto_tuner.py`
5. `regression/*`
6. `rbf/radial_basis_function.py`

原因是：

- 先定基类协议
- 再收最复杂的 `GPR / KRG`
- 再让 `AutoTuner` 接正式接口
- 最后把其余核心模型补齐到新接口
