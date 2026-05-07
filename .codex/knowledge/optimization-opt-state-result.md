# Optimization 架构约定：OptState / OptResult / OptHistory

## 结论

`optimization` 这一轮先统一收敛到三类核心对象：

| 类 | 定位 |
|---|---|
| `OptState` | 算法内部运行态 |
| `OptResult` | `run()` 结束后的对外结果 |
| `OptHistory` | 运行过程历史轨迹 |

主原则：

1. 单目标、多目标、昂贵优化先共用一套主结构。
2. 只暴露 `best` 语义，不额外暴露 `pareto*` 主字段。
3. 不再长期保存 `*_True` 这类真值/显示态镜像字段。
4. `Verbose`、NetCDF、xarray、GUI 都是外围层，不能反向污染主流程。

---

## 命名结论

| 旧名字 | 新名字 |
|---|---|
| `Result` | `OptState` / `OptResult` 分拆 |

命名选择原因：

| 名字 | 结论 |
|---|---|
| `Result` | 太泛，不再保留为核心运行态名字 |
| `OptState` | 接受，作为运行态类名 |
| `OptResult` | 接受，作为返回结果类名 |
| `OptHistory` | 接受，作为历史轨迹类名 |

---

## 三者职责

### 1. `OptState`

职责：  
算法在 `run()` 期间持续读写的内部状态。

特点：

| 项 | 说明 |
|---|---|
| 生命周期 | 运行期间可变 |
| 持有者 | `Algorithm` |
| 面向对象 | 算法内部 |

### 2. `OptResult`

职责：  
算法结束后，对外提供稳定、统一的结果视图。

特点：

| 项 | 说明 |
|---|---|
| 生命周期 | 运行结束后生成 |
| 持有者 | `Algorithm.run()` 返回 |
| 面向对象 | 用户、测试、上层模块 |

### 3. `OptHistory`

职责：  
专门保存优化过程中的历史轨迹，不再把历史字段平铺到顶层。

特点：

| 项 | 说明 |
|---|---|
| 生命周期 | 挂在 `OptState` 中逐步累积 |
| 面向对象 | 结果查询、可视化、导出 |

---

## `best` 语义统一规则

`OptResult` 对外只保留 `best` 主语义。

| 场景 | `best` 的含义 |
|---|---|
| 单目标 | 当前唯一最优解 |
| 多目标 | 当前最优非支配解集 |
| 昂贵优化 | 当前已确认的最优解或最优解集 |

结论：

1. 不单独引入 `paretoDecs`、`paretoObjs` 作为主字段。
2. 多目标中 `best*` 就表示最优解集。

---

## `OptState` 字段清单

`OptState` 最小必要字段如下：

| 字段 | 说明 |
|---|---|
| `bestDecs` | 当前 best 决策变量 |
| `bestObjs` | 当前 best 目标值 |
| `bestCons` | 当前 best 约束值，没有则为 `None` |
| `bestMetric` | 当前 best 辅助指标，如 HV |
| `bestFeasible` | 当前 best 是否可行 |
| `appearFEs` | best 首次出现时的评估次数 |
| `appearIters` | best 首次出现时的迭代数 |
| `currentPop` | 当前活动种群或样本集 |
| `FEs` | 当前函数评估次数 |
| `iters` | 当前迭代次数 |
| `runtime` | 总运行时间 |
| `history` | `OptHistory` 实例 |
| `extra` | 扩展状态，给昂贵优化等特殊策略使用 |

---

## `OptResult` 字段清单

`OptResult` 从 `OptState` 提炼生成，对外暴露稳定接口：

| 字段 | 说明 |
|---|---|
| `bestDecs` | 最优决策变量 |
| `bestObjs` | 最优目标值 |
| `bestCons` | 最优约束值 |
| `bestMetric` | 辅助指标摘要 |
| `bestFeasible` | `best` 是否可行 |
| `appearFEs` | `best` 首次出现时的评估次数 |
| `appearIters` | `best` 首次出现时的迭代次数 |
| `FEs` | 总评估次数 |
| `iters` | 总迭代次数 |
| `runtime` | 总耗时 |
| `history` | `OptHistory` 视图 |
| `extra` | 扩展结果 |

---

## `OptHistory` 字段清单

`OptHistory` 先保持最小结构：

| 字段 | 说明 |
|---|---|
| `populations` | 每轮种群或样本集快照 |
| `bests` | 每轮 best 快照 |
| `metrics` | 每轮指标 |
| `iterToFEs` | 迭代号到评估次数映射 |
| `events` | 可选，昂贵优化特殊事件 |

---

## 已确认删除的旧设计

以下旧字段和做法不再保留：

| 旧设计 | 处理 |
|---|---|
| `bestDecs_True` | 删除 |
| `bestObjs_True` | 删除 |
| `historyDecs_True` | 删除 |
| `historyObjs_True` | 删除 |
| `historyBestDecs_True` | 删除 |
| `historyBestObjs_True` | 删除 |
| 状态里同时保留算法态和显示态两份数据 | 删除 |
| `run()` 返回导出物而不是结果对象 | 删除 |

结论：  
`OptState` 只保存优化运行本身必需的内部事实。

---

## 硬规则

### 数据规则

| 规则 | 内容 |
|---|---|
| 1 | `bestDecs` / `bestObjs` / `bestCons` 永远保持二维数组语义 |
| 2 | 单目标不做标量特判 |
| 3 | `OptState` 不保存 `*_True` 镜像字段 |
| 4 | `extra` 只放扩展能力，不放通用主字段 |

### 依赖规则

| 规则 | 内容 |
|---|---|
| 1 | `Algorithm` 依赖 `OptState`、`Population`、`Problem` |
| 2 | `Population` 不再直接承担 `problem.evaluate()` 协议 |
| 3 | `OptResult` 不依赖 `xarray`、NetCDF、GUI、`Verbose` |
| 4 | serializer 只消费 `OptState` 或 `OptResult`，不进入主循环 |
| 5 | `Verbose` 只能读取状态或结果，不能改写 `run()` 返回值 |

---

## 对 `Population` 的边界判断

当前结论：

> `Population` 是数据对象，不是评估调度对象。

因此后续迁移方向是：

| 现在 | 目标 |
|---|---|
| `Population.evaluate(problem)` | 后续删除 |
| `Population` 内部直接做 `problem` 相关变换 | 后续删除 |
| `Population` 承担评估前后协议 | 改由 `Algorithm` 或独立 evaluator 承担 |

---

## 后续迁移顺序

| 阶段 | 动作 |
|---|---|
| 1 | 引入 `OptState` / `OptResult` / `OptHistory` |
| 2 | 删除 `*_True` 镜像字段 |
| 3 | 让 `run()` 正式返回 `OptResult` |
| 4 | 把导出逻辑从核心结果对象中拆出去 |
| 5 | 收紧 `Population` 边界，移除 `Population.evaluate(problem)` |
| 6 | 明确 `Algorithm.evaluate(pop)` 的统一评估协议 |
| 7 | 最后整理 `Verbose`、serializer、GUI 依赖方向 |

---

## `Algorithm.evaluate(pop)` 约定

### 结论

`Algorithm.evaluate(pop)` 是优化主链里唯一正式的评估编排入口。

它负责三件事：

| 职责 | 说明 |
|---|---|
| 输入转换 | 使用 `problem.apply_var_type(pop.decs)` 得到可评估输入 |
| 真实评估 | 调用 `problem.evaluate(X)` |
| 结果写回 | 把 `Eval.objs` / `Eval.cons` 写回 `Population`，并累计 `FEs` |

### 正式语义

`Population.decs` 保存的是 `algorithm-space`。  
用户实现的 `evaluate/objFunc/conFunc` 接收到的是已经可直接计算的输入。

因此标准流程是：

```python
X = problem.apply_var_type(pop.decs)
eval = problem.evaluate(X)
pop.objs = eval.objs
pop.cons = eval.cons
state.FEs += len(pop)
```

### 它应该做什么

| 步骤 | 内容 |
|---|---|
| 1 | 检查 `pop.decs` 存在 |
| 2 | 用 `problem.apply_var_type()` 进行评估前转换 |
| 3 | 调 `problem.evaluate(X)` 得到 `Eval` |
| 4 | 原地写回 `pop.objs`、`pop.cons` |
| 5 | 更新 `state.FEs` |
| 6 | 返回 `pop` 本身 |

### 它不应该做什么

| 不该做的事 | 原因 |
|---|---|
| 更新 `best/history` | 属于 `updateState()` |
| 打印 verbose | 属于外围层 |
| 判断终止 | 属于 `checkTermination()` |
| 导出 NetCDF / xarray | 属于 serializer |
| 变量空间解释散落在算法各处 | 转换应统一收在 `evaluate()` 里 |

### `FEs` 规则

| 规则 | 内容 |
|---|---|
| 计数对象 | 真实问题评估次数 |
| 一次 `evaluate(pop)` 增量 | `len(pop)` |
| surrogate 预测 | 不计入 `FEs` |
| 重评估 | 计入 `FEs`，因为是真实再次调用 |

### 转换规则

当前正式约定：

| 项 | 结论 |
|---|---|
| `Population.decs` 是不是 unit-space | 不是 |
| 有没有别的编码体系 | 当前没有 |
| 正式评估前转换入口 | `problem.apply_var_type()` |
| `unit_to_space()` 是否进入优化主链 | 否 |

结论：  
当前不要为了抽象继续加新入口，先直接把 `apply_var_type()` 定成正式评估前转换入口。

---

## `Algorithm.updateState(pop)` 约定

### 结论

`Algorithm.updateState(pop)` 负责用当前已评估的 `Population` 更新 `OptState`。  
它是状态写入入口，不负责真实评估，也不负责终止判断。

### 它应该做什么

| 职责 | 说明 |
|---|---|
| 更新当前群体 | `state.currentPop = pop` |
| 更新 `best*` | 根据单目标/多目标规则更新当前最优 |
| 更新可行性信息 | `bestFeasible` |
| 更新辅助指标 | 如多目标 HV |
| 记录首次出现位置 | `appearFEs`、`appearIters` |
| 记录历史 | 写入 `OptHistory` |

### 它不应该做什么

| 不该做的事 | 原因 |
|---|---|
| 调 `problem.evaluate()` | 那是 `evaluate(pop)` 的职责 |
| 判断是否停止 | 那是 `checkTermination()` |
| 打印 verbose | 那是外围层 |
| 导出数据 | 那是 serializer |

### 更新顺序

建议顺序：

| 步骤 | 内容 |
|---|---|
| 1 | 断言 `pop` 已评估 |
| 2 | 更新 `state.currentPop` |
| 3 | 计算当前 best |
| 4 | 更新 `bestDecs` / `bestObjs` / `bestCons` |
| 5 | 更新 `bestMetric` / `bestFeasible` |
| 6 | 必要时更新 `appearFEs` / `appearIters` |
| 7 | 向 `history` 追加当前轮快照 |

### 单目标与多目标统一规则

| 场景 | `best` 语义 |
|---|---|
| 单目标 | 唯一最优解 |
| 多目标 | 当前最优非支配解集 |

因此：

| 字段 | 单目标 | 多目标 |
|---|---|---|
| `bestDecs` | 1 行 | 多行 |
| `bestObjs` | 1 行 | 多行 |
| `bestCons` | 1 行或 `None` | 多行或 `None` |
| `bestMetric` | 通常 `None` | 如 HV |

---

## `Algorithm.checkTermination()` 约定

### 结论

`checkTermination()` 应尽量做成纯判断接口，只回答“要不要继续”。  
不要再把记录、打印、GUI 事件混在里面。

### 它应该检查什么

| 条件 | 说明 |
|---|---|
| `FEs` | 是否达到 `maxFEs` |
| `iters` | 是否达到 `maxIters` |
| `tolerateTimes` | 是否达到 `maxTolerates` |
| 外部停止信号 | 如 GUI stop |

### 它不应该做什么

| 不该做的事 | 原因 |
|---|---|
| 调 `record()` | 副作用太重 |
| 更新 `OptState` | 状态更新属于 `updateState()` |
| 打印日志 | 属于外围层 |
| 导出结果 | 属于外围层 |

### 容忍终止规则

当前建议：

| 项 | 结论 |
|---|---|
| `tolerateTimes` 只用于单目标 | 是 |
| 比较对象 | 当前 `state.bestObjs` 与新一轮 best |
| 多目标是否沿用这一套 | 当前不建议强行共用 |

---

## `Algorithm.run()` 约定

### 结论

`run(problem, seed=None)` 是一次完整优化的主入口。  
它的正式返回值必须是 `OptResult`，不是导出物，不是 NetCDF，不是 xarray。

### 运行骨架

建议主流程：

| 阶段 | 内容 |
|---|---|
| `setup` | 绑定 `problem`、初始化随机种子、状态、参数 |
| `initialize` | 生成初始 `Population` |
| `evaluate` | 完成真实评估 |
| `updateState` | 写入当前运行态 |
| `loop` | 迭代生成新群体并重复评估与状态更新 |
| `buildResult` | 从 `OptState` 生成 `OptResult` |

逻辑骨架：

```python
self.setup(problem, seed)
pop = self.initialize()
pop = self.evaluate(pop)
self.updateState(pop)

while not self.checkTermination():
    pop = self.advance()
    pop = self.evaluate(pop)
    self.updateState(pop)

return self.buildResult()
```

### 正式返回协议

| 返回值 | 结论 |
|---|---|
| `OptResult` | 正式协议 |
| NetCDF/xarray 导出物 | 不是 `run()` 返回值 |
| `OptState` | 不直接暴露为主返回值 |

---

## `Algorithm.buildResult()` 约定

### 结论

`buildResult()` 从当前 `OptState` 提炼出对外稳定的 `OptResult`。  
它不应再附带文件导出逻辑。

### 它应该做什么

| 职责 | 说明 |
|---|---|
| 读取 `state.best*` | 生成结果核心字段 |
| 读取 `state.history` | 透传历史视图 |
| 读取 `state.FEs/iters/runtime` | 生成运行摘要 |
| 读取 `state.extra` | 透传扩展结果 |

### 它不应该做什么

| 不该做的事 | 原因 |
|---|---|
| 调 NetCDF 导出 | 不属于结果构造 |
| 保存文件 | 属于外围层 |
| 打印 verbose | 属于外围层 |

---

## `AlgorithmABC` 工具边界

### 结论

`AlgorithmABC` 不提供死板流程模板，只提供少数几个固定工具。  
具体算法如何组织循环，由子类自己决定。

### 核心工具

当前建议只保留 4 个核心工具：

| 方法 | 用途 |
|---|---|
| `setup(problem, seed)` | 初始化运行上下文 |
| `evaluate(pop)` | 统一评估 `Population` |
| `update(pop)` | 更新状态并触发阶段输出 |
| `finalize()` | 构造结果并做收尾 |

说明：

| 聚合关系 | 结论 |
|---|---|
| `updateState()` + `afterStateUpdate()` | 聚合为 `update(pop)` |
| `buildResult()` + `afterRun()` | 聚合为 `finalize()` |

### 辅助工具

除核心工具外，只保留少量辅助能力：

| 方法 | 用途 |
|---|---|
| `checkTermination()` | 判断是否继续 |
| `initPop(nInit)` | 默认初始化工具 |
| `setParaVal()` | 参数写入 |
| `getParaVal()` | 参数读取 |

### 设计原则

| 原则 | 内容 |
|---|---|
| 1 | 统一关键原语，不统一细流程 |
| 2 | 不强制 `step(pop)` 这类死模板 |
| 3 | 子类 `run()` 自己组织循环 |
| 4 | 评估、状态更新、收尾必须走公共入口 |

### 推荐使用方式

推荐心智模型：

```python
self.setup(problem, seed)

pop = ...
while self.checkTermination(pop):
    ...
    self.evaluate(pop_or_offspring)
    self.update(pop)

return self.finalize()
```

结论：  
`AlgorithmABC` 的职责是提供稳定工具，不是定义唯一流程框架。

---

## 参数管理约定

### 结论

参数系统统一成一个轻量参数容器，不再维护多份并行状态。  
当前不建议继续使用 `Setting(keys, values, dicts)` 这种三份并存结构。

### 参数系统要解决的事情

当前参数管理至少要覆盖：

| 需求 | 是否支持 |
|---|---|
| 存参数值 | 是 |
| 取参数值 | 是 |
| 打印参数 | 是 |
| 保存到结果摘要 | 是 |
| 便于算法子类开发 | 是 |

### 推荐结构

建议统一成轻量 `Params` 容器。

| 字段 | 用途 |
|---|---|
| `data` | 唯一主存储，`dict[str, Any]` |

### 推荐接口

| 方法 | 用途 |
|---|---|
| `set(name, value)` | 写参数 |
| `get(*names)` | 取一个或多个参数 |
| `asDict()` | 导出参数字典 |
| `items()` | 供 verbose/save 使用 |
| `keys()` / `values()` | 只读视图 |

### 设计原则

| 原则 | 内容 |
|---|---|
| 1 | 单一数据源，只保留一个主存储 dict |
| 2 | 展示形式按需生成，不长期维护 `keys/values` 镜像 |
| 3 | 不提前引入重参数系统、schema 或 dataclass |
| 4 | 参数管理服务于算法配置、展示和保存 |

### 参数和其他字段的边界

| 类型 | 放置位置 |
|---|---|
| 运行期状态 | `OptState` |
| 算法超参数 | `Params` |
| 基类运行控制项 | 可继续直接挂 `AlgorithmABC`，如 `maxFEs`, `maxIters`, `verboseFlag` |

### 兼容策略

当前建议：

| 旧接口 | 处理 |
|---|---|
| `setParaVal()` | 暂时保留为薄包装 |
| `getParaVal()` | 暂时保留为薄包装 |
| `Setting.keys/values/dicts` | 后续移除，不再作为正式结构 |

### 最终方向

一句话：

> 参数系统统一成一个轻量 `Params` 容器，单一 dict 为主，不再维护 `keys/values/dicts` 三份状态。

---

## 打印输出约定

### 结论

打印输出先收敛成三层，不再继续做复杂表格体系：

| 层 | 作用 | 形式 |
|---|---|---|
| `progress` | 实时看当前运行进度 | 单行动态刷新 |
| `summary` | 低频记录关键阶段摘要 | 单行摘要 |
| `final` | 运行结束后的结果总结 | 块状摘要 |

主原则：

1. 打印输出只负责“让人看懂优化有没有正常推进”。
2. 不负责详细查看。
3. 不负责结果保存。
4. GUI 当前不考虑。

### `progress`

用途：  
运行过程中实时显示当前状态，不保留历史。

典型内容：

| 单目标 | 多目标 |
|---|---|
| `FEs`, `iters`, `bestObj`, `bestFeasible` | `FEs`, `iters`, `numBest`, `bestMetric`, `bestFeasible` |

示例：

```text
GA | FEs 1240/50000 | Iter 25 | Best 1.23e-04 | Feasible True
NSGAII | FEs 3200/50000 | Iter 32 | ND 58 | Metric 7.81e-01 | Feasible True
```

### `summary`

用途：  
每隔一段时间输出一条可回看的阶段摘要。

特点：

| 项 | 结论 |
|---|---|
| 是否低频输出 | 是 |
| 是否保留在终端历史中 | 是 |
| 是否输出完整历史 | 否 |

示例：

```text
[Summary] GA | Iter 40 | FEs 2000 | Best 7.52e-05 | Feasible True | Improved True
[Summary] NSGAII | Iter 40 | FEs 4000 | ND 63 | Metric 7.88e-01 | Feasible True | Improved True
```

### `final`

用途：  
运行结束后输出最终总结。

典型内容：

| 单目标 | 多目标 |
|---|---|
| runtime, total `FEs`, total `iters`, final `bestObj`, `bestFeasible` | runtime, total `FEs`, total `iters`, final `numBest`, final `bestMetric`, `bestFeasible` |

示例：

```text
Runtime: 12.34 s
Used FEs: 50000
Used Iters: 500
Final ND: 91
Final Metric: 8.73e-01
Feasible: True
```

### 默认不输出 `X`

当前正式结论：

| 场景 | 是否输出 `X` |
|---|---|
| `progress` | 否 |
| `summary` | 否 |
| `final` | 否 |

结论：  
`X` 不进入主输出通道。  
主输出只保留进度和结果摘要；如需查看决策变量，后续通过单独 inspect/detail 接口处理。

### 单目标与多目标默认摘要字段

| 层 | 单目标 | 多目标 |
|---|---|---|
| `progress` | `FEs`, `iters`, `bestObj`, `bestFeasible` | `FEs`, `iters`, `numBest`, `bestMetric`, `bestFeasible` |
| `summary` | `iters`, `FEs`, `bestObj`, `bestFeasible`, `improved` | `iters`, `FEs`, `numBest`, `bestMetric`, `bestFeasible`, `improved` |
| `final` | runtime, total `FEs`, total `iters`, final `bestObj`, `bestFeasible` | runtime, total `FEs`, total `iters`, final `numBest`, final `bestMetric`, `bestFeasible` |

### 与其他模块的边界

| 模块 | 与打印输出的关系 |
|---|---|
| `Algorithm` | 在生命周期节点触发输出 |
| `OptState` | 提供原始状态数据 |
| `OptResult` | 提供最终结果摘要 |
| inspect/detail | 不属于主输出通道 |
| exporter/save | 不属于主输出通道 |

### 轻量 verbose 结构

当前正式结论：

`optimization/verbose.py` 第一版采用轻量结构，不引入重协议层。

| 组件 | 是否保留 | 用途 |
|---|---|---|
| `VerboseConfig` | 是 | 控制输出频率、精度、preview 数量 |
| `ProgressState` | 是 | 统一提供给渲染层的摘要状态 |
| `SingleObjectiveRenderer` | 是 | 单目标输出 |
| `MultiObjectiveRenderer` | 是 | 多目标输出 |
| `VerboseReporter` | 是 | 负责 `progress / summary / final` 调度 |

当前不做：

| 组件 | 当前处理 |
|---|---|
| `BaseRenderer` / `Protocol` | 不引入 |
| history / inspect / plot | 不放进 `verbose.py` |
| 复杂 internal metrics | 暂不引入 |

### 终端 / 日志 / 保存分工

当前正式结论：

| 通道 | 作用 | 内容 |
|---|---|---|
| terminal | 实时查看 | `progress + summary preview + final preview` |
| log | 回看文本记录 | `summary full + final full` |
| save | 结果文件保存 | 独立于 verbose/log |

规则：

| 类型 | terminal | log |
|---|---|---|
| `progress` | 有 | 没有 |
| `summary` | 有（preview） | 有（full） |
| `final` | 有（preview） | 有（full） |

结论：  
`logFlag` 不记录 progress，只记录 `summary` 和 `final` 的完整版本。

### `summary` / `final` 预览规则

单目标：

| 项 | 规则 |
|---|---|
| `best X` 是否输出 | `summary` 和 `final` 都输出 |
| 完整输出阈值 | `nInput <= 8` |
| 超过阈值 | 只 preview 前 8 个，加 `...` |
| 数值格式 | 科学计数法 |

多目标：

| 项 | 规则 |
|---|---|
| `Pareto Preview` 是否输出 | `summary` 和 `final` 都输出 |
| Preview 数量 | `5` |
| 是否输出完整 Pareto | 否 |
| 是否输出对应 `X` | 否 |
| 每个点显示内容 | 目标值向量概况 |

### preview / full 规则

单目标：

| 输出位置 | 内容 |
|---|---|
| terminal `summary/final` | `best X` preview |
| log `summary/final` | `best X` full |

多目标：

| 输出位置 | 内容 |
|---|---|
| terminal `summary/final` | `Pareto preview` |
| log `summary/final` | `Pareto` full |

### 输出层规则补充

| 层 | 单目标 | 多目标 |
|---|---|---|
| `progress` | 不输出 `X` | 不输出 Pareto preview |
| `summary` | 输出 `best X` preview | 输出 Pareto preview（5 个） |
| `final` | 输出 `best X` preview | 输出 Pareto preview（5 个） |

### `progress` 刷新行为

`progress` 是实时刷新行，不是滚动日志。

| 场景 | 行为 |
|---|---|
| TTY | 使用单行覆盖刷新 |
| 输出 `summary/final` 前 | 先换行 |
| 非 TTY | 自动退化为普通逐行输出 |

结论：  
`progress` 只服务“当前状态查看”，不承担历史记录功能。

### key-value 对齐

当前正式结论：

| 输出层 | 冒号对齐 |
|---|---|
| `progress` | 不对齐，保持紧凑 |
| `summary` | 对齐 |
| `final` | 对齐 |

结论：  
`summary/final` 使用对齐的 key-value block，提升可读性。

### `verboseFreq` 语义

当前决定保留 `verboseFreq` 这个名字。

正式语义：

> `verboseFreq` 表示阶段性 verbose 输出频率，主要控制 `summary` 输出；不影响 `progress` 刷新，也不影响 `final` 输出。

对应关系：

| 输出层 | 是否受 `verboseFreq` 控制 |
|---|---|
| `progress` | 否 |
| `summary` | 是 |
| `final` | 否 |

---

## 结果保存约定

### 结论

结果保存第一版先采用单文件 `npz` 方案。  
当前不把 NetCDF 作为主方案，也暂时不切到 `sqlite3`。

| 项 | 结论 |
|---|---|
| 默认保存格式 | `npz` |
| 是否单文件 | 是 |
| 当前主方案是否 NetCDF | 否 |
| 当前是否使用 `sqlite3` | 否，先不做 |

### 保存范围

第一版 `npz` 只保存以下内容：

| 类别 | 建议保存 |
|---|---|
| 最终结果 | `bestDecs`, `bestObjs`, `bestCons` |
| 摘要历史 | 单目标的 `bestObjHistory`；多目标的 `numBestHistory`, `bestMetricHistory` |
| 进度映射 | `iterToFEs` |
| 元信息 | `summaryJson` |

### 默认不保存的内容

| 内容 | 当前处理 |
|---|---|
| 完整 population 历史 | 默认不保存 |
| 单目标每轮完整 `bestDecs` 历史 | 默认不保存 |
| 多目标每轮完整 front 历史 | 默认不保存 |
| GUI 相关信息 | 不保存 |

结论：  
第一版只保存最终结果和摘要历史，不保存完整群体级历史。

### 推荐键名

建议 `npz` 中使用以下键名：

| 键名 | 含义 |
|---|---|
| `bestDecs` | 最终 best 决策变量 |
| `bestObjs` | 最终 best 目标值 |
| `bestCons` | 最终 best 约束值，可缺省 |
| `iterToFEs` | 迭代号到评估次数映射 |
| `bestObjHistory` | 单目标每轮 best 摘要历史 |
| `numBestHistory` | 多目标每轮最优解集大小历史 |
| `bestMetricHistory` | 多目标每轮摘要指标历史 |
| `summaryJson` | JSON 字符串形式的元信息摘要 |

### `summaryJson` 建议内容

`summaryJson` 中建议包含：

| 字段 | 说明 |
|---|---|
| `algorithm` | 算法名 |
| `problem` | 问题名 |
| `nInput` | 输入维度 |
| `nObj` | 目标维度 |
| `nCon` | 约束维度 |
| `FEs` | 总评估次数 |
| `iters` | 总迭代次数 |
| `runtime` | 总耗时 |
| `bestFeasible` | 最终 best 是否可行 |
| `bestMetric` | 多目标摘要指标，可选 |
| `appearFEs` | 首次出现位置 |
| `appearIters` | 首次出现位置 |

### 多目标历史保存策略

当前正式结论：

| 内容 | 默认策略 |
|---|---|
| 多目标最优历史 | 保存摘要历史 |
| 多目标每轮完整 front 历史 | 不默认保存 |

也就是说，多目标历史默认只保存：

| 键名 |
|---|
| `numBestHistory` |
| `bestMetricHistory` |
| `iterToFEs` |

而不是每轮完整 front 集合。

---

## sqlite3 保存方案约定

### 结论

结果保存主方案后续转向 `sqlite3`。  
当前目标不是热启动优先，而是：

| 目标 | 是否纳入 |
|---|---|
| 完整可复现 | 是 |
| 按轮次查询 population / best / Pareto | 是 |
| 热启动 | 暂不优先 |

### 频率控制

当前沿用这两个名字：

| 参数 | 含义 |
|---|---|
| `verboseFreq` | `summary` 输出与记录频率 |
| `saveFreq` | sqlite 快照保存频率 |

说明：

| 事件 | 是否受 `verboseFreq` 控制 | 是否受 `saveFreq` 控制 |
|---|---|---|
| `progress` | 否 | 否 |
| `summary` | 是 | 否 |
| `snapshot` 保存 | 否 | 是 |
| `final` | 否 | 否，结束时始终保存 |

### 主表结构

第一版 sqlite3 采用共用主结构，不为单目标和多目标拆两套 schema。

| 表 | 用途 |
|---|---|
| `run` | 一次运行的元信息 |
| `runParam` | 参数键值 |
| `snapshot` | 某轮保存下来的状态快照 |
| `snapshotMember` | 该快照中的成员明细 |

### `run`

建议字段：

| 字段 |
|---|
| `runId` |
| `algorithm` |
| `problem` |
| `seed` |
| `nInput` |
| `nObj` |
| `nCon` |
| `maxFEs` |
| `maxIters` |
| `verboseFreq` |
| `saveFreq` |
| `status` |
| `finalFEs` |
| `finalIters` |
| `runtime` |
| `createdAt` |
| `finishedAt` |

### `runParam`

建议字段：

| 字段 |
|---|
| `runId` |
| `name` |
| `value` |

### `snapshot`

它表示某一轮保存下来的状态截面。

建议字段：

| 字段 |
|---|
| `snapshotId` |
| `runId` |
| `iter` |
| `fe` |
| `elapsed` |
| `bestObj` |
| `paretoSize` |
| `hypervolume` |
| `constraintViolation` |
| `populationPayload` |
| `bestPayload` |

说明：

| 字段 | 单目标 | 多目标 |
|---|---|---|
| `bestObj` | 使用 | 为空 |
| `paretoSize` | 为空 | 使用 |
| `hypervolume` | 为空 | 使用 |
| `constraintViolation` | 都可用 | 都可用 |
| `populationPayload` | 都可用 | 都可用 |
| `bestPayload` | 都可用 | 都可用 |

结论：

1. 使用 `bestObj`，不用 `bestValue`。
2. 不引入 `summaryPayload`。
3. `elapsed` 保留。
4. `isFinal` 不单独存，可由 `run` 和最后一个 `snapshot` 推导。

### `snapshotMember`

它表示某个快照中的成员明细。

建议字段：

| 字段 |
|---|
| `snapshotId` |
| `idx` |
| `role` |
| `decs` |
| `objs` |
| `cons` |
| `frontNo` |
| `crowdDis` |

`role` 建议值：

| role | 含义 |
|---|---|
| `population` | 普通成员 |
| `best` | 单目标最优点 |
| `pareto` | 多目标 Pareto 成员 |

### payload + detail 双轨

当前正式结论：

| 层 | 用途 |
|---|---|
| `payload` | 快速恢复为 NumPy / Population |
| `detail` | 灵活 SQL 查询 |

也就是说：

| 存储位置 | 内容 |
|---|---|
| `snapshot.populationPayload` | 当前轮 population 的整体载荷 |
| `snapshot.bestPayload` | 当前轮 best / Pareto 的整体载荷 |
| `snapshotMember` | 当前轮成员明细，供 SQL 查询 |

结论：  
sqlite 第一版采用“摘要字段 + payload + detail”三层并存方案。

### 查询目标

这套设计主要支持：

| 查询 | 方式 |
|---|---|
| 某次 run 基本信息 | 查 `run` |
| 某次 run 参数 | 查 `runParam` |
| 某轮快照摘要 | 查 `snapshot` |
| 某轮 population 明细 | 查 `snapshotMember` 中 `role='population'` |
| 某轮 best | 查 `snapshotMember` 中 `role='best'` |
| 某轮 Pareto | 查 `snapshotMember` 中 `role='pareto'` |

---

## `OptReader` 约定

### 结论

sqlite 结果读取层命名为 `OptReader`。  
它负责读取 sqlite 结果，不负责渲染 verbose，不负责 inspect 展示样式。

### 分层

`OptReader` 分两层使用：

| 层 | 用途 |
|---|---|
| 目录级 | 列出结果目录里的 run 文件 |
| 单文件级 | 读取某一个 sqlite run 文件 |

### 目录级接口

建议保留：

| 接口 | 用途 |
|---|---|
| `listRuns(resultDir)` | 列出结果目录里的所有 sqlite run 文件及其摘要 |

说明：  
这里的 `listRuns` 是目录级接口，不是在单个 sqlite 文件内部列 run。

### 单文件级接口

建议接口如下：

| 接口 | 用途 | 返回 |
|---|---|---|
| `getRun()` | 读取当前 sqlite 文件的 run 元信息 | `dict` |
| `getRunParams()` | 读取当前 run 参数 | `dict` |
| `listSnapshots()` | 列出当前 run 保存过的 snapshots | `list[dict]` |
| `loadAlgorithm()` | 读取当前 run 对应的 algorithm 对象 | algorithm object |
| `loadProblem()` | 读取当前 run 对应的 problem 对象 | problem object |
| `loadPopulation(snapshotId)` | 读取某个 snapshot 的 population | `Population` |
| `loadBest(snapshotId)` | 读取某个 snapshot 的 best / pareto | `Population` |
| `loadLastPopulation()` | 读取最后一个 snapshot 的 population | `Population` |
| `loadLastBest()` | 读取最后一个 snapshot 的 best / pareto | `Population` |

### 返回约定

统一约定：

| 接口 | 返回 |
|---|---|
| `loadPopulation()` | `Population` |
| `loadBest()` | `Population` |

说明：

| 场景 | `loadBest()` 返回含义 |
|---|---|
| 单目标 | 只含 1 个成员的 `Population` |
| 多目标 | 含 Pareto 解集的 `Population` |

复现相关接口：

| 接口 | 返回 |
|---|---|
| `loadAlgorithm()` | algorithm 对象 |
| `loadProblem()` | problem 对象 |

说明：

| 规则 | 内容 |
|---|---|
| 1 | algorithm 和 problem 分开读取 |
| 2 | 不强制提供合并后的 `loadRunObjects()` |
| 3 | BLOB 主要服务“直接复现再运行” |

### 设计原则

| 原则 | 内容 |
|---|---|
| 1 | 目录级和单文件级接口分开 |
| 2 | sqlite 只是存储层，对外窗口继续使用 `Population` |
| 3 | `loadBest()` 不返回裸 `decs/objs/cons` 元组 |
| 4 | `OptReader` 不承担终端展示职责 |
| 5 | 结构化表负责查询，BLOB 负责复现 |

一句话：

> `OptReader` 负责读 sqlite 结果；目录级用 `listRuns(resultDir)`，单文件级用 `getRun / getRunParams / listSnapshots / loadAlgorithm / loadProblem / loadPopulation / loadBest / loadLastPopulation / loadLastBest`，其中 population 和 best 都返回 `Population`。

---

## 一句话总纲

`OptState` 负责“怎么跑”，`OptResult` 负责“给什么”，`OptHistory` 负责“跑过什么”；  
外围展示和导出只消费它们，不能再反过来控制优化主流程。
