# optimization 约定

## Population 接口优先

在优化算法实现里：

1. 能通过 `Population` 表达的语义，优先调用 `Population` 的接口。
2. 只有在需要底层批量比较或排序时，才直接调用约束/排序核心函数。

具体规则：

| 场景 | 优先使用 |
|---|---|
| 从一个 `Population` 中取最优解/最优解集 | `getBest()` |
| 从一个 `Population` 中取 Pareto 前沿 | `getParetoFront()` |
| 比较两个等长解集谁更优 | `betterMask()` |
| 对原始数组做单目标约束排序 | `argsortSolutions()` |

一句话：

> 优先使用 `Population` 作为对外窗口，底层比较函数只做支撑，不直接暴露成主调用习惯。

## 约束处理统一规则

优化模块统一采用 Deb feasibility rules。

### 约束违反度

约束满足条件：

```python
cons <= 0
```

聚合约束违反度定义：

```python
CV = sum(max(0, con_i))
```

如果有 `conWgt`，则先加权后再聚合：

```python
CV = sum(max(0, conWgt_i * con_i))
```

### 单点比较规则

比较两个解时：

1. 一个可行、一个不可行：可行解更好
2. 都可行：按目标值比较
3. 都不可行：按 `CV` 比较，`CV` 更小者更好

### 单目标

单目标排序、最优解更新、个体替换统一复用这套规则。

| 场景 | 优先使用 |
|---|---|
| 对一个 `Population` 排序 | `argsort()` |
| 对一个 `Population` 取 best | `getBest()` |
| 两组等长解逐点比较 | `betterMask()` |
| 两个单点解比较 | `compareSolutions()` |

### 多目标

多目标统一规则：

1. 先按可行/不可行分层
2. 可行集上做正常非支配排序
3. 不可行集按 `CV` 排到后面

补充：

| 场景 | 规则 |
|---|---|
| 多目标 `getBest()` / Pareto 提取 | 可行优先，无可行解时按 `CV` |
| `MOEAD` 替换 | 先按 feasibility / `CV`，再按 aggregation |
| 多目标 mating (`tourSelect`) | 暂时继续依赖 `frontNo + crowdDis`，约束通过 `frontNo` 间接生效 |

一句话：

> 约束规则统一，算法不再各自发明约束比较方式。
