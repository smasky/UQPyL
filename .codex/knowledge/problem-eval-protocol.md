# Problem / Eval 协议草案

## 结论

`problem` 模块当前正式支持两类问题协议：

| 类型 | 用途 | 允许输出 |
| --- | --- | --- |
| `ProblemBase` | 静态优化、推断、敏感性分析 | `objs`、`cons` |
| `ModelProblem` | 仿真/校准 | `sim` |

`Eval` 只作为统一结果容器，不承载配置、状态或推断逻辑。

---

## 1. `Eval` 正式协议

### 1.1 字段

`Eval` 目前正式字段只有：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `objs` | `np.ndarray | None` | 目标函数值 |
| `cons` | `np.ndarray | None` | 约束函数值 |
| `sim` | `np.ndarray | None` | 仿真输出 |

除这三个字段外，其他输出当前都不属于正式协议。

### 1.2 空值规则

未提供的输出块必须为 `None`。

禁止：

- 用空数组代替缺失输出；
- 同时返回与问题类型无关的输出块；
- 返回 `dict`、tuple、裸 `ndarray` 代替 `Eval`。

### 1.3 形状规则

| 字段 | 形状要求 |
| --- | --- |
| `objs` | `(n_samples, n_obj)` |
| `cons` | `(n_samples, n_con)` |
| `sim` | `(n_samples, n_time, n_series)` |

补充约束：

- 所有正式输出都必须是数值型 `np.ndarray`；
- 第一维必须始终表示样本维 `n_samples`；
- 不允许返回带 `NaN` 的 `sim`；
- `objs` 和 `cons` 默认按二维矩阵解释，不接受面向调用方的高维语义。

---

## 2. `evaluate()` 正式协议

### 2.1 签名

正式公共入口统一为：

```python
evaluate(X, target=None) -> Eval
```

### 2.2 输入规则

`X` 必须在进入实际计算前被规整为：

```python
(n_samples, n_input)
```

调用方不应依赖一维输入特判。问题对象内部可以兼容单样本输入，但正式协议对外只认批量二维输入。

### 2.3 返回规则

`evaluate()` 必须返回 `Eval` 实例。

禁止：

- 返回裸 `np.ndarray`；
- 返回 `(objs, cons)` tuple；
- 返回 `dict`；
- 根据调用方类型隐式切换返回格式。

### 2.4 `target` 规则

`target` 的语义是“请求哪个输出块”。

| 问题类型 | 合法 `target` |
| --- | --- |
| `ProblemBase` | `None`、`"objs"`、`"cons"` |
| `ModelProblem` | `None`、`"sim"` |

非法 `target` 必须立即报错，禁止兜底推断。

---

## 3. `ProblemBase` 协议

### 3.1 定位

`ProblemBase` 只表示静态问题：

- 输入一批决策变量；
- 返回目标值；
- 可选返回约束值。

它不承载时序仿真输出协议。

### 3.2 必备元信息

`ProblemBase` 实例必须稳定提供：

| 字段 | 含义 |
| --- | --- |
| `nInput` | 输入维度 |
| `nObj` | 目标维度 |
| `nCon` | 约束维度 |
| `opt` | 目标方向，供内部统一转成最小化 |
| `objLabels` | 目标标签 |
| `conLabels` | 约束标签，可为空 |
| `space` | 输入空间定义 |

兼容字段如 `nOutput`、`nCons` 可以保留，但不应再扩展新语义。

### 3.3 输出约束

| `target` | 允许返回 |
| --- | --- |
| `None` | 必须返回 `objs`；有约束时可返回 `cons` |
| `"objs"` | 只返回 `objs`，`cons=None` |
| `"cons"` | 只返回 `cons`，`objs=None` |

补充规则：

- `ProblemBase` 不得返回 `sim`；
- 若 `nCon == 0`，则 `cons` 必须为 `None`；
- `objs.shape[1]` 必须等于 `nObj`；
- 若返回 `cons`，则 `cons.shape[1]` 必须等于 `nCon`。

---

## 4. `ModelProblem` 协议

### 4.1 定位

`ModelProblem` 表示仿真/校准问题：

- 输入一批参数；
- 输出与观测空间对齐的仿真张量；
- 不通过 `objs/cons` 暴露正式结果。

### 4.2 必备元信息

`ModelProblem` 实例必须稳定提供：

| 字段 | 含义 |
| --- | --- |
| `nInput` | 输入维度 |
| `obs` | 观测矩阵，形状 `(n_time, n_series)` |
| `mask` | 缺测掩码，形状与 `obs` 一致或为 `None` |
| `obsShape` | 即 `obs.shape` |
| `simLabels` | 序列标签 |
| `nObs` | 展平后的观测总长度 |
| `space` | 输入空间定义 |

`ModelProblem` 当前不再把 `nOutput` 作为正式协议字段。

### 4.3 输出约束

| `target` | 允许返回 |
| --- | --- |
| `None` | 返回 `sim` |
| `"sim"` | 返回 `sim` |

补充规则：

- `objs` 必须为 `None`；
- `cons` 必须为 `None`；
- `sim.shape` 必须严格等于 `(n_samples, *obsShape)`；
- `sim` 必须是数值型三维数组；
- `sim` 不允许包含 `NaN`。

---

## 5. 调用方依赖边界

不同模块只允许依赖自己关心的输出块：

| 模块 | 允许依赖 |
| --- | --- |
| `optimization` | `objs`、`cons` |
| `inference` | `objs`、`cons` |
| `analysis` | `objs`、`cons` |
| `calibration` | `sim` |

禁止跨边界偷用：

- `optimization` 直接依赖 `sim`；
- `calibration` 直接依赖 `objs/cons` 作为正式输入协议；
- `analysis` 假设任意问题都可返回 `sim`。

---

## 6. 当前不做的事

当前阶段先不做以下扩展：

- 不新增 `residual`、`likelihood`、`posterior` 等字段到 `Eval`；
- 不把动态问题、同化问题并入 `ProblemBase`；
- 不为兼容旧代码引入自动猜测输出类型的兜底逻辑；
- 不把 `ModelProblem` 强行伪装成单目标优化问题。

---

## 7. 下一步最小落地项

建议按下面顺序收口：

| 顺序 | 动作 |
| --- | --- |
| 1 | 给 `problem` 模块写正式 docstring / docs 版协议页 |
| 2 | 给 `Eval` 增加 shape 校验与构造约束 |
| 3 | 给 `ProblemBase.evaluate()` / `ModelProblem.evaluate()` 补一致性校验 |
| 4 | 给 `analysis`、`optimization`、`inference`、`calibration` 各补一组协议测试 |
| 5 | 再决定是否引入 `DynamicProblem` / `AssimilationProblem` |

一句话总结：

先把“静态问题返回 `objs/cons`、模型问题返回 `sim`”这条边界钉死，再谈后续扩展。
