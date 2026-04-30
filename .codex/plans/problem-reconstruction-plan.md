# Problem 重构计划

## 结论

当前 `problem` 模块下一阶段不继续扩功能，只做**内部收口**。

目标只有一条：

> 把现在已经成型的 `Space + EvalResult + Problem` 主链，继续收成清晰、单义、可维护的静态问题抽象。

本轮暂时不做：

- 同化
- diagnostics
- 全仓库旧接口扫描
- 外层算法全面适配

---

## 本轮目标

| 目标 | 说明 |
|---|---|
| 收 `ProblemABC` | 让它更像真正的 base / protocol，不再像历史工具堆 |
| 清理命名 | 模块内部统一以 `nObj / nCon / objLabels / conLabels` 为主 |
| 收紧 `Space` | 把输入空间校验、I/F/D 变换、边界语义做扎实 |
| 收构造器 | 把 `Problem` 的构造参数收成明确的主参数和兼容参数 |
| 明确迁移边界 | `problem` 模块内部先切干净，外部依赖后续渐进迁移 |

---

## 分阶段计划

### 阶段 1：收 `ProblemABC`

目标：让 `ProblemABC` 更像真正的基础抽象，而不是“大而全历史类”。

计划：

| 动作 | 说明 |
|---|---|
| 明确 `ProblemABC` 的职责 | 只保留问题元数据、评估协议、最小兼容包装 |
| 继续剥离实现细节 | 能放进 `Space` 的，不留在 `ProblemABC` |
| 评估是否重命名 | 内部先向 `ProblemBase` 语义靠拢，是否改名后置决定 |

产出：

- `ProblemABC` 只承担协议与少量兼容行为
- `Space` 负责输入空间逻辑
- `Problem` 负责静态问题实例化

---

### 阶段 2：收 `Problem` 构造器

目标：把构造器语义收成“新接口优先，旧接口有限兼容”。

计划：

| 动作 | 说明 |
|---|---|
| 明确主参数 | `space / nObj / nCon / objLabels / conLabels / optType` |
| 保留必要兼容参数 | `nOutput / nCons` 暂保留为兼容入口 |
| 逐步削弱旧命名 | 内部全部走新字段，旧字段只作为 alias |

`Problem` 组合模式的硬规则：

| 组合 | 是否允许 | 说明 |
|---|---|---|
| 只传 `objFunc` | 允许 | 简单问题 |
| 传 `objFunc + conFunc` | 允许 | 常规约束优化 |
| 只传 `evaluate` | 允许 | 一次仿真同时得到 `obj/con` |
| 继承 `ProblemBase` | 允许 | 复杂问题、benchmark、后续扩展 |
| 只传 `conFunc` | 不允许 | 当前框架不支持无 objective 问题 |
| 同时传 `evaluate + objFunc` | 不允许 | 主协议冲突 |
| 同时传 `evaluate + conFunc` | 不允许 | 主协议冲突 |
| 同时传 `evaluate + objFunc + conFunc` | 不允许 | 最乱，必须禁止 |
| 三者都不传 | 不允许 | 问题定义不完整 |

结论：

> `Problem` 的 callable 组合只允许三种合法形式：`objFunc`、`objFunc + conFunc`、`evaluate`。

优先级：

1. 内部统一
2. 构造器统一
3. 对外兼容最后处理

---

### 阶段 3：继续收 `Space`

目标：把输入空间定义做成真正稳定的一层。

计划：

| 动作 | 说明 |
|---|---|
| 补足校验 | 维度、边界、离散变量集合、类型一致性 |
| 梳理 I/F/D 语义 | float/int/discrete 的行为边界明确 |
| 统一变换入口 | `validate / transform / _transform_to_I_D / _transform_unit_X` |
| 检查命名 | `idxF / idxI / idxD` 是否保留，还是改成更清晰的内部名 |

当前命名决策：

| 类别 | 方案 |
|---|---|
| 索引名 | 保留 `idxF / idxI / idxD` |
| `_transform_unit_X` | 改成 `unit_to_space` |
| `_transform_to_I_D` | 改成 `apply_var_type` |
| `_transform_int_var` | 改成 `cast_int_vars` |
| `_transform_discrete_var` | 改成 `map_discrete_vars` |
| `encoding="mix"` | 后续删除，改为直接依据 `idxI / idxD` 判断是否需要类型处理 |

重点检查：

| 点 | 当前关注 |
|---|---|
| unit 空间映射 | 已修，后续防回归 |
| discrete 映射 | 边界、缺键、空集情况 |
| integer rounding | 是否需要边界裁剪或后置顺序说明 |
| mixed 编码 | `encoding="mix"` 是否只是历史标志，后续要不要收掉 |

---

### 阶段 4：整理模块导出

目标：让 `UQPyL.problem` 对外暴露的层次更清楚。

计划：

| 动作 | 说明 |
|---|---|
| 明确公开对象 | `Problem / ProblemABC / EvalResult / SpaceBase / Space` |
| 不再暴露历史残留语义 | 避免导出层继续强化旧抽象 |
| 保持 benchmark 单独存在 | benchmark 是实例集合，不和核心协议混叠 |

---

### 阶段 5：补测试

目标：让 `problem` 模块自己的行为先稳住。

计划：

| 动作 | 说明 |
|---|---|
| 扩展 `problem` 定向测试 | 新字段、新构造器、新空间校验、新 target 协议 |
| 针对 I/F/D 加回归测试 | 防止再次出现算法级错误 |
| 不扩到全仓库扫描 | 只测当前触达的 `problem` 主链 |

---

## 执行顺序

| 顺序 | 动作 |
|---|---|
| 1 | 收 `ProblemABC` 职责边界 |
| 2 | 收 `Problem` 构造器和字段主命名 |
| 3 | 继续收 `Space` 的 I/F/D 与校验 |
| 4 | 补 `problem` 自身定向测试 |
| 5 | 最后再看是否需要调整 `__init__` 导出 |

---

## 这轮不做的事

| 不做项 | 原因 |
|---|---|
| 全仓库替换 `res['objs']` / `res['cons']` | 价值低，后续触达式迁移即可 |
| 同化抽象 | 当前还不该进入 `Problem` 主链 |
| 动态问题设计 | 等静态抽象完全稳定后再说 |
| 大范围 benchmark 整理 | 先别扩大范围 |

---

## 验收标准

本轮结束时至少满足：

| 标准 | 含义 |
|---|---|
| `problem` 模块内部主命名统一 | 新字段为主，旧字段只剩必要兼容 |
| `Space` 行为稳定 | I/F/D 与单位空间映射逻辑自洽 |
| `ProblemABC` 职责明显变窄 | 不再像历史杂糅类 |
| `problem` 定向测试稳定通过 | 当前主链可控 |

---

## 下一轮接口判断

本轮完成后，再决定是否进入：

| 下一轮方向 | 前提 |
|---|---|
| 外部依赖渐进迁移 | `problem` 模块自身已经完全稳定 |
| `ProblemABC` 正式改名 | 内部职责已经足够清楚 |
| 动态 / 同化新抽象 | 静态抽象彻底收口 |
