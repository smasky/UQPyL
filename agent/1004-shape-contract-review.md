# 2026-10-04 样本轴与观测轴全链路审查

本轮按用户要求检查形状逻辑，未实施接口迁移，未修改生产代码、文档示例或 pytest。以下问题尚待处理，不应将本报告解读为已修复。此前 3094 项全量回归仍是上一轮结果。

## 检查范围与证据

静态追踪 Space/Problem/ModelProblem、Simulator/SimContext/Evaluator/Eval、校准公共层及 GLUE/SUFI2/ES/IES、结果/SQLite/reader、敏感性 X/Y 入口、替代模型训练/预测入口及中英文 Problem 示例。

- [独立可重跑脚本](verification/check_shape_contracts.py)：53 条输入/输出/拒绝/一致性记录，见[原始 JSON](verification/1004-shape-contracts.json)。捕获错误用于记录当前行为，不是忽略生产错误；其中四方法的 8 组形状对照要求严格相等，不满足则脚本失败。
- 相关既有测试：**238 passed，1.57 秒，-W error**，见[日志](verification/1004-shape-existing-tests.txt)。测试通过与本轮发现不矛盾：部分现状本来就是旧协议，另有尚未覆盖的跨形状/文档缺口。
- 本轮没有重复全量、重建安装包、提交、推送或发布。

## 当前实际协议

| 位置 | 当前行为 |
|---|---|
| Problem.validate/evaluate | `np.atleast_2d(X)`，一维解释为单行样本；再检查列数等于 nInput。单参数多样本必须显式 `(N,1)`。标量只可对应一个单参数样本 |
| GLUE/SUFI2、ES/IES 初始集合 | 使用单行解释；ES/IES 还要求至少两个成员，不能用一维向量暗示单参数集合 |
| 敏感性分析 | 最终检查 X 必须二维；一维 Y 解释为单列多样本输出 |
| 替代模型 fit | 一维 xTrain 解释为 `(N,1)`；一维 yTrain 也是 `(N,1)` |
| 替代模型 predict | 一维 X 解释为 `(1,D)`，与 fit 的一维 X 含义不同 |
| ModelProblem.obs | 必须为二维数值 ndarray；`(T,S)`、`(K,1)` 和 `(1,K)` 都接受，一维拒绝 |
| ModelProblem.mask | 与 obs 完全同形；非布尔数组自动转 bool，NaN 和非零数也会转 True；None 表示不显式屏蔽 |
| ModelProblem.sims | 数值 ndarray，第一维必须等于样本数；没有统一强制尾部形状或观测个数。无 NaN 时，二维/三维及与 obs 尾部不等的形状都可能接受 |
| 校准计算 | obs/mask 用默认 C 顺序 reshape(-1)，sims 用 reshape(N,-1)；评分/更新使用展平结果 |
| ES/IES 的 R | 有效观测数 K×K；已屏蔽观测必须先排除，不能直接传完整观测数的矩阵 |
| CalResult/SQLite | obs/mask 仍是二维；simulations 是二维展平矩阵；nTime/nSeries/seriesLabels 以及数据库 schema 仍依赖二维观测结构 |

ModelProblem 允许模拟形状独立于 obs 是既有设计，`test_model_problem_allows_sim_shape_independent_from_obs` 明确保留这一点。因此不应把这一能力整体称为新发现的算法 bug；若迁移至严格对齐的向量协议，需要明确改变这一设计。

## 确认的差异与缺陷

### SH01：模拟缺失值处理依赖是否展平

同一组 `obs.shape=(2,2)`、mask 屏蔽第二个观测位置：

- sims 为 `(N,2,2)`，被屏蔽位置是 NaN：ModelProblem 与 GLUE 接受，评分 0。
- 将完全相同的数据 reshape 为 `(N,4)`：被拒绝，提示模拟不能有 NaN。
- 去掉 NaN 后，两种布局都接受，且评分相同。

原因是 `_validate_sim` 只有在 `sims.shape[1:] == obs.shape` 时才允许广播 mask。若支持展平模拟，此处必须同步展平匹配；若改为唯一形状协议，应在入口一次性规范并校验，不应由是否存在 NaN 决定可接受布局。

### SH02：替代模型 fit/predict 对同一一维 X 的含义不一致

`LinearRegression.fit([0,1,2], [0,2,4])`（实际传 ndarray）将 X 视为三行一列，成功训练；`predict([0,1,2])` 将 X 视为一行三列，抛出矩阵维度错误。显式预测列向量成功。

这是公共入口约定差异，不是已经证明训练公式错误。全局统一时建议批量 X 必须明确二维；如保留一维便捷入口，只能定义为单样本，不能在不同方法中静默变成样本列。

### SH03：文档掩码示例在降维后使用失效的轴

中英文 `docs_v2/problem.md` / `docs_v2/cn/problem.md` 多处含：

```python
err = simContext.sims - simContext.obs
err = err[:, ~simContext.mask]
np.mean(err**2, axis=(1, 2))
```

布尔二维 mask 索引后 err 已是 `(N,K_valid)`，继续访问 axis=2 触发 AxisError。脚本提取并执行中文原文件里的函数，直接复现；这不是仅凭文本推测。修复时先将误差和 mask 按同一顺序展平，再选择有效列并沿 axis=1 归约，同时处理无有效观测。

### SH04：观测顺序无法仅靠形状判断

obs=`[[1,2],[3,4]]` 默认展平为 `[1,2,3,4]`。将模拟的时间/序列轴转置后，方阵仍保持 `(2,2)`，但展平变成 `[1,3,2,4]`。测试中评分由 0 变为 0.57735，无形状异常。

这不是程序自动可恢复的错误，也不能靠改成一维彻底解决。向量协议应明确“obs[j]、mask[j]、sims[:,j]、R 对应位置顺序一致”；可选观测标签/分组可帮助发现错位。不要擅自推断/转置。

## 数值对照结果

GLUE、SUFI2、ES、IES 各测试无 mask、有 mask 两种情况，共 8 组。保持 C 顺序，对同一模拟进行 `(N,2,2)` 与 `(N,4)` 的显式 reshape：返回 samples、simulations、scores **逐值完全相等**。

这支持向量化协议不会改变算法公式；没有证明任意重排、错误 mask 或错误 R 顺序仍然等价。ES/IES 对 4 个观测屏蔽 1 个的案例中，3×3 R 接受，4×4 R 明确拒绝，行为符合有效观测协议。

SQLite 往返核对也成功：当前保存的是 obs/mask `(2,2)`、simulations `(24,4)`，nTime=2、nSeries=2、nObs=4。数据库迁移不能只改入口校验。

## 建议的统一目标（尚未实施）

| 数据 | 目标形状与语义 |
|---|---|
| 批量 X | `(N,D)`；一维便捷输入统一只表示一个样本，不自动猜测样本轴 |
| obs | `(K,)`，用户按明确顺序合并观测；不自动猜测站点/时间布局 |
| mask | `(K,)` 布尔，True 忽略；None 可在内部规范为全 False |
| sims | `(N,K)`，包含全部观测列，屏蔽不改变原始列编号 |
| objs / cons | 继续 `(N,nObj)` / `(N,nCon)`；单目标也保留列轴 |
| 有效观测 | `obs[~mask]`、`sims[:,~mask]` |
| R | `(K_valid,K_valid)`，与有效观测同顺序 |
| 观测结构信息 | 可选标签/分组元数据；不能把 nObs 假称为真实 nTime，或将每个位置假称为一个站点 |

建议实施顺序：

1. 确定 obs/mask/sims 唯一协议及是否仍支持无 obs 的纯模拟问题；无 obs 可保留二维模拟但不做观测对齐。
2. 更新 ModelProblem、SimContext、模拟验证与 NaN/mask 处理、单样本适配。
3. 同步校准共同入口、回调、四方法、CalResult、SQLite/reader/export。算法计算继续使用既有有效观测空间。
4. 明确替代模型 fit 的一维 X 政策；更新相关调用方和说明，避免破坏单参数训练流程。
5. 更新中英文示例与测试，增加形状拒绝、mask/NaN、顺序与 R、四方法数值不变和持久化往返验证。

本轮结论是支持一维 obs + 二维 sims 的计算方向，但这是跨模块接口改动，不是给 obs 简单 reshape。此前“批量二维、单个向量一维”的讨论需要补上现有 surrogate.fit 的例外；当前代码还没有全面遵循该约定。
