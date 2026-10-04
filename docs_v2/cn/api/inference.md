# Inference API

自定义 `logProbFunc` 每行必须返回一个实数：批量形状为 `(n,)` 或 `(n,1)`，单点也可返回标量。允许 `-inf` 表示零概率；NaN、正无穷、复数或错误形状会停止运行，避免将无效链作为成功结果返回。初始化只保留约束可行且 logProb 有限的状态，最多尝试 `maxInitAttempts` 批 LHS；找不到足够初值会停止。零概率提议直接拒绝，不计算 `-inf - -inf`。回调也会用于初始化校验，同一状态可能调用多次，应返回确定的对数概率。

`DREAM_ZS(ps=1)` 在存在非固定维时默认混入 10% 的全维对称高斯随机游走，防止小档案使纯 snooker 永远困在低维空间。`snookerRefreshProb` 可设为 `(0,1]`，只在 `ps=1` 时生效；刷新标准差为各维跨度的 0.1 倍，越界直接拒绝。运行时发出 `RuntimeWarning` 并在 `diagnostics['sampler']['proposal_settings']` 保存 `full_support_refresh_probability`、`effective_snooker_probability` 和 `refresh_scale`。因此默认设置的有效 snooker 比例为 90%；常规 `ps<1` 路径不启用此刷新。

AMH 使用增量中心矩计算历史无偏协方差，保留拒绝后的重复状态、原有缩放和协方差下限。每步仅处理新追加行，额外缓存内存为每链一个均值和散布矩阵，重置时清空；完整采样历史仍保留。结果与全量重算可能有浮点舍入差异，不保证长链逐位一致。

MH/AMH/MH_Gibbs 的初始提议尺度按参数范围设置：高斯标准差为 `gamma * (ub-lb)`，均匀提议的半宽为同一值，MH/MH_Gibbs 使用反射边界，AMH 拒绝原始越界提议。均匀分布对应的方差为半宽平方除以 3，不与高斯方差强制相等。AMH 自适应提议从协方差对角值开平方得到均匀半宽；高斯提议保留完整协方差。自适应协方差下限为 `1e-3 * diag((ub-lb)^2)`，按输入单位换算，固定参数不加入额外下限；历史不足且已有协方差时继续复用其副本。

运行摘要仅处理各条链新增且共同完成的采样点；解码结果写入预分配缓冲区，接受率、可行率、平均 logProb、最优样本及参数均值/标准差增量更新。并列最优沿用“先链序、再采样序”的选择。SQLite 中间快照只读取每条链的末点；显式请求结果或运行结束时才复制完整数组与独立历史。运行期间 Chain 历史按追加方式使用；完整轨迹内存仍随链数×采样数×维度增长，增量浮点统计与全量重算可能存在舍入误差。

InfResult 的 history、settings、diagnostics、extra 与运行态及其他返回结果独立；复用或重置算法不会改写既有结果。公开 objs、bestObjs 及 SQLite 样本目标统一使用 Problem 的原始目标方向，最大化目标不再导出内部负号；logProb 保留实际采样使用的对数概率。内部 Chain/InfState 和自定义 logProbFunc 接收的目标仍使用最小化方向；默认 logProb=-原始目标×problem.opt。此处是开发期返回语义修正，旧数据库不自动迁移。

DEMC 默认 nChains=3；构造时要求整数且至少为 3，拒绝布尔值。 各 Reader 的 `list_runs()` 使用 `run_id`、`created_at`、`finished_at`、`final_fes`/`final_iters`（适用时）、`db_path`、`file_name`；数据库列名及内部对象字段保持原协议。

## `UQPyL.inference`

`inference` 模块对标量 `Problem` 运行 MCMC 风格采样。

## 导入

```python
from UQPyL.inference import MH, AMH, MH_Gibbs, DEMC, DREAM_ZS, InfReader
```

## 公共对象

| 对象 | 作用 |
|---|---|
| `MH` | Metropolis-Hastings。 |
| `AMH` | Adaptive Metropolis-Hastings。 |
| `MH_Gibbs` | 坐标式 Gibbs/MH 更新。 |
| `DEMC` | Differential Evolution MCMC。 |
| `DREAM_ZS` | DREAM(ZS) 风格采样器。 |
| `Chain` | 单条链的运行状态。 |
| `InfResult` | 推断结果对象。 |
| `InfHistory` | 推断历史。 |
| `InfReader` | 读取保存的推断 sqlite。 |

## 通用调用

```text
result = method.run(problem, gamma=0.2, seed=123)
```

| 参数 | 含义 |
|---|---|
| `problem` | 标量输出 `Problem`。 |
| `gamma` | proposal scale，可以是标量、向量或 `(nChains, nInput)` 矩阵。 |
| `seed` | 随机种子。 |

构造函数常用参数：

| 参数 | 含义 |
|---|---|
| `nChains` | 链数。 |
| `warmUp` | warm-up 长度。 |
| `maxIters` | 最大迭代次数。 |
| `logProbFunc` | 自定义对数概率函数。 |
| `verboseFlag`, `logFlag`, `saveFlag` | 运行输出和保存控制。 |

## 方法选择

| 方法 | 适合场景 |
|---|---|
| `MH` | 简单标量问题的默认起点。 |
| `AMH` | proposal scale 难手动固定时。 |
| `MH_Gibbs` | 每次更新一个变量更稳定时。 |
| `DEMC` | 多链共享差分 proposal。 |
| `DREAM_ZS` | 更复杂后验形状和 archive-based proposal。 |

`DREAM_ZS` 的提议 archive 使用内部连续坐标，预热时通过 reservoir sampling 保存占据状态（包含拒绝后的重复点），容量为 `nChains * archSize`；正式采样固定档案、交叉概率和步长尺度。提议只使用档案供体，避免同时更新当前种群引入依赖。`adpInterval` 和 `acTarget` 只控制预热；`warmUp=0` 时使用初始档案和未适应配置。

AMH/DEMC/DREAM 对越界提议直接拒绝，记录 `accepted=False`，不调用用户模型，`FEs` 只计实际评价。DEMC 按链顺序条件更新；默认尺度按非固定维数计算，并以 10% 概率使用单位尺度跳跃，显式 `gamma` 不切换该尺度。扰动是各维独立的零均值高斯噪声，标准差为参数跨度的 `1e-6`。DREAM 的 DE 尺度随选中维数和差分对数调整；snooker 在归一化非固定维上投影，使用独立标量 `U[1.2,2.2]` 步长，与传入的 DE `gamma` 分开，Hastings 修正在对数域计算。
更新当前链不会改写已经存入的历史点。

## `InfResult`

| 字段 | 含义 |
|---|---|
| `decs` | 参数样本，shape 为 `(n_chains, draws, n_input)`。 |
| `objs` | 目标值，shape 为 `(n_chains, draws, n_output)`。 |
| `cons` | 约束值，无约束时为 `None`。 |
| `logProb` | 对数概率，shape 为 `(n_chains, draws)`。 |
| `accepted` | accepted proposal 标记。 |
| `feasibleMask` | 可行性标记。 |
| `acceptanceRate` | 每条链的接受率。 |
| `bestDecs`, `bestObjs`, `bestCons` | 最佳样本及其输出。 |
| `FEs`, `iters` | 函数评估次数和迭代数。 |
| `history` | 运行历史。 |
| `summary()` | 摘要字典。 |

## `InfReader`

| 方法 | 含义 |
|---|---|
| `list_runs(result_dir)` | 列出保存的推断结果。 |
| `get_run_summary()` | 读取摘要。 |
| `get_run_params()` | 读取参数。 |
| `list_snapshots()` | 列出快照。 |
| `load_last_snapshot_members()` | 读取最后快照的链成员。 |
| `load_result()` | 重建 `InfResult`。 |

## 下一步

| 目标 | 阅读 |
|---|---|
| 用户指南 | [Inference](../inference.md) |
| 标量问题定义 | [Problem API](problem.md) |
| 校准工作流 | [Calibration API](calibration.md) |


持久化结果带有模块标识；reader 会拒绝其他模块及没有标识的旧数据库。每次运行都有基于 UUID 的独立 ID，数据库和日志共用，即使关闭 SQLite 保存也有 ID。所有 reader 支持 `with` 和重复 `close()`。内部运行对象统一用 `state`、`params`，返回结果的正式字段不变。

## 统一配置、部分结果与按需诊断

五种采样器均使用 `maxIters`，包括 AMH、DEMC；旧 `maxIterTimes` 名称已移除。
返回结果 `stopReason` 和摘要/reader 的 `stop_reason` 记录 `max_iters`，通用最终出口兜底为 `completed`；异常保持 failed/interrupted 状态。
`InfResult` 属性仍为驼峰，`toDict()` 的固定导出键统一为 `log_prob`、`feasible_mask`、`acceptance_rate`、`best_decs`、`best_objs`、`best_cons` 等 snake_case。

`InfReader.load_partial_result()` 无需最终 artifact，可读取失败、运行中或已完成数据库中的已保存链端点，返回 run 摘要、`snapshots`、`last_saved_iter`、`last_saved_fes`。
固定标记 `complete=False`、`resumable=False`、`sample_scope="saved_chain_endpoints"`；无快照时返回空列表及空的最后保存位置。
每个快照包含该时点每条链的最后一点，参数为真实坐标，目标为原始方向。间隔期间的完整样本没有保存，不能据此计算完整链诊断或精确续跑；正常 `load_result()` 仍要求完整 artifact。

```python
partial = reader.load_partial_result()
report = result.computeDiagnostics()
```

采样期间 `diagnostics["chains"]` 为 `not_computed`。显式 `computeDiagnostics()` 才扫描正式样本并写入返回结果的诊断副本，不增加真实模型调用、不消费随机流、不改变采样停止条件，也不自动回写数据库。
现在提供以下逐变量指标，每项包含 `values` 和 `status`：

| 字段 | 含义 |
|---|---|
| `split_rhat` | 保留经典 split R-hat，便于对照。 |
| `rhat` | 秩归一化 split R-hat 与折叠 split R-hat 的较大值。 |
| `ess_bulk` | 秩归一化分链的主体有效样本量。 |
| `ess_tail` | 5% 与 95% 分位数指示序列 ESS 的较小值。 |

计算口径参照 [Stan 诊断说明](https://mc-stan.org/docs/reference-manual/analysis.html)，并与 [ArviZ 0.22.0](https://github.com/arviz-devs/arviz/blob/v0.22.0/arviz/stats/diagnostics.py) 对照；ArviZ 不属于运行时依赖。秩转换使用平均并列秩和 `(rank-3/8)/(S+1/4)`；经典指标仍依赖有限边际方差。

至少四个正式样本是计算门槛，不表示采样已经充分。R-hat 需要至少两条链，ESS 可以用于一条链；负相关下 ESS 可以超过实际样本数。奇数长度分链时丢弃中间一个样本；tail ESS 的分位数阈值仍用分链前的全部正式样本计算，因此中间样本可能影响该阈值。

样本不足、非数值、非有限数据、任一原始链恒定时，相关指标返回 `None` 及明确状态。经典指标的恒定半链、现代 R-hat 无法区分的分链/折叠序列，以及任一尾部指示序列完全恒定，也分别标记 `constant_chain`、`constant_split` / `constant_folded`、`constant_tail`。这比参考库在某些退化情况下给出完整样本量或忽略无效折叠统计更保守。

```python
report = result.computeDiagnostics()
rhat = report["rhat"]["values"]
essBulk = report["ess_bulk"]["values"]
essTail = report["ess_tail"]["values"]
# 使用数值前检查对应 status；None 表示无法可靠计算该指标。
```

自相关通过 FFT 按需计算。诊断不自动回写 SQLite，不改变 RNG、真实评价次数、采样结果或停止条件。没有全局“已收敛”布尔结论；这些参数诊断不能单独证明所有后验特征已收敛。普通十进制缩放可能通过浮点舍入改变折叠后的并列秩，因此不承诺逐位不变。

## 公共参数与采样策略

五种方法统一检查公共参数：`nChains` 为正整数（DEMC/DREAM 至少3），`warmUp` 为非负整数，`maxIters`、`maxInitAttempts`、`verboseFreq`、`saveFreq` 为正整数；计数参数不接受布尔值。`logProbFunc` 为 callable 或 None。每次运行都会重新检查，包含通过 `set()` 修改的配置。

显式 `gamma` 统一支持实数标量、长度为 `nInput` 的列表/数组、`(1,nInput)` 或 `(nChains,nInput)` 矩阵，元素须有限且非负；零值保留。MH_Gibbs 同样支持列表和整数标量。非法 gamma 在调用用户模型前拒绝，DEMC/DREAM 的 `gamma=None` 继续表示默认规则。公共参数无效时抛出 ValueError，不静默改成其他值。

所有 `InfResult` 都提供 `diagnostics["sampler"]`，公共字段如下：

| 字段 | 含义 |
|---|---|
| `boundary_policy` | `reflect` 或 `reject`。 |
| `update_mode` | 独立整向量、逐坐标、逐链条件更新或给定档案的独立更新。 |
| `adaptation_phase` | `none`、`formal_sampling` 或 `warmup_only`。 |
| `proposal_family` | 随机游走、差分进化或差分进化加 snooker。 |
| `proposal_settings` | 至少含 `gamma`、`distribution`；显式/已解析 gamma 展开为链×参数矩阵。DEMC 自动 gamma 保留 None；不适用的 distribution 为 None。 |

DREAM 的档案策略、大小、冻结尺度和交叉概率继续保留；DEMC 在 proposal_settings 中记录单位跳跃概率与噪声尺度。这些字段描述算法配置，不是收敛证明，随结果一起保存至 SQLite，导出副本相互独立。
