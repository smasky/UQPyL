# C21 按需推断诊断 · 2026-09-19

本轮补齐现代 R-hat、bulk/tail ESS，并与独立参考实现对照。此前停止原因已完成；C21 本轮范围现已完成。采样过程、停止条件和默认诊断开关保持不变。

## 实现

入口继续使用 `InfResult.computeDiagnostics()`，结果写入该返回对象的 `diagnostics["chains"]`，并返回独立副本：

| 字段 | 方法 |
|---|---|
| `split_rhat` | 保留经典 split R-hat |
| `rhat` | rank-normalized split 与 folded split R-hat 的较大值 |
| `ess_bulk` | 秩归一化分链的 ESS |
| `ess_tail` | 5%/95% 分位数指示序列 ESS 的较小值 |

每项都有按变量对齐的 `values` 和 `status`，固定键名使用 snake_case。原 `ess: not_implemented` 占位已由两个实际指标替换。未显式调用时仍为 `not_computed`。

数学定义参考 [Vehtari 等的论文](https://doi.org/10.1214/20-BA1221) 与 [Stan 说明](https://mc-stan.org/docs/reference-manual/analysis.html)。秩变换采用平均并列秩和 Blom 变换 `(rank-3/8)/(S+1/4)`；ESS 使用 FFT 自协方差、Geyer 初始正序列/单调序列及有限样本修正，与 [ArviZ 0.22.0](https://github.com/arviz-devs/arviz/blob/v0.22.0/arviz/stats/diagnostics.py) 数值对照。

计算逐变量进行，不构造样本数平方级矩阵；FFT 和排序主要随样本数按 N log N 增长。不会在每轮采样中计算，不增加真实模型调用或消耗随机流，不推导全局“已收敛”布尔值，也不自动回写数据库。

## 边界与参考差异

- R-hat 至少两条链，ESS 至少一条链；每条至少四个正式样本。这仅是计算门槛，不是充分采样标准。
- 分链丢弃奇数长度的中间样本；tail 分位数阈值按分链前全部正式样本计算，与固定参考版本一致。
- 负相关允许 ESS 大于实际样本数，不人为截断到 N；沿参考估计约定限制自相关时间的下限。
- 无效数值、任一原始链恒定、无可用链/样本，返回 None 与具体状态，不使用接受率替代诊断。
- 分链整体无变化或折叠序列无法区分时，现代 R-hat 分别标记 `constant_split` / `constant_folded`。任一尾部指示序列完全恒定时，tail ESS 标记 `constant_tail`。参考库某些退化情形会返回样本数或保留仅主体 R-hat，本实现采取明确不可用策略，不声称这些边界数值完全相同。
- 对折叠、分位数和经典方差做二进制缩放，保护极大量纲；现代秩归一化读取原值以保留并列关系。输入本身已经舍入丢失的信息不能恢复。
- 十进制缩放不一定逐位保持折叠后的中央并列秩。专项中 `1e-250` 缩放使 R-hat 从约 1.000867078 变为 1.000863980，两次均与参考库完全一致，见 [舍入证据](verification/0919-c21-fold-roundoff.json)。因此严格缩放不变测试使用精确的二进制倍数，没有把不成立的逐位假设当作算法缺陷，也没有放宽参考对照容差。

## 验证与复现

- 新增 [53 项正式回归](../tests/test_inference_diagnostics.py)，覆盖独立链、强正/负相关 AR 链、厚尾、不同位置/尺度、趋势、离散 ties、极端二进制缩放、短链、常数/冻结链、非有限/非数值、逐变量对齐、FFT 调用和结果隔离。
- 参考固定样本 32 组、126 个指标值；最大绝对差 `1.82e-11`，统一对照容差 `rtol=2e-10, atol=2e-10`。另一个探索性对照覆盖 150 组链数/长度/seed 配置，387 个可用现代指标无不一致；常数尾部及单链 R-hat 按上述状态处理。
- 参考数据为本项目生成的数值样本，保存在 [NPZ](../tests/data/inference_diagnostics_arviz_022.npz) 与 [期望 JSON](../tests/data/inference_diagnostics_arviz_022.json)。正式测试直接使用固定数据，不安装/导入 ArviZ，不依赖联网，不使用 importorskip 跳过对照。
- [生成和验证脚本](verification/verify_c21_diagnostics.py)、[参考结果](verification/0919-c21-reference.json)。本机通过 py312 将 ArviZ 0.22.0、xarray 2025.1.2、xarray-einstats 0.8.0 以 `--no-deps --target /tmp/uqpyl-c21-reference` 隔离安装，未替换 py312 的 NumPy/SciPy。
- 实际 MH 与 SQLite 读回验证诊断前后样本、评价次数、RNG、停止原因不变；返回报告、运行态和数据库的诊断状态互不污染。
- **全量 1906 passed，零警告，29.36 秒**，见 [测试日志](verification/0919-c21-diagnostics-pytest.txt)。Ruff 检查及 `git diff --check` 通过，中英文 API 已同步。

复现参考夹具（需已准备隔离参考库）：

```bash
conda run -n py312 python agent/verification/verify_c21_diagnostics.py --reference-path /tmp/uqpyl-c21-reference
conda run -n py312 pytest -q -W error tests/test_inference_diagnostics.py
```

没有新增运行时依赖，没有提交、推送或发布，也没有将本地验证表述为远程 CI 已通过。后续仍保留 C20 精确 checkpoint；C17 约束选点与 C15 一般 R 优化按既有决定暂缓。
