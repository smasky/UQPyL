# 推断剩余问题修复（2026-10-03）

完成上一轮的三项：概率返回协议、DREAM 纯 snooker 探索退化、AMH 协方差计算成本。使用 conda py312；未提交、推送或重建 wheel。约束引导优化仍按用户要求暂缓。

## 实现与行为

1. `InferenceABC.log_prob` 统一要求每行一个实数，批量接受 `(n,)` / `(n,1)`，单点也接受标量；NaN、正无穷、复数和形状错误明确停止。合法 `-inf` 表示零概率，提议直接拒绝，避免无穷减无穷。初始化从有限 logProb 且满足硬约束的区域收集初值，受既有 `maxInitAttempts` 限制。无法找到足够初值也停止。这些无效输入不以 warning 后伪造结果；失败的已保存运行有 `failed` 状态，session 正常关闭。回调也用于初始化校验，需要确定性；本轮不再沿用上一轮“回调次数逐位相同”的结论。
2. `DREAM_ZS(ps=1)` 在有活动维时默认混入 10% 全维高斯随机游走。新增 `snookerRefreshProb` 范围 `(0,1]`，仅在 `ps=1` 生效，标准差为参数跨度的 0.1 倍。该核对称，log Hastings 修正为 0，越界直接拒绝；固定维不扰动。与原 snooker 核按固定概率混合，给出离开档案仿射子空间的路径。每次运行发出 RuntimeWarning，诊断记录请求比例、刷新比例和实际 snooker 比例，SQLite 往返保留；默认实际 snooker 比例为 90%。`ps<1` 路径不加入新刷新随机数。
3. AMH 使用相对首点的增量中心矩统计计算无偏协方差，只访问新增历史，包含拒绝后重复占据状态。保持原缩放/范围下限；每链额外缓存均值和散布矩阵，reset 清空，WeakKeyDictionary 避免缓存持有已释放链。返回矩阵不与缓存共享。冷缓存允许从已有历史初始化；正式运行历史按追加方式使用，不支持原地修改过去行后自动重建缓存。完整轨迹仍占用随采样数增长的内存。舍入差异可能改变长链随机轨迹，不承诺逐位一致。

## 回归与精度

新增 [39项回归](../tests/test_inference_probability_and_covariance.py)：五方法无效概率和零概率区域、批量形状、初始化预算、两端零概率、提议阶段失败持久化；纯 snooker 原反例全维性/单位换算/固定轴/参数检查；一维及四维、偏置 0/1e12、含重复点的独立 longdouble 中心矩参照；不重扫旧历史与 reset。

已有纯 snooker 运行测试显式检查新增 warning，未通过过滤全部警告来放行。新增专项 **39 passed，1.31秒**；全量 **2978 passed，103.87秒，`-W error`**，预期 warning 在对应测试中捕获。Ruff 与差异空白检查通过。

独立审计 [脚本](verification/check_inference_remaining_fixes.py) / [27条记录](verification/1003-inference-remaining-audit.json) 包含 6 条协方差性能记录和 **21 次完整采样运行**。种子 5/17/41，统计正式链后半段；不是将同一链的每个相关样本当作独立重复实验。

| 案例 | 配置/参照 | 结果 |
|---|---|---|
| AMH 正态 | 4链，warmUp=1000，draws=8000；各维方差约1 | 增量方差 0.980–1.044，旧全历史参照 0.988–1.050；增量均值绝对值最大0.043 |
| AMH 相关截断正态 | 同预算；积分参照方差约0.256 | 新旧方差均0.256–0.263；两者矩差约浮点舍入量级 |
| DREAM 四维均匀 | 3链、archSize=1、ps=1，warmUp=1000，draws=30000；方差1/3 | 三种子全部秩4；方差0.320–0.378，均值绝对值最大0.113。seed41仍混合慢，不能仅凭秩4宣称已收敛 |
| DREAM 正态 | 同小档案/ps=1，draws=15000；方差约1 | 方差0.964–1.015，均值绝对值最大0.022 |
| MH 零概率支持区间 | [-1,1]中只允许[0.4,1]，4链、draws=8000；均值0.7/方差0.03 | 均值0.6981–0.7016，方差0.02996–0.03017 |

6次 DREAM 运行各产生一次预期刷新 warning，其他15次没有 warning。原始的四维、3链、archSize=1、warmUp=20、100步、seed17退化反例由秩2恢复至秩4，已作为短回归固定。

最慢的 seed41 另延长到 **120000步**（不增加档案/刷新比例），后半段方差恢复至 **0.3283–0.3401**，均值绝对值最大 **0.0227**，rank-normalized folded R-hat 为 **1.0011–1.0078**，bulk ESS 为 **529–604**。18万个后半段样本的有效样本数仍较低，说明该小档案配置效率有限；结果支持慢混合解释，不宣称短预算充分。此额外控制使完整采样验证总数为 **22次**。见[原始数据](verification/1003-inference-snooker-long-control.json)，复现参数 `--snooker-control --output agent/verification/1003-inference-snooker-long-control.json`。

## 协方差性能

全量测试和分布审计结束后单独复测，固定四维历史，从第3行起逐前缀更新，每组重复3次取中位数。这里只测协方差更新，不把它写成完整采样器同倍数提速。

| 历史行数 | 全历史重算 | 增量计算 | 比值 |
|---|---:|---:|---:|
| 2000 | 0.0767秒 | 0.0183秒 | 4.2倍 |
| 8000 | 0.8118秒 | 0.0729秒 | 11.1倍 |
| 16000 | 3.4783秒 | 0.1732秒 | 20.1倍 |

最终协方差最大绝对差 **7.55e-15**。旧方式固定维数累计工作约 O(N²)，新方式约 O(N)，单步矩更新为 O(d²)。原完整 AMH 分布审计也记录端到端耗时，但早期部分与 pytest 并发，不据此声明精确的整体加速比。

[独立性能原始数据](verification/1003-inference-covariance-timing.json)，复现：

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run --no-capture-output -n py312 python agent/verification/check_inference_remaining_fixes.py --covariance-only --output agent/verification/1003-inference-covariance-timing.json
```

## 边界与交接

本轮修复的是已确认的三项；有限测试不能保证所有目标和预算下准确。DREAM 小档案即使解除子空间限制仍可能慢混合；DEMC 少链双峰限制保持。中英文 API/用户文档及测试导航已同步。上一轮176组逐值对照是结构重构的历史证据，不代表本轮新增初始化校验、纯 snooker 刷新或增量浮点算法保持旧轨迹。

日志：[新回归](verification/1003-inference-edge-new.txt)、[全量](verification/1003-inference-remaining-full.txt)、[分布审计](verification/1003-inference-remaining-audit.txt)。
