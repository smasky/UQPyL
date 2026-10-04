# 2026-09-19 再次复核：公开入口联动与状态一致性

后续状态：B01—B06 已修复，见 [B01—B03](0919-b01-b03-fixes.md) 与 [B04—B06](0919-b04-b06-fixes.md)。下文保留审查时的原始复现。

基线：`e4a6aad258aada37a6956e374a7e2dd4d2e7e4be`，已推送 dev，六组 wheel 构建/测试 CI 通过：[run 35411252001](https://github.com/smasky/UQPyL/actions/runs/35411252001)。用户明确暂不发布正式版本。本轮仅审查、复现和记录，没有修改生产代码，也没有重复运行未改代码的全量测试。

本轮集中追踪先前修复未覆盖的组合入口：AutoTuner 与 fit 失效规则、MultiSurrogate 对象所有权、MARS 非 predict 入口。以下六项均在 conda py312 下复现，不是推测。上轮 A03 公共 fit 的失效修复有效，但没有覆盖所有相关公开入口，不能将其理解为所有模型操作都已原子化。

| 编号 | 优先级 | 已确认问题 | 位置 | 实测与修复方向 |
|---|---|---|---|---|
| B01 | P1 | AutoTuner 错误出口会留下新预处理/训练数据配旧 fitState | [auto_tuner.py:228](../UQPyL/surrogate/auto_tuner.py#L228)、[base.py:216](../UQPyL/surrogate/base.py#L216) | 已训练 GPR 后调用 gridTune(paraGrid={})；报错前已修改模型。捕获 ValueError 后 predict 仍返回有限数，原查询预测最大变化 0.97385。应把无副作用校验提前，并为整个调参/最终重拟合过程统一失败失效或完整回滚；不能只覆盖全部候选失败分支。 |
| B02 | P1 | MultiSurrogate 允许同一模型实例重复占两个输出槽 | [base.py:325](../UQPyL/surrogate/base.py#L325)、[base.py:356](../UQPyL/surrogate/base.py#L356) | MultiSurrogate(2,[m,m]) 拟合 sin/cos 两输出后，两列预测完全相同，第一列最大误差 1.38215。容器复制不隔离重复元素；建议构造/append/fit 明确拒绝重复实例（无需更改已确认的独立模型实例引用语义）。 |
| B03 | P1 | MARS.predict_deriv 忽略输入/输出 Scaler，导数不在原量纲 | [mars.py:622](../UQPyL/surrogate/mars/mars.py#L622) | y=3x+2，输入输出均 StandardScaler：导数 API 返回 1，predict 的中心差分为 3。需与 predict 使用同一输入处理，按链式法则还原输出/输入缩放；polyFeature 与自定义 Scaler 无导数接口时须明确限制，不能静默给错导数。 |
| B04 | P2 | MARS 派生接口绕过拟合状态检查 | [mars.py:633](../UQPyL/surrogate/mars/mars.py#L633)、[mars.py:692](../UQPyL/surrogate/mars/mars.py#L692) | 正常拟合后用错行数触发 fit 失败，predict 已拒绝，但 predict_deriv 仍返回旧导数。basis_/coef_ 等旧实例属性仍在，导数及 transform 没有 requireFitted；需统一公开拟合后接口的有效状态。动态已复现导数路径，transform 同类漏洞为代码确认。 |
| B05 | P2 | MARS.score_samples 调用不存在的 predict 参数 | [mars.py:680](../UQPyL/surrogate/mars/mars.py#L680) | 正常拟合后调用即 TypeError：unexpected keyword argument 'missing'。需明确 missing 支持并接通正式预测接口；补实际 score_samples 回归。 |
| B06 | P2 | gridTune 将从未注册的参数名当作暂不活跃参数忽略 | [auto_tuner.py:198](../UQPyL/surrogate/auto_tuner.py#L198)、[base.py:186](../UQPyL/surrogate/base.py#L186) | paraGrid={'misspelled_parameter':[.1,.2]} 正常返回参数 None、有限分数 0.88468，实际没有调该参数。需区分拼写错误与合法的结构切换后不活跃参数；错误参数入口拒绝。 |

## 为什么之前测试没发现

- A03 回归验证公共 fit 失败后 predict，不包含 AutoTuner 在参数校验/最终重拟合中途异常，也不包含 MARS 的导数入口。
- MultiSurrogate RNG 测试使用两个独立 KRG，没有重复模型实例场景。
- MARS 验证主要覆盖拟合、预测与编译兼容性，未覆盖缩放后的导数、score_samples。
- AutoTuner 结构参数测试有合法不活跃参数，但没有区分“从未注册”与“暂时不活跃”。

## 原始证据与复现

- [脚本](verification/review_followup_0919.py)
- [六项结果](verification/0919-followup-review.json)
- 命令：`conda run --no-capture-output -n py312 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python agent/verification/review_followup_0919.py`

优先顺序：B01/B02/B03 的静默结果错误，再统一 B04/B05 的 MARS 入口，最后 B06 参数验证。这里没有宣称整仓再无其他问题，也没有把所有旧问题重新开启。
