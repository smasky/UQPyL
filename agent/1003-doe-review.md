# DoE 专项复核（2026-10-03，只检测）

后续状态：DOE01–DOE04已修复，见[修复记录](1003-doe-fixes.md)。下文保留检测时证据。

本轮没有修改生产实现或 pytest；已存在的 DoE 工作区改动不是本轮产生。使用 conda py312。47 项既有 DoE/LHS 测试通过（0.89 秒，`-W error`），独立审计生成 194 条记录，包含正常对照和缺陷复现，不将194条全称为通过。没有重复全仓库测试，2978为上一轮基线；未提交/推送。

## 已确认问题

| 编号 | 问题 | 可复现结果 | 建议 |
|---|---|---|---|
| DOE01 | LHS maximin / center_maximin 单样本退化未处理 | 三维问题、nSamples=1、seed17，pdist为空，对空数组取最小值导致 ValueError；classic/center/correlation正常 | 单点不存在点间最小距离，告警后退化为对应 classic/center 设计即可 |
| DOE02 | 两种 maximin 模式缺少 iterations 预检 | iterations=0或-1时循环不执行，返回未赋值H，引发UnboundLocalError | 对优化模式统一检查正整数；非法配置明确参数错误，不能返回未初始化结果 |
| DOE03 | Sobol / Saltelli 跳点合法性与质量提示不匹配 | skipValue=32、N=16被强制拒绝；SciPy跳过32后可正常生成16点，首列16分层各一点。反之 skipValue=4、N=16被静默接受，首列分层计数为[0,2,1,1]重复4次 | 去掉人为N>=skip限制；以实际块对齐/跳点风险给出质量warning。允许生成不代表保证均衡性 |
| DOE04 | FFD 层数元数据持有调用方列表引用 | levels=[2,3,4]生成24行；随后调用方改levels[0]=9，已返回meta也变为[9,3,4]，暗示108行 | 元数据保存独立副本；采样矩阵本身未被修改 |

DOE01 是合法边界输入失败；DOE02 是无效输入的诊断缺陷；DOE03 是接口限制和采样质量提示缺口；DOE04 是结果元数据隔离错误。本轮没有证据表明默认常规采样全面失准。以上均待修复，建议 DOE01/02 → DOE03 → DOE04。

Sobol 官方说明允许通过 fast_forward 跳点，同时明确提醒跳点可能破坏均衡性质：[SciPy Sobol 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Sobol.html)。本轮是否失去首列分层是直接计数验证，不只是引用文档推断；也不把首列分层通过当作所有高维投影质量的证明。FAST 的频率分配与相位方案对照了 [SALib 官方源码](https://salib.readthedocs.io/en/latest/_modules/SALib/sample/fast_sampler.html)。

## 正常参照和精度边界

- 90组 LHS：维数1/3/8、样本数2/16、种子5/17/41、五种criterion。各维严格一层一点；center类位于层中心，随机复现通过。此处验证输出分层，不证明有限候选搜索得到全局最优空间填充。
- 63组设计结构：Random/Sobol/Saltelli一阶与二阶/FAST/Morris两档层数，维数1/3/8和三个种子。检查范围、有限值、同实例复现、全局NumPy随机状态不变；Saltelli逐块核对A/B坐标替换；Morris每一步恰有一维改变、每维恰好一次、步长与网格一致；FAST用独立三角波表达式核对频率/相位生成结果。
- 7组混合域：全部七方法覆盖连续变量[-3,9]、整数[-2,3]、离散小数[0.1,2.5,8]、固定值7。全部落在合法取值内。固定维正确映射不等于 Morris 分析支持固定维；后者已有正范围限制。
- 4组 SciPy 序列逐值对照：scramble开/关、skip=0/16，N=16。结果完全一致。
- 9次线性公共流程：y=x0+2*x1+3*x2，独立U(0,1)。Sobol S1/ST对照[1,4,9]/14，最大误差小于7e-6；FAST误差小于0.0015；Morris mu_star=[1,2,3]且sigma接近浮点零。
- 6次乘积交互公共流程：y=x0*x1，第三维无关，理论S1=[3/7,3/7,0]、ST=[4/7,4/7,0]。Sobol/FAST两方法、三个种子；最大绝对误差约0.01190，来自有限阶FAST。不是各方法都有机器精度，也不以这两个模型证明全部非线性精度。
- FFD [2,3,4] 笛卡尔积逐行匹配独立 itertools.product 参照。

本轮未进行高维超大样本的性能/内存压力测试，未证明任意离散映射保持连续空间的低差异性质；FFD采样数超过离散选项数时出现重复真实值也不能直接认定为实现bug。

## 复现

[独立脚本](verification/check_doe_review.py)、[194条原始记录](verification/1003-doe-review.json)、[审计日志](verification/1003-doe-review.txt)、[47项既有测试](verification/1003-doe-review-pytest.txt)。脚本Ruff通过。

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run --no-capture-output -n py312 python agent/verification/check_doe_review.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run --no-capture-output -n py312 pytest -q tests/test_doe*.py tests/test_lhs_correlation.py -W error --basetemp=.cache/pytest/doe-review
```
