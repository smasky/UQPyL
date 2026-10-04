# 2026-10-02 优化模块修复与扩展验收

## 结论和范围

上一轮确认的 OPT01–OPT05 已修复；延伸检查确认的复形参数/重心/收缩、ABC 重复计数、种群索引/类型、奇数交叉、DE 交叉、单目标历史最优保留和多目标数值边界也已处理。本轮没有只屏蔽 warning、删除失败断言或给坏数值换一个任意有限数。

本轮范围内尚无未处理的、已独立复现的优化逻辑缺陷。不将有限测试表述为数学上的“所有输入绝无 bug”，也不保证随机搜索必达全局最优。已有 C17 代理辅助优化的约束引导选点、C20 精确断点续跑继续按此前决定暂缓；它们是明确的能力边界。

## 实际修复

1. **SCE_UA / ML_SCE_UA**：每次子种群替换后按同一约束优先规则重排，后续选点、最差行与 ML 最优成员不再使用过期行号。重心排除最差成员，收缩向重心而非反射点移动。`npg/nps/nspl` 从原来的“接收后被覆盖”改为实际生效；默认 None 分别解析为 `2*nInput+1`、`nInput+1`、`npg`，不再把子种群数误当输入维数。
2. **全评价最优记录**：单目标评价阶段保留一个待提交最佳候选，即使最终存活集合丢弃它也不会丢失结果；迭代历史和停滞仍在完整一轮提交时更新。保留发现批次的 FEs、完成轮次的 iters，重跑时清空待提交状态；预评估成员可在 FEs=0 成为最优，填充随机成员不会夺走等分先到成员。预评估整型目标在方向转换前转浮点，避免有符号整数最小值取负溢出及无符号负乘错误。
3. **ABC**：成功雇佣蜂清零失败次数；跟随蜂多次失败命中同一来源用累加而非丢失重复索引的赋值；同批成功的来源保留重置语义。全雇佣蜂时跳过空跟随阶段；小种群中正比例舍入为零时 warning 后使用一个雇佣蜂。非法比例和小于两个成员仍停止。
4. **几何数值稳定性**：拥挤距离按列安全缩放；RVEA 使用共同尺度平移目标和安全范数，保留目标的相对权重，参考向量适配能处理常数列/单成员以及极端尺度；MOEAD 四种分解策略与 NSGAIII 关联同样避免中间数值范围故障。MOASMO 高级选点的距离也作共同尺度处理。
5. **GD / IGD**：分块计算最近距离的尾数/二进制指数，避免直接平方，避免远点把极小的最近距离抹掉；在平均之后恢复物理单位，单个距离不可表示但平均仍可表示的情况也正确。公式仍是最近欧氏距离的算术平均。
6. **HV**：极端轴尺度下在安全尺度计算体积后恢复，避免前两维乘积先溢出、第三维小宽度本应抵消的错误；普通计算和 Monte Carlo 随机流保留。真实最终溢出/下溢 warning 后保留 inf/0。自动参考点扩展不可表示时 warning 并限制扩展到有限范围，不悄悄改目标数据。
7. **算子与种群**：`Population[-1]` 正确返回最后一行；整型容器替换浮点成员时提升 dtype，不截断目标/决策/约束。GA 奇数和单成员输入保持后代数量，固定坐标跳过变异除法。DE 二项交叉强制至少一个供体坐标（供体相同时仍可能无位移）。`NDSort(nSort=0)` 明确返回未排序前沿与 0。
8. **入口与调度**：非法核心参数提前拒绝；NaN、复数、缺失目标/必要约束和多目标非有限数据不能进入结果。保留合法单目标最差方向 infinity 排除惩罚，DeltaTest 空子集约定保持。RVEA 支持只设迭代预算的进度调度，拒绝同时取消评价/迭代预算。

## 依据与验收方式

复形排序、排除最差点的重心和向重心收缩，与 [SPOTPY 维护者的 SCE-UA 实现](https://github.com/thouska/spotpy/blob/master/src/spotpy/algorithms/sceua.py) 对照；DE 至少一个供体坐标与 [SciPy 的二项交叉实现](https://github.com/scipy/scipy/blob/main/scipy/optimize/_differentialevolution.py) 对照。未引入这些包为生产依赖，也未将当前 DE 的供体选择策略替换为 SciPy 的策略。

- 新增 `tests/test_optimization_logic_and_range.py`，**85 项回归**：复形逐步排序及所有评价点的最优值、手算反射/收缩、显式配置、停滞/FEs/重跑隔离、ABC 计数、种群 dtype/索引、奇数交叉/固定维、DE 零交叉率、独立 hypot/Decimal 距离与体积参照、各多目标算法跨共同量纲的完整轨迹、非法结果与合法 scalar penalty。
- 新增回归与 DeltaTest 联动：**99 passed，2.37 秒，-W error**。[日志](verification/1002-optimization-integration-final.txt)。
- 修复中期已有优化专项：**492 passed，21.51 秒**，这是当时 64 项新增后的结果，不当作最终全仓计数。[日志](verification/1002-optimization-fixes-targeted.txt)。
- 上轮独立 166 条反例/控制全部复跑：**0 条异常**，原来为 19 条异常。[修复后记录](verification/1002-optimization-review-after.json)、[修复前记录](verification/1002-optimization-review.json)。二者都保留。
- 额外 **231 条独立运行**：168 次公共运行覆盖 14 算法×3种子×连续/有约束/混合变量/全不可行四种场景，并穿插最大化与混合目标方向；重复使用同一算法对象。全部检验真实评价次数、合法变量、参数/目标对应、约束优先、单目标全评价最优及完整多目标非支配档案。另 63 次固定预算精度控制。**全部不变量通过，未产生 warning**。[脚本](verification/check_optimization_extended.py)、[记录](verification/1002-optimization-extended.json)、[摘要](verification/1002-optimization-extended.txt)。
- Ruff 与 `git diff --check` 通过。中英文使用/API 文档和测试导航同步。

最终全仓：conda py312，`python -m pytest -q -W error --basetemp=.cache/pytest/optimization-full-final`，**2880 passed，87.83 秒，无未捕获告警**。[完整日志](verification/1002-optimization-fixes-full.txt)。

## 精度结果的实际含义

二维 Sphere / Rosenbrock / Rastrigin，边界 [-2,2]，三个种子、评价停止阈值 1500、最多 100 轮。以下是各算法最终目标中位数，三个函数理论最优值均为 0；预算沿用完整迭代策略，实际 FEs 可越过阈值。因此本表不是严格等次数排名，也不是通用算法推荐。

| 算法 | Sphere | Rosenbrock | Rastrigin |
|---|---:|---:|---:|
| GA | 2.20e-7 | 1.54e-3 | 2.96e-5 |
| DE | 1.34e-7 | 1.48e-4 | 8.54e-3 |
| PSO | 5.19e-14 | 4.38e-2 | 8.25e-8 |
| ABC | 9.91e-15 | 1.88e-3 | 6.78e-9 |
| CSA | 0 | 6.87e-2 | 0 |
| SCE_UA | 0 | 0 | 7.11e-15 |
| ML_SCE_UA | 0 | 0 | 0.995 |

ML_SCE_UA 的该多峰例仍会停在局部极小，PSO/CSA 的 Rosenbrock 精度也没有达到机器精度。这些运行都正确返回各自评价过的最好点；不能把预算内未找到全局最优解释成逻辑错误，也不能声称精度限制已被消除。

## 中间失败的追溯

- 首轮专项 427 passed / 1 failed：RVEA 的安全参考向量一度统一缩放，角度正确但改变既有直接返回尺度；已调整为可表示时保持原物理向量，极端范围才用方向等价表示，原测试保留并通过。
- 首轮全仓 2854 passed / 5 failed：过严的有限数入口校验拒绝了 DeltaTest 故意用于空子集的 `+inf`。已恢复合法单目标排除协议，未修改 DeltaTest 或删除其断言。[日志](verification/1002-optimization-fixes-full-first.txt)。
- 下一轮全仓 2877 passed / 1 failed：敏感性 SQLite 测试发生 disk I/O error。该轮与另一次 pytest 共用默认 `--basetemp=.cache/pytest/tmp`，临时目录被并行测试清理；最终全仓改用独立 `--basetemp=.cache/pytest/optimization-full-final`，不通过修改存储实现掩盖环境碰撞。[当时日志](verification/1002-optimization-fixes-full-temp-collision.txt)。

未提交、推送、发布或重建 wheel；保留全部既有工作区修改。
