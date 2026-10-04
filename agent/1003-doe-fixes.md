# DoE 四项边界修复（2026-10-03）

完成 [DoE复核](1003-doe-review.md) 中 DOE01–DOE04，使用 conda py312。未提交、推送或重建 wheel；其他既有工作区改动保持。

## 修复行为

- **DOE01**：LHS `maximin` / `center_maximin` 仅一个样本时，不再对空距离数组取最小值。发出 RuntimeWarning 后分别退化为 classic / center；同种子与对应基础模式逐值一致。元数据保留请求的 `criterion`，新增公开字段 `effective_criterion` 记录实际模式；之后复用采样器生成多个样本仍使用请求模式。
- **DOE02**：三个优化模式在生成候选前统一校验 `iterations`，接受 Python/NumPy 正整数，拒绝零、负数、小数、布尔值、NaN。非法参数明确 ValueError，不再泄漏 UnboundLocalError。正常多候选的随机数消耗和最大最小距离选优保持。
- **DOE03**：Sobol/Saltelli 共用跳点校验，删除 `N >= skipValue` 人为限制，保留非负整数检查。基础样本数非2的幂，或为2的幂但跳点不是基础样本数整数倍时，发出一次 UserWarning 后继续。对齐的非2幂跳点也允许，例如 N=16、skip=48。Sobol 改用 fast_forward，避免分配整个被跳过前缀；序列与SciPy直接生成对应块逐值相同。仅屏蔽已由模块明确报告的SciPy同类均衡性警告，避免重复提醒，不屏蔽其他警告。元数据保留真实请求值，不改写N/skip。
- **DOE04**：FFD `levels` 元数据始终为独立列表。修改调用方输入或另一次返回的元数据不再改变已返回结果。真实样本矩阵和笛卡尔积顺序保持。

告警示例：

```text
RuntimeWarning: LHS: maximin has no pairwise distance for one sample; using classic.
UserWarning: Sobol skipValue=4 is not aligned to a block of 16 points; balance is not guaranteed. Use skipValue=0 or a multiple of the base size.
```

不对单样本的不存在距离编造优化结果，不通过补点/裁剪偷偷改变Sobol样本数。未对任意模型保证积分或敏感性误差收敛。

## 验证

新增 [51项回归](../tests/test_doe_edge_regressions.py)，覆盖单点恢复/告警/元数据与实例复用、非法优化次数、正常候选独立距离选优、两种设计×scramble开关×5档skip的独立序列/分层/混合矩阵参照、非2幂单次warning、非法skip、FFD双向隔离。原先要求N小于skip报错的旧测试改为验证可用对齐块，原无提示的不对齐配置改为明确检查warning；Sobol文档例子改用对齐skip。

专项 **102 passed，0.98秒，`-W error`**。全量 **3029 passed，79.95秒，`-W error`**；预期warning均由对应测试明确捕获。

重新执行上一轮194条独立记录，**183条正常记录完全一致**，11条受影响记录表现符合修复：两条单点失败变为带warning返回，四条未赋值错误变为明确参数错误，FFD元数据行数恢复，两个大skip正常返回，两个不对齐skip新增warning。正常记录含15次解析模型敏感性公共流程；原线性/乘积误差保持，没有把FAST有限阶误差当作这次修复消除。

- [专项日志](verification/1003-doe-fix-targeted.txt)
- [全量日志](verification/1003-doe-fix-full.txt)
- [修复后194条记录](verification/1003-doe-fix-audit.json)
- [正常记录对照](verification/1003-doe-fix-parity.json)
- [审计脚本](verification/check_doe_review.py)，使用 `--output agent/verification/1003-doe-fix-audit.json` 保留旧复现数据。

Ruff/差异空白检查通过，中英文API文档及测试导航已同步。大型FFD/高维超大样本性能、任意目标的有限预算精度仍不在本轮证明范围内。
