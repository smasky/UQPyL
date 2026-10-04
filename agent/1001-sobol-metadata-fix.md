# 2026-10-01 Sobol 采样元数据一致性补齐

按用户最新选择，SA10 的元数据诊断改为 **`RuntimeWarning`**：Sobol 核对基础样本量 `N`、`secondOrder`、可选 `blockSize` 和实际行数，不再因这些字段不一致而抛异常。能从完整采样结构确认布局时恢复配置并继续计算，无法确认时返回标记未估计的零占位，避免按错误步长输出误导分数。

## 当前协议

- 元数据仍声明 `designType="saltelli"`。N 应为排除布尔值的正整数，secondOrder 应为布尔值，支持对应 NumPy 标量；这些字段缺失/无效时进入告警与恢复流程。
- 一阶块长为 `nInput + 2`，二阶为 `2*nInput + 2`。可选 blockSize 应为对应正整数，行数应为 N×块长；字段类型、矛盾配置和缺/多行一次运行发出一次 warning，恒定输出也不隐藏问题。
- 元数据一致时保留原计算路径和公式。需要恢复时，检查两种候选完整布局中的 A/B 混合坐标复制关系；唯一支持的布局，或剩余一致字段能选定的受支持布局，按实际完整块数计算，状态为 `recovered`。
- 不完整块或无法消除的布局歧义，返回有限零占位，状态为 `not_estimated`、`metrics_available=False`；没有模型调用。原来提供的 Y 按输出选择保留，未提供 Y 时保存 `None`，不伪造输出。
- 不删除、补齐或重排行，不修改调用者的元数据。`result.meta` 保留原声明，settings 的 secondOrder 记录有效阶数，未估计时为 None。

`result.extra["sobol_design"]` 保存 status（validated/recovered/not_estimated）、n_samples、effective_n、effective_block_size、effective_second_order、metrics_available、recovery_basis、issues。它与原始元数据一起持久化；AnaReader 同时补齐对空 artifact payload 的读取，使缺失 Y 的未估计结果能够正常加载。

完整块缺失/多出可以告警后使用剩余/现有完整块，但结构可恢复不证明采样质量或统计精度。常规一致元数据不额外验证全部混合坐标，任意 Y 与 X 是否对应仍由调用者负责；不是完整设计正确性的证明。缺少整个 meta、错误 designType、数组形状/非有限输出及既有 A/B 零方差条件保持原错误协议。本次 warning 约定针对新增元数据校验。

## 验证范围

先前 23 项改为最新 warning / 恢复 / 未估计语义，另新增 5 项到既有 `tests/test_analysis_design_validation.py`，覆盖两种布局均成立但元数据矛盾、可整除却没有混合结构、所选多输出/缺失 Y 的保存与读取、原元数据与有效阶数共同保存。原 `test_analysis_sobol_more.py` 行数检查也调整为 warning 与未估计标记；数值和正常配置对照仍保留。

最新 warning 协议修改前相关两文件 **29 failed、18 passed**。实现后发现 Reader 读取 None payload 的一个失败，补齐后七个相关文件 **199 passed，7.06 秒，`-W error`**，覆盖单位坐标、多输出、极端输出、统计参照和运行生命周期。预期 warning 均由用例捕获。

独立 7 组对照通过：两个正确元数据控制不告警；完整块删除但 N 不变、只改 N、只改 secondOrder 三组 warning 后恢复，指标与正确完整块参照逐值相同；部分块缺失/多出两组 warning 后标记未估计。尤其二阶标记冲突恢复实际二阶布局，没有重现旧的错误分数。原始 X/Y/元数据保持。

本轮最终 py312 全量 **2230 passed，78.74 秒，`-W error` 零未捕获警告**。五个触达 Python 文件 Ruff 静态/格式与 `git diff --check` 通过。此前严格报错阶段的 2225 passed / 67.53 秒保留为历史验证。

最新证据：[warning 修改前日志](verification/1001-sobol-warning-before.txt)、[相关测试](verification/1001-sobol-warning-targeted.txt)、[全量测试](verification/1001-sobol-warning-full.txt)、[独立 7 组对照](verification/1001-sobol-warning-reference.json)、[复核日志](verification/1001-sobol-warning-reference.txt)、[复核脚本](verification/verify_sobol_metadata.py)。此前严格报错阶段的 [数据](verification/1001-sobol-metadata-reference.json) / [全量日志](verification/1001-sobol-metadata-full.txt) 保留追溯，本轮没有覆盖。

SA10 已按最新 warning 约定完成。生产修改涉及 Sobol 元数据恢复/未估计结果及 AnaReader 空 artifact 读取；API 中英文、测试导航、TODO/交接同步。SA09/SA11 仍待处理，MARS 高阶贡献局限保持；未重建 3.14 wheel，未提交/推送。
