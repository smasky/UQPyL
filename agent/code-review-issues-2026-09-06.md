# 本地代码包检视：问题记录

记录日期：2026-09-06\
状态基线：本地 `dev` 工作区，包含尚未提交的重构和本次会话中已完成的修正。\
用途：后续逐项讨论、修正和验证；本文件中的修复方向是建议，不代表已经实施或所有设计选择均已确定。

## 工作约定

- 以本地版本及用户确认的设计为准；远程仅作历史参照。
- 按用户要求将本轮记录保存在 `agent/`。
- 一次处理一个问题，更新对应条目的状态、决定、修改文件和验证结果。
- 不通过削弱测试或恢复旧接口来绕过新设计。
- P1：优先处理，可能导致错误结果或主要流程不可用。P2：后续处理，涉及边界、数值稳健性或工程交付。
- 证据分为“已复现”“静态确认”“待进一步验证”；不把静态风险写成已测得故障。

## 检视与测试基线

- 173 个 Python 文件完成语法扫描，无语法错误。
- conda `py312` 环境完整测试：313 passed，4 warnings。
- 带分支覆盖率运行耗时 19.21 秒。
- Python 语句覆盖率 87.7%，分支覆盖率 69.6%；不代表原生扩展覆盖率。
- 4 条警告来自 FAST 小样本配置复用辅助频率。
- 审查覆盖各模块结构及关键调用链，并对重点问题做小规模复现；不等于逐算法数学证明、跨平台验证或原生代码完整内存审计。

| 模块 | 语句覆盖率 | 分支覆盖率 |
|---|---:|---:|
| problem | 93.0% | 78.7% |
| doe | 94.6% | 78.4% |
| optimization | 89.2% | 69.2% |
| inference | 85.5% | 72.0% |
| analysis | 90.6% | 75.7% |
| calibration | 90.3% | 63.2% |
| surrogate | 83.3% | 65.4% |
| core | 92.5% | 76.0% |
| viz | 77.1% | 54.7% |

## 待处理索引

| 编号 | 优先级 | 问题 | 证据 | 状态 |
|---|---|---|---|---|
| R01 | P1 | 代理模型共享可变默认组件 | 核隔离与列表隔离已验证 | 已修复核隔离及 MultiSurrogate 列表共享 |
| R02 | P1 | 离散变量重复映射，记录值与评估值不一致 | 优化及五种推断混合变量回归 | 已修复优化/推断评估与记录坐标 |
| R03 | P1 | 不可行解的历史最优比较违反统一约束规则 | 已复现 | 已修复，复用约束优先比较 |
| R04 | P1 | 问题约束权重未传入普通算法种群 | 已复现 | 已修复传递、排序和保存读回 |
| R05 | P1 | EGO 调用 KRG 旧预测接口 | 已复现 | 已修复并通过真实 KRG 集成测试 |
| R06 | P1 | ES/IES 最优样本选择忽略指标方向 | ES、IES 双方向回归验证 | 已修复，使用 normalizedScore 选优 |
| R07 | P1 | PBIAS 直接最小化导致偏差判据错误 | 四种校准方法回归验证 | 已修复距零比较及双侧阈值 |
| R08 | P1 | 标准化后的预测方差未恢复量纲 | GPR、KRG 缩放关系已验证 | 已修复统一逆缩放及多输出形状 |
| R09 | P2 | OptResult 历史引用运行态，被后续运行改写 | 已复现 | 已修复结果深复制并回归验证 |
| R10 | P2 | ES/IES 默认协方差求逆易遇到奇异矩阵 | 两算法秩不足/零秩/满秩验证 | 已修复统一秩感知求解 |
| R11 | P2 | Problem/Space 输入输出校验缺口及重复校验 | 输入规整、必要字段及自定义入口回归验证 | 已修复评估入口与必要字段检查 |
| R12 | P2 | AutoTuner 在数据划分前拟合预处理组件 | 两入口/两模式回归验证 | 已修复先划分再拟合预处理 |
| R13 | P2 | 调参与数据划分随机数链未统一 | 种子复现/覆盖率回归验证 | 已修复独立随机流及 KFold 余数 |
| R14 | P2 | Python 支持声明和 CI 与当前代码不一致 | 三平台 CI 配置完成，本机干净 wheel 安装 844 项通过 | 实现已完成；远程六组矩阵待实跑 |
| R15 | P2 | Pages 仍使用旧文档配置 | 隔离严格构建及 3288 处本地引用验证通过 | 已修复构建入口，未执行线上部署 |
| R16 | P2 | sdist 构建后未纳入发布步骤 | YAML 与产物目录衔接检查通过 | 已修复下载目录，未执行发布 |
| R17 | P2 | 运行异常缺少统一连接关闭与失败状态收尾 | 四类入口、事务及清理失败共 36 项回归通过 | 已修复共享异常收尾 |
| R18 | P2 | 历史存储与高维 HV 的资源开销 | HV 分块及历史降频均经等价回归与资源测量 | 已修复 HV 分块和独立历史频率 |

## R01：代理模型共享可变默认组件

- 位置：`UQPyL/surrogate/gp/gaussian_process.py` 的构造参数与 `setKernel()`；`surrogate/kriging/kriging.py`、`surrogate/rbf/radial_basis_function.py` 有同类写法；`surrogate/base.py` 的 `MultiSurrogate`。
- 原因：`kernel=RBF()` 等默认对象在函数定义时创建，实例共享同一核对象；`setKernel()` 将 `kernel.setting` 绑定为模型自己的 setting，改变其他实例所用状态。
- 已复现：训练 GPR A 后仅创建 GPR B，`A.kernel is B.kernel` 为真，A 的预测显著改变，出现约 1e11 量级结果。KRG 默认核共享身份也已确认；RBF 代理模型的影响尚未单独数值复现。
- 同类已复现：`MultiSurrogate(..., models_list=[])` 的默认列表共享；向 A 添加模型后，新建 B 已包含该模型。
- 建议：默认使用 `None` 并在实例内创建组件；明确显式传入核对象、scaler、模型列表时的所有权与复制规则。
- 验收：创建、训练或改动 B 不改变 A 的参数与预测；不同 MultiSurrogate 的列表独立。

### 2026-09-06：默认核修正

- 讨论决定：本步只修正默认核；显式将同一个核传给多个模型的归属/复制策略留待讨论。
- 修改：GPR、KRG、RBF 代理模型的 `kernel` 默认值改成 `None`，在各次构造时分别创建 RBF、Guass、Cubic 核。显式传入的核仍按原流程使用。
- 回归测试：新增三种模型的实例隔离测试；训练 A 后创建并训练 B，A 的预测保持不变，且各自的默认核及参数归属独立。
- 验证：conda `py312` 运行 `tests/test_surrogate_base_rbf_krg_gpr.py`、`tests/test_surrogate_gp_kriging.py`、`tests/test_surrogate_kernels.py`、`tests/test_surrogate_autotuner.py`，33 项全部通过。
- 未完成：显式共享核的策略、MultiSurrogate 默认列表问题；本步未重跑完整测试包。

### 2026-09-06：传入核作为构建模板

- 讨论决定：外部核只用于构建。构造、setKernel 均复制当前核配置；同一模板可以传给多个模型，不增加归属标记。模板后续修改不影响现有模型。
- 修改：三类 BaseKernel 共享 `KernelTemplate.clone()`；复制当前核参数、范围、类型与配置，排除模型参数和候选选择字段。模型内部仍使用现有 Setting 调参机制，未另行重构参数体系。
- 候选核：setKernelChoices 登记时复制模板，切换时复制选中模板，不将训练参数写回候选；实际运行核通过 model.kernel 查看。
- 自定义扩展：自定义核若持有训练缓存或外部资源，应覆写 clone，仅保留独立的配置。
- 修改文件：`UQPyL/surrogate/_kernel.py`；gp/kriging/rbf 三类核基类及对应模型；`tests/test_surrogate_kernel_templates.py`；中英文 surrogate 文档。
- 验证：新增 12 项参数化回归，覆盖模板复用与训练隔离、外部修改与 setKernel、内部核配置复制、候选核切换。相关 41 项测试通过；随后完整测试 328 项通过，4 条原有 FAST 警告。
- 未完成：MultiSurrogate 默认列表问题仍待独立处理。

## R02：离散变量重复映射

- 位置：`UQPyL/doe/base.py::sampleWithMeta()`、`problem/space.py::map_discrete_vars()`、`optimization/base.py::_sampleInitialPop()/evaluate()`、`inference/base.py::evaluate()`。
- 已复现：输入编码区间 `[0, 1]`，离散集合 `[10, 20, 30]`。GA 初始种群记录 `[30, 10, 10, 20]`，目标函数实际收到 `[30, 30, 30, 30]`。
- 原因：DOE 已调用 `unit_to_space()` 映射为真实取值，算法评估再次调用 `apply_var_type()`；评估后也没有同步写回实际决策值。
- 影响：目标值与记录参数不对应，搜索与后续结果复用可能失真。推断存在同类转换路径，需单独回归验证。
- 决策与实施：优化内部种群/搜索边界统一 `[0,1]`，DOE 新增 `output="unit"`；用户初始解编码一次；评估用解码副本；输出/历史/日志/SQLite/NPZ 使用真实决策值及原始目标方向。代理模型训练、预测与去重使用规范单位编码；推断坐标变换保持独立，相关路径尚未修复。
- 验收：连续、整数、离散及混合变量端到端测试，记录的实际参数与模型收到的参数一致；映射不重复应用。

## R03：不可行解的历史最优比较错误

- 位置：`UQPyL/optimization/runtime/result.py::OptState._updateSingle()`。
- 已复现：旧解目标 0、违反度 10，新解目标 100、违反度 1，历史 best 仍保留旧解。
- 原因：可行性相同时直接按目标值比较，两个不可行解没有走违反度比较。
- 建议：复用 `optimization/core/constraint.py` 的统一比较规则，避免另写一套历史最优逻辑。
- 验收：覆盖可行对不可行、双方可行、双方不可行及权重场景；最终 best 与 Population 选优语义一致。

## R04：约束权重未传播

- 位置：`UQPyL/optimization/base.py::evaluate()` 及种群构造路径。
- 已复现：问题设置 `conWgt=[100, 1]`，评估后的 `Population.conWgt` 仍是 `None`，选优与加权规则相反。
- 建议：明确权重归属，并确保初始种群、子代、合并、替换和历史最优都使用一致的权重。
- 验收：构造加权与无权重排序相反的两个解，验证算法实际选择与问题权重一致。

## R05：默认 EGO 与 KRG 接口不兼容

- 位置：`UQPyL/optimization/expensive/ego.py::EI()`；`surrogate/kriging/kriging.py::predict()`。
- 已复现：普通一维平方问题，`EGO(nInit=4, maxFEs=5)` 第一次正式迭代报 `TypeError: KRG.predict() got an unexpected keyword argument 'only_value'`。
- 原因：EGO 使用旧 `only_value=False`，KRG 当前接口为 `returnStd/returnVar`。
- 测试缺口：`tests/test_optimization_more_algorithms_smoke.py` 的 `_DummySurrogate` 保留旧参数，掩盖真实组件不兼容。
- 建议：统一预测调用，并核对 EI 消费的是方差还是标准差；零方差处的稳定处理一并检查。
- 验收：默认真实 KRG 与小预算内部优化器至少完成一轮 EGO 更新，不仅使用替身测试。

## R06：ES/IES 忽略指标方向

- 位置：`UQPyL/calibration/methods/es.py::_runCore()`、`methods/ies.py::_runCore()`。
- 已复现：ES 配置 `metric='nse'`，后验分数约 `[0.757, 0.939, 0.757]`，记录的 best 对应较低分。
- 原因：直接执行 `argmin(scores)`，没有使用基类已经提供的方向归一化逻辑。IES 有相同代码路径，未单独数值复现。
- 建议：选择使用 `normalizedScore()`，展示与诊断保留原始分数。
- 验收：RMSE 最小、NSE/KGE 最大两个方向均正确，ES 和 IES 均覆盖。

## R07：PBIAS 的零偏差语义

- 位置：`UQPyL/calibration/base.py::_resolve_metric()/normalizedScore()`、`calibration/util.py::pbias()`、`methods/glue.py`。
- 已复现：观测为 1，模拟候选为 0 和 1，GLUE 使用 PBIAS 时把模拟为 0 的样本选为 best；阈值 5 也会接受大幅负偏差。
- 原因：带符号 PBIAS 被当作越小越好，负偏差越大反而越优。
- 待定：采用绝对 PBIAS、距零损失或双侧区间阈值；不要擅自改变原始指标显示含义。
- 验收：等幅正负偏差具有对称质量判定，零偏差最佳，阈值解释明确。

## R08：预测方差未逆缩放

- 位置：`UQPyL/surrogate/gp/gaussian_process.py::predict()`、`surrogate/kriging/kriging.py::predict()`、`surrogate/base.py` 与 `scaler.py`。
- 原因：GPR 的均值经过 `__Y_inverse_transform__()`，方差仍处于训练时的缩放空间；KRG 的缩放处理不完整，不能笼统表述为完全没有处理。
- 已复现：隔离默认核共享问题、显式创建独立核，GPR 使用 StandardScaler；目标扩大 10 倍后均值约扩大 10 倍，返回方差比例仍为 1，而非 100。
- 2026-09-06 核专项复核更正：KRG 在 `_objFunc()` 中对 StandardScaler 的 sigma2 已有缩放处理；普通 StandardScaler 的对应方差比例实测为 100。MinMaxScaler 路径实测比例仍为 1。其他自定义缩放配置与多输出行为需进一步验证。
- 建议：为 scaler 定义标准差/方差逆变换能力，明确多输出方差形状与量纲。
- 验收：对输出的平移和缩放做数值关系测试，同时检查 `returnStd`、`returnVar`。

## R09：OptResult 与运行态共享历史

- 位置：`UQPyL/optimization/runtime/result.py::buildResult()/reset()`。
- 已复现：同一个 GA 连续运行两次，`r1.history is r2.history` 为真；第一次结果中的历史参数变成第二次运行的数据。
- 建议：在 Result 构建时生成独立快照，或在新运行开始时创建独立 State/History；同时核对嵌套数组和 extra 的共享语义。
- 验收：保存 r1 后再次运行、重置或修改算法，r1 不变；其他模块 State/Result 按相同原则检查。

## R10：ES/IES 默认求逆的秩不足问题

- 位置：`UQPyL/calibration/methods/es.py` 的 `inv(c_yy + r)`，`methods/ies.py` 的 `inv(c_yy + r + lam * I)`。
- 已复现：3 个集合成员、5 个观测值、默认零误差协方差，ES 报 `LinAlgError: Singular matrix`。
- 数学原因：样本协方差秩不超过集合规模减一；默认 IES 的 `lam=0` 也没有解决秩不足。
- 待定：要求用户提供有效 R，还是提供显式正则化/低秩求解策略；避免自动添加缺乏说明的误差模型。
- 建议：使用合适的线性求解替代显式求逆，补协方差形状、有限性、对称性与相关数值条件检查。
- 验收：观测维数大于集合规模、重复观测、病态协方差等场景有确定行为与可理解错误。

## R11：Problem/Space 校验边界不完整

- 位置：`UQPyL/problem/space.py::validate()`、`problem.py::_validate_eval_result()`、`base.py::__init_subclass__()`。
- 已复现：`objFunc` 返回 None 时，`evaluate(..., target='objs')` 正常返回空 Eval；三维 `(2, 1, 3)` 输入被 Space.validate 接受。
- 静态确认：子类包装器计算 `x2d` 后仍将原始 `X` 传给覆写方法；默认 Problem/ModelProblem 的输入与结果校验存在重复调用。
- 建议：统一一次输入规整、一次执行和一次结果校验；要求请求的必要字段存在，维数符合契约。
- 待定：NaN/Inf 的支持政策、边界值与零宽区间策略；不要无差别禁止可能作为惩罚值的 Inf。
- 验收：一维单样本、二维批量、高维错误输入、缺失字段和自定义 evaluator/覆写方法均覆盖。
- 完成日期：2026-09-07。
- 修改：包装器先检查合法 target，并将规整后的二维数组传给覆写方法；移除默认 Problem/ModelProblem.evaluate 内重复的输入和 Eval 结果校验。普通问题的全部/目标请求必须有 objs；两类问题声明 nCon > 0 时全部/约束请求必须有 cons。修复标量 sims 的 IndexError，改为明确的维度错误。Space 高维输入拒绝已在此前空间调整中完成，本次补充入口回归。
- 保留：纯模拟 ModelProblem 的 target=None 允许只有 sims；sims 必要性仍只针对 None/sims 请求。Inf 目标惩罚值保持支持。独立 simFunc/simulate 的边界校验仍保留；子类显式调用 super().evaluate() 时各层仍校验自己的返回值，防止外层修改结果后绕过检查，未引入共享“正在校验”标记。
- 修改文件：problem/base.py、problem/problem.py、problem/model_problem.py；tests/test_problem_eval_contract.py；中英文 problem API 文档。
- 验证：py312 全量测试 752 passed、4 条既有 FAST 频率警告（15.82 秒）；随后补充自定义 evaluator 空结果与 super 调用后改坏结果的回归，专项共 29 项。全量日志 `/tmp/uqpyl-problem-contract-full.log`。

## R12：AutoTuner 预处理先于数据划分

- 位置：`UQPyL/surrogate/auto_tuner.py::optTune()/gridTune()`。
- 静态确认：对全量数据调用 `prepareTrainingData()`，其中 scaler 执行 fit，再划分训练/验证集。
- 影响：验证数据参与预处理统计量计算；实际评分偏差大小未做基准量化。
- 建议：先划分原始数据，仅用训练集拟合 scaler，验证集仅 transform；选择最佳参数后在全量数据重新拟合。
- 验收：验证集极端值不影响训练 scaler 的统计量，最终模型仍使用全量训练数据。

## R13：调参与拆分随机数链不统一

- 位置：`UQPyL/surrogate/split.py`、`surrogate/auto_tuner.py`。
- 静态确认：RandSelect/KFold 使用全局 `np.random.shuffle`，不消费 AutoTuner.rng。
- 建议：拆分器显式接受 seed/rng，调参器派生子种子，结果记录拆分信息。
- 验收：同一 seed 可复现拆分与调参结果，且不改变全局 NumPy 随机状态。
- 附带待验证：KFold 样本数不能整除折数时，当前尾部样本不进入验证折；需单独构造覆盖测试，再决定修正。

## R14：Python 支持范围与 CI

- 位置：`pyproject.toml`、`.github/workflows/ci.yml`、`optimization/runtime/result.py` 等。
- 静态确认：声明支持 Python >=3.8，但存在未启用延迟求值的 `np.ndarray | None` 等运行时类型标注；与旧解释器不兼容。未在 Python 3.8/3.9 实测。
- CI 当前仅 Windows + Python 3.8/3.9，自动执行依赖提交消息包含 `CI`，没有正常 PR 测试触发。
- 待定：真实最低支持版本。支持声明、代码和构建矩阵应统一。
- 建议：选定支持范围后，在正常 push/PR 执行检查，并验证安装后的 wheel，不能仅在源码目录测试。
- 验收：所声明支持的代表性平台/解释器能够安装、导入并运行必要测试。
- 2026-09-07 决定与修改：最低 Python 版本设为 3.10，开发环境保持 3.12。pyproject 声明、classifiers 和 wheel 构建矩阵移除 3.8/3.9；NumPy 运行依赖简化为 numpy>=1.26；CI 测试版本改为 3.10/3.12，移除 3.8 专用 netCDF4 安装分支；中英文 README 和快速开始同步。
- 验证：py312 下 TOML/YAML 解析通过，Requires-Python 拒绝 3.8/3.9 并接受 3.10/3.12；包内全部 Python 源码通过 Python 3.10 语法解析；本地 3.12 导入检查通过。当前机器只有 py312 环境，未实际运行 3.10，也未触发远程 CI。
- 剩余：本次按用户要求调整支持版本，CI 提交消息门槛、PR 触发、跨平台测试及安装 wheel 后验证仍待处理，R14 尚未全部关闭。
- 2026-09-07 后续实现：CI 去除提交消息门槛，普通分支 push、pull_request 和手动触发均执行；矩阵为 Ubuntu/Windows/macOS × Python 3.10/3.12，共六组。每组以 pyproject 隔离构建 wheel，再由 .github/scripts/test_wheel.py 创建全新 venv 安装该 wheel[viz] 与 pytest/pytest-cov，将完整测试集复制到源码目录之外执行。先检查 UQPyL 来自 venv，并强制导入全部 10 个编译扩展，避免可选导入跳过掩盖缺失。CI 不执行发布。
- 测试适配：四项源码布局/文档检查改从实际导入的 UQPyL 路径取文件；conftest 为配置的 basetemp 创建父目录，修复干净环境启动时的 FileNotFoundError。移除 pytest 9 已删除的 PytestRemovedIn9Warning 过滤项。干净安装发现可视化代码使用 seaborn，但 viz extra 未声明；补齐可选依赖，核心运行依赖未增加。
- 报告：CI 保存 wheel、JUnit 和覆盖率 XML；覆盖率文件路径映射回仓库，避免临时 venv 删除后无法关联源代码。六组任务仅配置 contents: read，不授予发布权限。
- 本地构建验证：复制当前包源码及构建配置到 /tmp 隔离目录，首次构建排除既有 .so/.pyd 等二进制文件。隔离构建产物为 uqpyl-2.1.6-cp312-cp312-macosx_12_0_arm64.whl，包含 10 个本地扩展，元数据 Requires-Python >=3.10；修正 viz 元数据后重新构建并验证。
- 本地安装验证：通过与 CI 相同的脚本在全新 venv 测试，pip check 通过；全部 10 个扩展导入成功；844 passed、0 skipped、4 条既有 FAST 警告（44.14 秒），覆盖率约 90%。运行依赖由 pip 独立解析（NumPy 2.5.3、SciPy 1.18.1），pytest 9.1.1；不借用开发环境的 site-packages。另在开发环境的全新 basetemp 下运行受影响及异常测试，48 passed。
- 证据：构建日志 `/tmp/uqpyl-wheel-build.log`，安装/测试日志 `/tmp/uqpyl-wheel-test.log`，报告 `/tmp/uqpyl-wheel-test-reports/`，wheel `/tmp/uqpyl-wheel-artifacts/`。YAML 触发/矩阵/权限检查、Python 3.10 语法检查和相关 diff 空白检查通过。
- 验收边界：当前仅实测 macOS ARM64/Python 3.12，尚未推送或触发远程工作流，Linux、Windows、Python 3.10 以及远程 runner 上的六组结果仍待首次 CI 验证；不能将本地通过等同于全部平台已通过。

## R15：Pages 仍指向已删除旧文档

- 位置：`.github/workflows/pages.yml`、`mkdocs.yml`、`docs_v2/`。
- 静态确认：Pages 执行 `mkdocs build -f mkdocs.yml`；配置仍引用旧 docs 页面，工作区已删除 docs 目录。未执行实际部署。
- 建议：确定 docs_v2 的正式构建入口，同步导航和资源引用。
- 验收：从干净检出构建文档，导航和资源有效，再验证部署产物。
- 完成日期：2026-09-07。
- 修改：mkdocs.yml 显式使用 docs_v2，英文/中文两个导航入口覆盖全部 38 个文档页面；移除旧页面与失效 CSS 引用，使用 Material 内置样式、现有 logo/favicon。补充中文优化章节固定锚点，修复中文优化 API 的章节链接。
- 构建：Pages 与 Read the Docs 共用 docs-requirements.txt；明确 MkDocs >=1.6,<2，Pages 增加 --strict，导航遗漏和无效章节锚点按警告处理并使严格构建失败。保留现有发布触发规则。
- 验证：在 /tmp 隔离 Python 3.12 文档环境安装已有文档依赖，将当前 mkdocs.yml 和 docs_v2 单独复制到全新临时目录，执行 `python -m mkdocs build -f mkdocs.yml --clean --strict` 成功（0.99 秒）。38 个源文档生成 39 个 HTML（含 404），全部 3288 处本地 href/src 与章节锚点检查通过，搜索索引存在，源 assets 复制后字节一致。日志 `/tmp/uqpyl-docs-verify.log`。
- 范围：未推送、未触发 Pages/Read the Docs 部署，未检查外部站点/CDN 可用性；验证对象为当前本地文档快照，并非远程已提交版本。

## R16：sdist 没有被发布

- 位置：`.github/workflows/build.yml`。
- 静态确认：源码包下载到 dist/，发布步骤 `packages-dir` 只有 wheelhouse/。
- 建议：将本次发布所需 wheel 与 sdist 放入一致的发布目录，校验版本及产物清单。
- 验收：发布前的产物检查同时看到 wheel 和 sdist；本记录不授权执行发布。
- 完成日期：2026-09-07。
- 讨论决定与修改：按最小修改方案，仅将 publish 任务中 Download sdist 的 path 从 dist 改为 wheelhouse。源码包仍在独立构建任务中生成至 dist/*.tar.gz 并上传为 source-distribution；发布任务将 wheel 和源码包汇集到 wheelhouse，由现有发布步骤一起读取。未增加重复的 twine 检查。
- 验证：py312 解析 workflow YAML，通过断言确认两类产物下载目录与 packages-dir 一致、源码包 artifact 名称衔接正确、wheel 合并下载及发布任务构建依赖保持；git diff --check 通过。
- 未覆盖：未执行远程构建或 PyPI 发布，未验证源码包安装编译；实际发布产物清单仍需在下次发布运行中确认。

## R17：异常路径的运行收尾

- 位置：各模块 base 的 run/analyze/setup/finalize，以及 `core/runtime_storage.py`。
- 静态确认：主要流程依赖正常走到 finalize 才关闭 session；异常路径缺少统一失败标记和 finally 清理。尚未逐模块注入异常复现。
- 建议：建立共享生命周期保障，异常时记录失败状态并释放资源，同时保留原始异常。
- 验收：模型、评估器、存储阶段分别注入异常，检查连接关闭、数据库状态及再次运行行为。
- 完成日期：2026-09-07。
- 复现：GA 目标函数抛出 RuntimeError 后，原实现的 session 仍挂在实例上，数据库状态为 running，连接仍可执行 SELECT。分析、校准的 saveResult 在结果写入完成前先提交 finished 状态亦已确认。
- 讨论决定与实现：新增 core/runtime_lifecycle.py 的共享 RunLifecycle，由四类基类统一保护公开 run/analyze（包括子类覆写和 super 委托），从初始化到最终输出共用一次会话生命周期；正常运行结束关闭连接，异常回滚未提交事务、写入 failed/结束时间/运行时长，优化推断记录已有 FEs/iters，然后关闭并原样抛出异常。未改具体算法计算主体。
- 存储：BaseSqliteStorage.create_run 对未返回给调用者的连接也执行异常清理，先提交基本运行记录，再写参数；推断连接 PRAGMA 移入受保护的连接配置钩子。分析/校准的结果数据和 finished 状态在完整写入后一起提交。已提交快照保留。close 成功后清空 session.conn，可重复关闭。
- 异常优先级：收尾错误附加到原始异常（Python 3.10 附加诊断警告）；回滚失败时不继续提交状态，防止误提交半成品；仍尝试关闭连接。关闭本身失败则保留已经挂载的会话供排查，禁止新运行覆盖。无自动重试、断点续跑或部分成功返回。
- Reader/文档：AnaReader 摘要补充 status，与其他三类 Reader 一致；中英文 API 入口说明失败记录、部分数据和手动 setup/finalize 的责任边界。
- 验证：tests/test_runtime_failures.py 共 36 项通过，覆盖四类真实方法的模型错误、setup 后失败、保存失败、finalize 后失败与实例复用；覆盖建库/参数初始化失败、KeyboardInterrupt、子类 super 生命周期、真实结果序列化回滚、部分快照保留并经 Reader 读回、回滚/状态写入/关闭二次异常。py312 全量 792 passed、4 条既有 FAST 警告（17.84 秒），日志 `/tmp/uqpyl-runtime-failures-full.log`。
- 限制：存储不可写时无法保证 failed 状态落盘；没有可用 run 记录的早期建库失败仅保证尝试资源清理；进程强制终止/断电不在范围内。直接调用 setup/finalize 的调用者仍负责异常时的会话管理。本次未在其他 Python 版本实测。

## R18：长任务资源开销

- 位置：`UQPyL/optimization/runtime/result.py::_updateHistory()`、`optimization/metric/hv.py::HV()`。
- 静态确认：每轮无条件复制完整种群，内存随迭代增长；高维 HV 默认一百万采样点，广播比较形成与采样量、种群量和目标维数乘积相关的中间数组。
- 待定：历史记录频率、是否保存全量种群、内存与 SQLite 的分工、HV 计算频率及精度预算。
- 建议：历史策略可配置，HV 分块计算；高维 HV 的随机数应接入显式 rng，避免指标复现漂移。
- 验收：在明确规模与内存预算下做压力测试，记录实际耗时、峰值内存和历史完整性。
- 2026-09-07 本步决定：先修复高维 HV 中间数组开销，暂不调整历史记录、计算频率和采样精度预算。rng 接口及优化运行态固定随机流已在此前实现，本步保持。
- 修改：HV 增加仅限关键字的 batchSize（默认 4096），Monte Carlo 采样按批生成，候选解按最多 256 个分块比较，逐批累加被支配点数。默认采样量仍为 100 万；相同 NumPy RNG 初始状态下，样本序列、估计值及计算后 RNG 状态保持一致；低维精确分支不变。高维非退化计算拒绝非正 nSamples 和非法 batchSize。
- 验证：新增 tests/test_hv_batching.py 共 21 项，覆盖单点、跨候选块、跨采样块及尾批、负目标与归一化、参考点外候选、RNG 消费一致性、非法参数和精确分支。相关专项 63 项通过；py312 全量 813 passed、4 条既有 FAST 警告（16.60 秒），日志 `/tmp/uqpyl-hv-batching-full.log`。
- 资源实测：独立子进程比较旧整块广播与新实现，峰值为 macOS resource.ru_maxrss（含约 35 MiB 的进程基线）。100 万样本/40 解/6 目标：348.39→43.61 MiB，0.765→0.734 秒；20 万样本/300 解/8 目标：562.67→85.78 MiB，1.001→0.966 秒。两组 HV 及下一随机数均完全相同。一次本机测量仅用于验证资源改善，不代表通用加速比例。脚本 `/tmp/uqpyl-hv-benchmark.py`，原始结果 `/tmp/uqpyl-hv-benchmark-results.jsonl`。
- 修改文件：optimization/metric/hv.py；tests/test_hv_batching.py；中英文 optimization API 文档。
- 剩余：每轮完整种群/最优档案的历史复制、历史保存频率及内存与 SQLite 分工仍待讨论，R18 未全部关闭。
- 2026-09-07 后续完成历史策略：14 个内置优化器统一支持 historyFreq，默认 10，记录初始更新、迭代编号为 10 倍数的更新及最终更新；1 为逐更新完整快照，None 为仅最终快照。最终快照不重复追加；完整当前最优解、候选与非支配档案保留。通过 set('historyFreq', value) 可在运行前调整；非正、非整数和布尔值拒绝。
- 每轮统计与快照分离：bestObjHistory、bestMetricHistory、numBestHistory、metrics、improvedHistory、iterToFEs 继续逐次更新；populations/bests 按 snapshotIterToFEs 对齐，导出键为 snapshot_iter_to_fes。result.extra 的 history_freq 记录本次策略；最终真实坐标和目标方向恢复不变，结果仍为独立副本。
- SQLite：saveFlag/saveFreq 与内存频率独立；周期保存调用 includeHistory=False 的当前结果视图，避免为了单个 SQLite 快照深复制整段历史。最终保存仍使用完整最终结果。NPZ 保持原有最终值与统计曲线协议，不额外加入完整种群历史。
- 历史专项：tests/test_optimization_history_policy.py 共 31 项通过，覆盖单/多目标带约束最大化方向、四种频率的曲线与结果等价、初始/最终和尾轮映射、短任务去重、再次运行隔离、SQLite 独立频率与 Reader 读回、非法频率以及全部 14 个优化器传参。py312 全量 844 passed、4 条既有 FAST 警告（16.61 秒），日志 `/tmp/uqpyl-history-policy-full.log`。
- 历史资源实测：独立进程用固定 400 解、30 变量种群模拟 1500 次状态更新并生成最终结果，隔离历史维护成本。historyFreq=1/10/None 分别保留 1500/151/1 个快照、全部 1500 条统计；快照数组总量 142.26/14.32/0.095 MiB，进程峰值（含约 108 MiB 基线）403.02/137.89/109.13 MiB。最终最优值与统计曲线求和一致。脚本 `/tmp/uqpyl-history-benchmark.py`，原始数据 `/tmp/uqpyl-history-benchmark-results.jsonl`；这不是完整算法性能基准。
- 边界：降频降低增长速度，不是固定内存上限；轻量统计、算法实时非支配档案和 SQLite 文件仍随工作量增长。需要避免完整历史累积时可选 None；本步未裁剪真实 Pareto 档案，未改变 HV 精度预算或计算频率。上述已确认的 R18 两处开销现均已处理。

## 其他后续检查建议

以下尚未提升为已确认缺陷，处理时先补证据：

- 初始已评估种群的目标方向、约束完整性与维度是否得到充分检查。
- maxFEs/maxIters 的严格上限与整批评估之间如何约定。
- ES/IES 更新后参数是否允许越界、是否支持整数与离散空间；仿真是否可以避免重复执行。
- 常量列/常量目标经过 scaler 时的零分母行为。
- 高维 HV 的参考点、归一化与跨轮次可比较性。
- 原生扩展的干净构建、数组边界、资源释放和跨平台行为。
- 历史设计草稿与当前文档中相互冲突的接口描述。

## 本轮审查前已修复的事项

这些事项不属于上面的待处理清单，避免后续重复讨论。

| 事项 | 用户确认的决定 / 修正 | 验证 |
|---|---|---|
| ModelProblem target 与 sims 冲突 | None 返回所有可用字段；objs/cons 仅返回所请求字段；sims 仅返回仿真输出并跳过目标约束计算；仿真输出仍在仿真阶段校验 | 新增 8 项回归；相关 65 项通过 |
| 推断测试类未适配抽象基类 | ConstrainedQuadratic 改为继承 Problem，使用 validate | 推断 37 项通过 |
| pytest.warns(None) 不兼容 | 改用 warnings.catch_warnings，保留 SyntaxWarning 检查 | 完整测试通过 |
| 重启次数测试与本地默认不一致 | 保留本地 GPR/KRG 默认 nRestartTimes=1，更新测试预期 | 完整测试通过；不代表重启行为和拟合质量已经全面验证 |

## 推荐处理顺序

1. R01：组件隔离。
2. R02：变量空间语义。
3. R03、R04：逐项统一约束处理。
4. R05、R08：真实 EGO 集成与不确定性量纲。
5. R06、R07、R10：逐项处理校准指标与数值策略。
6. R09、R11、R12、R13：结果快照、契约和调参验证。
7. R14—R18：交付、异常收尾与资源策略。

## 逐项处理记录模板

处理某项后在对应条目追加：

- 讨论决定：
- 修改文件：
- 验证命令与结果：
- 未覆盖范围：
- 完成日期：

R01 的默认核与传入核隔离已修复，MultiSurrogate 默认列表仍待处理；R02 优化链路、R05、R09 已修复；R02 推断路径与其余条目继续按各条状态跟进。

## 核函数专项复核

新增问题及逐核结论见同目录 `kernel-review-2026-09-06.md`，编号 K01—K08。已修复 K01—K05、K07；K06 已增加稳定积分路径并修复复现的大 nu/极小距离溢出；K08 已完成内置 Python 核参数域、有限性、维度及调优边界校验。详细验证见专项记录的修复状态。

## 2026-09-06：优化单位区间改造

- 新增 `space_to_unit()`、`canonicalize_unit()`；整数与数值离散变量采用等宽区间，中点作为唯一代表编码。支持固定连续维和非整数上下界内的合法整数；无界优化不支持本单位区间约定。
- 7 个 DOE 采样器支持 real/unit 输出，默认 real；7 个单目标、4 个多目标、3 个代理辅助优化算法统一内部单位坐标，保留 Problem 原边界。
- EGO/ASMO/MOASMO 内层问题采用连续单位区间、最小化方向；实际模型使用解码真实值。训练集等价真实解保留首次观测。小型有限域候选耗尽可提前结束，避免重复真实评估。
- 默认代理模型无额外输入缩放；用户显式提供的 scaler 继续生效，不改变其模型配置。
- PSO 初始速度改为零，避免单位坐标全非负带来的方向偏移。ASMO 距离阈值按单位空间解释。
- 输出真实坐标需要创建快照，顺带修复 R09 的历史引用共享；结果 extra 同样深复制。
- R05 改用 `returnVar=True`，处理零方差 EI，并通过真实 KRG 的 EGO 测试。输出边界同时修复最大化分数未还原的问题，支持多目标混合方向；内部状态保留最小化分数。
- 新增回归文件 `tests/test_optimization_unit_space.py`，覆盖变换/采样、所有普通优化器混合变量路径、真实代理训练预测、初始解、ModelProblem 约束评估、SQLite/NPZ、最大化和结果历史隔离。
- 本次不把优化单位坐标约定施加到推断/贝叶斯算法；R02 的推断路径仍需独立处理。
- 最终验证：新增单位空间回归 **33 项通过**；conda py312 完整测试 **539 passed, 4 warnings，14.62 秒**。4 条为既有 FAST 频率警告，日志 `/tmp/uqpyl-unit-full-tests.log`。相关文件 diff 空白检查通过。

## 2026-09-07：R03 历史最优约束比较

- `OptState._updateSingle()` 复用 `compareSolutions()`：可行优先，两个不可行解比较加权约束违反程度，两个可行解比较最小化方向的目标分数；不可行违反度相同保留旧解。
- `bestFeasible` 使用与比较函数一致的约束违反计算，沿用种群现有 conWgt（包括零权重的现有语义）。当时 R04 的传播尚待处理；现已完成，见后续 R04 修复记录。
- 新增 10 项回归，覆盖更优/更差违反度、违反度相同、可行性转换、可行目标比较、无约束、加权比较及零权重；同时检查最优决策/约束、出现时刻和历史 improved 标记。
- 验证：conda py312 完整测试 **549 passed, 4 warnings，16.12 秒**；4 条为既有 FAST 频率警告。日志：`/tmp/uqpyl-feasibility-history-tests.log`。

## 2026-09-07：R04 约束权重贯通

- 优化基类在预评估初始种群入口及每次实际评估后接入 Problem.conWgt；当前 Problem 是权重来源，覆盖传入 Population 的旧配置。权重独立复制，切片、选优、合并、替换保留权重并清理失效排名缓存。
- 权重必须与 nCon 匹配、有限且非负。`sum(max(cons*conWgt,0))` 的公式不变；None 表示不加权，零权重忽略对应约束。
- NDSort 增加 conWgt 并贯通 Population、NSGAII、NSGAIII、MOASMO 调用；RVEA 原环境选择完全不读取约束，本次补为有可行解时仅对可行解做参考向量选择，否则按加权违反度保留候选。
- 同时修复 NDSort 在存在不可行解时忽略 nSort 返回最终全部排名的问题，避免环境选择保留过多解。
- 多目标 bestFeasible 与加权可行性一致。结果 extra、历史种群及 SQLite JSON 快照记录 constraint_weights，OptReader 恢复种群权重；NPZ 同样包含 constraint_weights。约束数组仍保存原始值，仅违反程度汇总使用权重，避免重复加权。
- 打印、日志、SQLite 汇总统一加权违反程度。
- 新增 `tests/test_optimization_constraint_weights.py` 共 27 项：权重校验/公式、预评估初始种群、复制切片合并替换、11 个普通优化算法、3 个真实代理辅助优化、多目标排序、零权重及 SQLite/NPZ 读写。
- 验证：conda py312 完整测试 **576 passed, 4 warnings，14.86 秒**；4 条是既有 FAST 频率警告。日志：`/tmp/uqpyl-weights-tests.log`。

## 2026-09-07：多目标与约束专项复核

- 专项问题 M01—M08 见 `multiobjective-review-2026-09-07.md`，复现脚本 `multiobjective-review-repro-2026-09-07.py`。
- 确认结果/前沿/HV 语义、MOEA/D 空后代与 NaN、RVEA 零参考向量、等违反度排名等问题；MOASMO 约束候选引导尚未实现。NSGA-III 截距边界和不可行理想点影响列为进一步验证项。
- 本轮仅检查和记录，没有修改生产代码。独立普通排序 200 组、加权优先级 100 组通过；现有约束权重、结果、单位空间相关测试 **75 passed，0.78 秒**。上述缺陷由专项例子复现，说明现有测试尚未覆盖，不能据通过数判定算法完全正确。

## 2026-09-07：多目标专项修复

- 按确认方案完成 M01—M07：历史可行档案、不可行候选分离，固定参考点/原始尺度 HV，MOEA/D 小种群及聚合数值修复，RVEA 参考尺度回退，等违反度同层；同时保护 NSGA-III 退化几何。
- 档案在所有真实评估之后、环境选择之前更新，包含预评估初始解；输出、历史、打印、SQLite/Reader/NPZ 与文档同步。
- M08 约束代理及廉价真实约束候选引导仍待独立讨论；精确档案增加筛选/内存开销，高维 HV 的 R18 不在本轮关闭。
- 详细协议、限制及新增 27 项回归见 `multiobjective-review-2026-09-07.md` 的实施状态。
- 最终验证：conda py312 全量 **603 passed, 4 warnings，15.35 秒**；4 条既有 FAST 频率警告；日志 `/tmp/uqpyl-mo-full.log`。触达文件 diff 空白检查通过。
- 后续用户明确：MOASMO 当前按无约束问题定位，M08 不作为当前待修缺陷，约束扩展暂不讨论。


## 2026-09-07：R08 Scaler 与不确定性逆变换

- 保留可选 scalers 和原默认值，不更改优化的 0–1 输入编码，不引入依赖。
- 内置 Scaler 统一保存仿射逆变换尺度，新增 inverse_transform_std/var；均值带平移，标准差和方差仅按一次/二次尺度恢复。
- GPR/KRG 通过预测输出公共接口统一恢复方差；KRG sigma2 不再提前针对 StandardScaler 缩放，修复自定义 sitaX、MinMaxScaler 与多输出方差广播/聚合问题。
- 单样本/常数列使用单位源尺度回退，保留常规 StandardScaler ddof=1；补充拟合状态、有限值、特征数及目标范围/标准差校验。1D 输入保持单行语义。
- 自定义 Scaler 未实现 inverse_transform_var 时，均值预测可继续使用，不确定性请求显式报不支持；不自动假设任意非线性变换也能线性还原方差。
- 新增 tests/test_surrogate_scaler_regressions.py 共 42 项：直接仿射公式、常数列/单样本、非法参数、10 倍目标+平移关系、多输出与独立单输出对照、自定义 Scaler 错误提示。
- 验证：conda py312 全量 **645 passed, 4 warnings，15.20 秒**；4 条为既有 FAST 警告，日志 `/tmp/uqpyl-scaler-full.log`。本次触达代码与文档 diff 空白检查通过。


## 2026-09-07：R06 ES/IES 指标方向

- ES、IES 最终 best 索引改为 argmin(normalizedScore(后验模拟))；该方法只统一最小化方向，不做尺度归一化。diagnostics 的原始 scores、priorScores 及 IES 的 scoreMean 保留原指标值。
- 不改变集合更新公式；PBIAS 的距零语义 R07、协方差奇异矩阵 R10 仍独立待处理。
- 新增 tests/test_calibration_metric_direction.py 共 6 项，覆盖 ES/IES × RMSE/NSE/KGE，使用真实 ModelProblem 执行更新，独立计算原始指标并核对 bestIdx/bestDecs/bestSim、原始分数及后验集合不随指标选择而改变。
- 验证：conda py312 全量 **651 passed, 4 warnings，15.01 秒**；4 条为既有 FAST 警告；日志 `/tmp/uqpyl-cal-direction-full.log`。触达代码/文档 diff 空白检查通过。


## 2026-09-07：R07 PBIAS 距零比较

- 按用户确认，保留原始带符号 PBIAS 公式；字符串 metric="pbias" 启用 metricClosestToZero，normalizedScore 返回 abs(raw)。自定义函数仍按现有默认最小化规则，不按函数名推断。
- GLUE 复用统一比较分数，阈值 5 表示 [-5%, +5%]，端点包含；拒绝负数及非有限 PBIAS 阈值。SUFI2、ES、IES 自动沿用统一距零排序。
- diagnostics/scores/behavioralScores/eliteScores 仍使用原始带符号值，未改变模型评估与集合更新公式。
- 新增 tests/test_calibration_pbias.py 11 项，覆盖符号及误差相抵、对称阈值/端点/零容差、非法阈值、大负偏差拒绝、四种方法选优和自定义函数约定。
- 验证：conda py312 全量 **662 passed, 4 warnings，15.89 秒**；4 条为既有 FAST 警告；日志 `/tmp/uqpyl-pbias-full.log`。触达代码与文档 diff 空白检查通过。


## 2026-09-07：R10 ES/IES 协方差求解

- 共用 methods/_ensemble.py：R 的维度、有限性、对称性、半正定检查；lam 为有限非负标量。只在舍入容差内对称化及截断 R 的微小负特征值。
- 满数值秩用 solve 替代显式 inv；秩不足用对称特征分解的伪逆，阈值为 nObs*eps*最大绝对特征值；零秩增益为零。不改变默认 R=0、lam=0，不暗加误差模型或正则化。
- ES 记录 covarianceSolve，IES 每轮记录 covarianceSolves（solver/rank/dimension/cutoff）。继续使用原指标与更新公式。
- 新增 tests/test_calibration_covariance.py 20 项：默认 3 成员/5 观测与独立 pinv 对照、零集合跨度、多轮 IES、有效 R、非法 R/lam、满秩 solve 数值对照和病态方向截断。
- 限制：伪逆不能拟合集合未表达的信息；当前观测空间特征分解增加秩检测开销，不涉及集合空间加速或超长观测内存优化。
- 验证：conda py312 全量 **682 passed, 4 warnings，15.20 秒**；4 条为既有 FAST 警告；日志 `/tmp/uqpyl-covariance-full.log`。触达代码与文档 diff 空白检查通过。


## 2026-09-07：R12 AutoTuner 预处理泄漏

- optTune/gridTune 在原始数据上先生成训练/验证划分，只用训练子集调用 prepareTrainingData；验证预测输入保持原始值，复用训练集拟合的 Scaler，并按原始 Y 评分。
- 选定最优参数后重新以全量原始数据拟合预处理，再按 joint/separate 模式拟合最终模型，不复用训练子集的统计量。
- 新增 tests/test_autotuner_preprocessing.py 8 项：两入口 × 两模式 × StandardScaler/MinMaxScaler。固定划分将极端值留在验证集，追踪 X/Y Scaler 拟合数据、验证原始输入，独立核对候选评分和最优参数，并对照全量拟合模型及输入不变性。覆盖多项式特征路径。
- R13 的随机数链和 KFold 余数问题仍单独待处理，本轮不修改划分策略。
- 验证：conda py312 全量 **690 passed, 4 warnings，15.37 秒**；4 条为既有 FAST 警告；日志 `/tmp/uqpyl-tuner-preprocessing-full.log`。触达代码与文档 diff 空白检查通过。


## 2026-09-07：R13 数据划分与随机数链

- RandSelect/KFold.split 增加 keyword-only seed/rng，禁止同时传入；采用独立 Generator，不修改全局 np.random。RandSelect 按实际验证比例切分，KFold 使用 array_split 均匀分配余数；校验比例、折数和 mode。
- AutoTuner 两个入口增加 seed/rng，未传时沿用自身 rng；拆分、模型及外层优化器使用三个派生种子。每次候选和最终拟合重置模型到同一模型种子，不互相消耗拆分/外层优化随机流。
- tuner.lastSplit 内存记录 snake_case 的训练/验证索引和子种子，既有返回二元组不变，不增加自动文件保存。自定义模型需使用提供的 rng，不能保证自行调用外部全局随机源的组件可复现。
- 新增 tests/test_surrogate_split_randomness.py 共 18 项：KFold 全覆盖/均衡/单折一致、seed复现/rng推进/冲突校验、全局状态及输入不变、真实 DE optTune 与 gridTune 结果重现、separate 模式模型随机源重置、RandSelect 比例。R12 固定拆分测试桩同步接收新增关键词。
- 验证：conda py312 全量 **708 passed, 4 warnings，15.17 秒**；4 条为既有 FAST 警告；日志 `/tmp/uqpyl-split-full.log`。触达代码与文档 diff 空白检查通过。


## 2026-09-07：R01 MultiSurrogate 列表隔离收尾

- 按用户确认，将 models_list 默认值改为 None，每个实例创建独立列表；显式传入列表复制容器，保留模型实例引用，不深复制模型。
- 直接验证默认实例互不影响、外部清空不影响内部、内部 append 不改变外部空列表，以及模型 identity 保持不变。
- 验证：代理回归、优化权重和单位空间相关测试 **62 passed，0.77 秒**，列表隔离直接检查通过。本次未重复运行全量测试；触达文件 diff 空白检查通过。


## 2026-09-07：R02 推断坐标收尾

- 根因确认：initialSampling 使用 DOE 默认真实值，evaluate 再 apply_var_type，导致离散值重复映射；后续提议存链时还可能记录映射前值。
- 修复采用连续潜在状态：初始 DOE output=unit 后按原边界映射内部坐标；整数/离散轴通过统一等宽解码器映射合法真实值，连续轴保持原物理值。模型评价和 logProbFunc 均使用真实参数副本，内部提议/链/适应性协方差/历史提议档案不取整、不吸附中点。
- InfState 收集时解码；最终结果、best、打印、SQLite 快照和结果 artifact 坐标一致。未更改既有全链 objs 的内部目标方向约定或硬约束规则。
- 共享反射边界处理固定维，避免零跨度取模；多选项离散变量的零跨度内部区间显式拒绝。
- 新增 tests/test_inference_mixed_coordinates.py 19 项，覆盖五算法×有/无约束、逐次真实评估/记录/logProb 对齐、潜在提议不被修改、固定维、SQLite 往返、等宽整数区间及 MH 已知三点质量分布。
- 范围限制：本轮没有对所有提议核（尤其自适应/snooker）的整体理论正确性及收敛性做独立证明；没有将推断所有连续轴改成单位尺度，也未处理 R17 异常清理和 R18 资源策略。
- 生命周期测试的简化 Problem 桩补充 Space/unit_to_space 协议，以覆盖新增运行前边界校验。最终 conda py312 全量 **727 passed, 4 warnings，15.73 秒**；4 条既有 FAST 警告；日志 `/tmp/uqpyl-inference-coordinates-full.log`。触达代码、文档与测试桩 diff 空白检查通过。
