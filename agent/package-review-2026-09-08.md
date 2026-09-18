# UQPyL 整体复审：正确性、模块布局与精简

日期：2026-09-08。范围：当前本地工作区；保留此前全部未提交修改。本轮不修改实现。

处理状态与验收标准统一维护在 [TODO 表](TODO.md)；本文保留审查时的证据与判断。

## 审查范围和基线

- 统计到 177 个 Python 源文件，约 19,556 行；另有 Cython/C++ 扩展源文件。按模块审查公共入口、主要算法路径、组件交界、结果与持久化，针对可疑分支做小型复现。
- Python 使用 `/opt/homebrew/anaconda3/envs/py312/bin/python`。
- 本轮完整测试：**844 passed，4 warnings，18.23 秒**。4 条仍为 FAST 辅助频率复用警告。日志：`/tmp/uqpyl-review-20260908-full.log`。
- 上轮已完成本机干净 wheel 安装后的 844 项测试。本轮未重新构建 wheel，也未执行远程六组 CI、发布或部署。
- 原生扩展本轮侧重 Python 调用边界、数据所有权和构建入口，未对全部第三方 C/C++/Cython 源码逐行证明正确性。
- 下列 N01—N20 为**本轮新确认的 20 组问题**，有运行复现或明确代码证据；结构建议单列，不把风格偏好当作 bug。
- 最小复现通过临时 Python 程序执行，未另外堆积复现脚本。本文保留输入、关键观测和定位，便于逐项修复时转成针对性测试。

## 整体判断

保留现有领域划分：`problem` 定义问题与空间，`doe` 产生设计，`surrogate` 拟合近似模型，`optimization / inference / calibration / analysis` 各自执行任务，`runtime / reader / viz` 管理结果与展示。当前的主要问题不在目录数量，而在数值语义、可变数据所有权、重复计算和多处维护同一份信息。

现有 844 项测试说明已覆盖路径稳定通过，但不能据此判定数值算法整体可靠。此次复现的似然方向、平滑矩阵、输出尺度不变性、历史状态独立性，都需要检查数学性质或运行之间的关系，单看形状和有限值不足以发现。

既有决策继续保留：普通 Problem 与 ModelProblem 分开；Eval 为正式评价协议；优化内部使用 unit 坐标；推断保留既有 latent 坐标策略；外部核作为构建模板复制；Scaler 恢复预测量纲；校准按 metric label 解释方向；PBIAS 保留符号并按距零评分。本轮不讨论 MOASMO 的约束支持。

## 模块布局意见

| 模块 | Python 文件 / 行数 | 判断与最有价值的精简方向 |
|---|---:|---|
| problem | 17 / 2,460 | Space、Eval、Problem/ModelProblem 的职责应保留。把变量类型等检查放在空间构造入口，清理旧 transform 别名与错误导出；不让每个算法再次补相同校验。 |
| doe | 10 / 1,050 | base + methods 已经清楚。优先纠正 LHS 相关性矩阵方向，一维特例直接退化成普通 LHS。保留采样默认真实值及显式 unit 输出。 |
| surrogate | 50 / 4,400 | 数学家族继续分开。重点是修正目标方向、平滑与调参数据流；删除无用方差计算、废弃 ensemble 占位和重复参数切片逻辑。 |
| optimization | 37 / 4,912 | soea / moea / expensive / core / runtime 划分合理。保留各算法独立循环；把终止检查变成只读判断，在状态更新处维护计数和停滞信息。 |
| inference | 14 / 2,361 | base / chain / methods / runtime 合理。先修 DREAM archive 的数据所有权，再收拢重复的参数检查；避免为了统一而合并各采样算法。 |
| calibration | 11 / 1,317 | 最值得精简的模块：一次模拟同时供打分、协方差更新和结果保存使用。ES/IES 可共用单步更新；runtime 文件拆分仅属后续整理。 |
| analysis | 22 / 2,037 | 在公共 analyze 入口一次整理二维数据、列选择与标签，方法内部只保留分析计算。删除外层 7 个旧转发文件。 |
| core | 9 / 447 | 保持小型共享基础层。适合统一 runId、连接与基础 reader 行为，不宜扩成通用算法框架。Params/Setting 的两种用途仍有区别。 |
| viz | 6 / 536 | 只负责绘图。SQLite → 真实结果对象的恢复应由 reader 完成，删除动态伪 History 和手工拼结果的重复实现。 |

包入口另有 1 个文件、36 行。不同模块不必有完全相同的目录深度；目录对称本身不是重构理由。

## 新确认的问题

### N01 · P1 · GPR 内部超参数优化方向反了

位置：[似然返回](../UQPyL/surrogate/gp/gaussian_process.py:223)、[传给优化器](../UQPyL/surrogate/gp/gaussian_process.py:150)。MP 和 EA 路径都在最小化直接返回的 log likelihood。

复现：12 个 `[0,1]` 均匀点、`y=sin(6x)`，固定 `C=1e-6`，长度尺度范围 `[0.1,3]`，LBFGSB，模型 RNG seed=2。固定长度 0.3 时 log likelihood=22.5145、训练 RMSE=0.0000252；内部优化选择长度 3，log likelihood=-531107.25、RMSE=0.1969。

建议：优化目标统一为负 log likelihood；明确 fitState 中记录的是原似然还是最小化目标。补“优化后目标不会向错误方向退化”的小型验证。

### N02 · P1 · Boxmin 可越界；内部优化器还会修改全局 RNG

位置：[Boxmin 移动](../UQPyL/surrogate/util/boxmin.py:86)、[Boxmin 随机数](../UQPyL/surrogate/util/boxmin.py:21)、[LBFGSB 随机数](../UQPyL/surrogate/util/lbfgsb.py:22)。乘法移动负坐标且未统一裁剪；Setting 的 log 坐标经常为负，这不是仅存在于无关自定义输入中的问题。

复现：bounds `[-3,-1]`，xInit `[-3]`，目标 `f(x)=x`，Boxmin 返回 -3.56762135，目标函数也收到越界输入。默认 GPR.fit 会改变 NumPy 全局随机状态。

建议：局部 Generator；内部优化器必须尊重整个实数有界区间。可讨论复用已经存在的 LBFGSB 作为默认，以减少维护一种内部优化器的负担；更换默认前应比较 GPR/KRG 的实际拟合效果，不能仅凭代码更短就删除 Boxmin。

### N03 · P1 · RBF 的 smoothing 加到了整个增广矩阵

位置：[radial_basis_function.py](../UQPyL/surrogate/rbf/radial_basis_function.py:135)。`get_A_Matrix(xTrain) + C_smooth` 会同时改动核块、趋势块、约束零块。

复现：12 个 `[0,1]` 点，默认 Cubic，`y=1+2x`。C_smooth=0 时最大训练误差 2.46e-13；C_smooth=1 时误差 0.7651。参照只对核块对角加平滑项的系统仍恢复趋势系数 `[2,1]`，径向系数约 1e-15。

建议：只修改对应核块的正规化项；逐一核对各核的符号约定。随后可将 LU + 两次 pinv + 相乘简化为直接解线性系统，并单独规定奇异时的处理方式。

### N04 · P2 · AutoTuner 的参数应用不支持展开的向量参数

位置：[SurrogateABC.applyParameterValues](../UQPyL/surrogate/base.py:160)、[AutoTuner 调用](../UQPyL/surrogate/auto_tuner.py:113)。按参数名取一项，与 Setting 按参数维度切片的语义不一致。

复现：二维异质 GPR 核初始化后，`applyParameterValues(['l'], np.log([2,3]))` 报 IndexError；实际 optTune + GA + joint 路径也失败。

建议：保留一份权威参数切片映射，先应用结构参数，再按当前有效切片写数值参数，删除按名字一项的第二套展开规则。

### N05 · P1 · Lasso 原地改写调参复用的训练数据

位置：[linear_regression.py](../UQPyL/surrogate/regression/linear_regression.py:169)、[原地中心化](../UQPyL/surrogate/regression/linear_regression.py:181)。`np.asarray(order='F')` 不保证复制。

复现：20 个 `[0,1]` 点、`y=10+2x`、Lasso、C=0.01；同一 prepared 数据连续训练两次，y 均值从 11 变为 0，预测最大变化 10.0543。实际 gridTune 放入两个完全相同的 C 候选，固定 seed=1，验证预测仍差约 10.0532。

建议：只在 Lasso 要原地修改的工作数组上显式复制，不让 AutoTuner 为所有模型、每个候选都做大范围深复制。

### N06 · P1 · DREAM_ZS archive 保存的是会改变的当前状态视图

位置：[初始 archive](../UQPyL/inference/methods/dream_zs.py:105)、[warm-up 写入](../UQPyL/inference/methods/dream_zs.py:131)、[正式采样写入](../UQPyL/inference/methods/dream_zs.py:169)。接受提议后原地改 X_cur，旧 archive 项随之改变。

复现：二维平坦目标，3 链，warmUp=0、maxIters=3、seed=3；两次提议入口之间，初始三个 archive 点已经改变，第二次 archive 六项全部与 X_cur 共享内存。

建议：写入 archive 时复制数组，保留现有列表结构。这里已证明的是历史数据污染，不据此声称 DREAM 其他数学步骤都已验证。

### N07 · P1 · 优化终止判断会在持续改进时早停，迭代计数也越界

位置：[checkTermination](../UQPyL/optimization/base.py:169)、[GA 先更新状态](../UQPyL/optimization/soea/ga.py:114)。比较 previousBest 时，state 已被上代 update 更新，所以经常是在比较同一代最优值，停滞次数仍不断增长。

复现：5 维 Sphere，GA(nPop=40,maxFEs=4000,maxTolerates=2,tolerate=1e-12,historyFreq=1)，seed=2。最佳值连续 12.6177→10.1408→7.7653→2.9923，improvedHistory 全 True，却在 FEs=160、tolerateTimes=4 停止。

另有 `iters <= maxIter` 与检查本身递增：maxIters=0 仍执行一代、返回 iters=2；maxIters=1 执行两代、返回 iters=3。

建议：checkTermination 只读判断；完成一代时加计数；状态更新时比较新旧结果并更新停滞次数。`nPop=40,maxFEs=41` 实际评估 80 次则另属批量预算契约，应先决定硬上限还是按整批结束的软上限，不与上述确定错误混为一谈。

### N08 · P1 · 校准结果仍与下一轮运行共享 history 等可变对象

位置：[CalState.buildResult](../UQPyL/calibration/runtime.py:124)、[reset](../UQPyL/calibration/runtime.py:98)。history 直接共享，diagnostics 和 extra 只有浅复制。

复现：IES(maxIters=2) 得到 r1，设 maxIters=1 再运行得到 r2，r1.history 从两条变一条，且 r1.history is r2.history。诊断中的 scores 数组也与 state 相同。

建议：在 buildResult 出口统一生成独立结果快照。此前优化结果的 R09 修复已完成；本项是校准出口仍有同类遗漏，不是撤销旧结论。

### N09 · P1 · 敏感度随输出量纲改变，非恒定输出被归零

位置：[Sobol](../UQPyL/analysis/methods/sobol.py:117)、[FAST](../UQPyL/analysis/methods/fast.py:61)、[RBDFAST](../UQPyL/analysis/methods/rbd_fast.py:100)。默认绝对容差 np.isclose(...,0) 将小标准差/方差当成常数。

复现：各方法使用对应的设计样本，分别比较 Y=X0+2X1 与 1e-10×Y。原始输出下，三种方法 S1 约为 `[.1999,.8001]`、`[.1995,.7982]`、`[.1781,.7919]`；各自仅改变输出尺度后，全部得到 `[0,0]`。

建议：稳定缩放后计算，基于实际变化判断退化输入。补输出乘非零常数后敏感度不变的验证，保留真正恒定输出的处理。

### N10 · P2 · LHS correlation 比较样本行，且一维会崩溃

位置：[lhs.py](../UQPyL/doe/methods/lhs.py:129)。corrcoef 默认 rowvar=True，得到 nSamples² 相关矩阵，应比较的是输入列之间的相关性。

复现：LHS('correlation') 对一维问题采 10 点、seed=1，报 UnboundLocalError: H。

建议：按列计算相关性、提取非对角项，一维直接用普通 LHS，同时去掉内部打印。这也将相关矩阵的空间规模从样本数平方降到维数平方。

### N11 · P2 · 非法 varType 被接受，编码时静默丢失该维度

位置：[Space 构造](../UQPyL/problem/space.py:46)、[编码默认值](../UQPyL/problem/space.py:145)。先转 int32，再仅收集 0/1/2，没有验证原值是否合法。

复现：Space(2,ub=10,lb=0,varType=[0,3]) 将真实 `[2,8]` 编码成 `[.2,.5]`，而 unit `[.2,.8]` 解码成 `[2,8]`，往返不一致且没有提示。

建议：构造时拒绝非 0/1/2、非整数的类型值；只在这一处维护规则。

### N12 · P2 · analysis 的输入整理顺序分散，并丢失所选输出标签

位置：[check_Y](../UQPyL/analysis/base.py:139)、[Sobol 提前读取列数](../UQPyL/analysis/methods/sobol.py:82)。方法在公共 reshape 之前就使用 Y.shape[1]。

复现：Sobol/FAST 传 `(n,)` 外部单输出 Y 会 IndexError。双输出问题 objLabels=['flow','temperature']，RSA index=1 的数值来自第二输出，标签却成 ['obj1']。

建议：在公共 analyze 入口一次整理 X/Y、样本对应关系、列选择和所选标签；删除各 method 重复的 setProblem/check_Y/__check_X_Y__ 调用和自行生成标签逻辑。

### N13 · P2 · 预评估 initialPop 绕过完整评价契约

位置：[isEvaluated](../UQPyL/optimization/population.py:28)、[跳过评价](../UQPyL/optimization/base.py:98)、[入口转换](../UQPyL/optimization/base.py:117)。仅凭 objs 非空判断已评价。

复现：nCon=1，四行 initialPop 有 objs、没有 cons；初始化被接受，评估完下一代后才在合并时报 inconsistent constraint state。

建议：在入口复用 Eval 的行数、目标维度和必要字段校验；完整数据复用，不完整数据立即报清晰错误或按确定策略补评价。

### N14 · P2 · ES/IES/SUFI2 对同一参数重复模拟

位置：[ES](../UQPyL/calibration/methods/es.py:60)、[IES 单步](../UQPyL/calibration/methods/ies.py:110)、[IES 收尾](../UQPyL/calibration/methods/ies.py:83)、[SUFI2 elite](../UQPyL/calibration/methods/sufi2.py:143)。

计数复现：ES 三批模拟只有两组唯一输入；IES(maxIters=3) 十批模拟只有四组唯一输入；SUFI2 已有全体模拟，还重跑 elite。对昂贵模型是直接成本，对随机模拟器还可能让打分与保存对应不同模拟实现。

建议：一次模拟保留完整 sims，通过 mask 取得有效观测；IES 把后验模拟结果传给下一步；elite 从已有数组索引。这是减少逻辑同时降低运行成本的优先项，不需要引入全局缓存系统。

### N15 · P2 · SQLite 恢复后绘制的 HV 曲线横坐标错位

位置：[viz 恢复结果](../UQPyL/viz/optimization.py:33)、[截断横坐标](../UQPyL/viz/optimization.py:62)。保留全部 iter/FEs，却单独过滤掉 None 指标。

复现方式：使用三个快照的最小 reader 重放恢复与取坐标函数，未另建真实 SQLite。FEs=[10,20,30]、HV=[None,2,3]，应画 `(20,2),(30,3)`，实际画 `(10,2),(20,3)`。还会硬写 bestFeasible=True；动态 History 没有 toDict，拼出的 OptResult.toDict 会失败。

建议：reader 恢复真实 OptResult/OptHistory，指标与坐标成对保留/过滤，viz 删除手工结果重建代码。

### N16 · P2 · 运行标识和日志文件名可能冲突，冲突清理会改写旧状态

位置：[runId](../UQPyL/core/runtime.py:20)、[创建失败清理](../UQPyL/core/runtime_storage.py:30)、[优化日志 fallback](../UQPyL/optimization/runtime/verbose.py:236)、[校准日志](../UQPyL/calibration/runtime.py:154)。

runId 时间精度为分钟，随机后缀只有 4 位十六进制。同一分钟连续生成同算法/同问题的 2,000 个 ID，本轮得到 1,966 个唯一值、34 个重复值。

强制同 ID 的两次 GA 保存复现：第一次状态 finished；第二次主键冲突报 IntegrityError，异常清理却把第一条既有记录改成 failed。说明清理没有确认该 run 是否由当前调用成功创建。

另有更常见日志覆盖：saveFlag=False、logFlag=True，连续运行同一 GA 两次，只留下同一个分钟+进程号日志；校准日志采用秒时间戳也有同秒覆盖风险。

建议：每次运行开始就创建统一且足够长的 runId，无论是否启用 SQLite，日志和数据库都复用它；创建失败只清理本次拥有的记录。这样同时删除各模块生成 fallback 文件名的多套逻辑。

### N17 · P2 · load_algorithm 没有恢复原始运行配置

位置：[reader](../UQPyL/optimization/runtime/reader.py:61)、[参数保存](../UQPyL/core/runtime_storage.py:79)、[属性配置](../UQPyL/optimization/base.py:31)。配置分散在 params 和对象属性；reader 只读 runParam，然后用构造默认值补缺失项。

真实保存/读回复现：GA maxFEs=18→50000，maxIters=3→1000，tolerate=None→1e-6，verboseFlag=False→True；nPop、saveFreq、historyFreq 能保留。

建议：明确一份可恢复配置导出，读取已有 run 表的预算信息，并明确组件实例等不能自动还原的边界；不要把不完整恢复包装成复现原运行。简单值解析可用 ast.literal_eval 替代 eval，不需要新包。

### N18 · P2 · CalReader.list_runs 会混入其他模块的数据库

位置：[BaseReader](../UQPyL/core/runtime_reader.py:23)、[CalReader](../UQPyL/calibration/reader.py:10)。仅凭几个通用列是否存在筛选数据库。

复现：目录中只保存一次 MH 推断运行，CalReader.list_runs 仍列出该 MH；再用 CalReader.get_run_summary 读取，报 KeyError: nSeries。

建议：持久化中有明确的领域/结果类型标识，reader 在入口核对所属模块；不要在每个后续方法内捕获错表错误。

### N19 · P2 · 优化运行耗时始终为 0

位置：[reset](../UQPyL/optimization/runtime/result.py:349)、[结果导出](../UQPyL/optimization/runtime/result.py:282)、[更新入口](../UQPyL/optimization/base.py:151)。runtime 初始化为 0，优化路径没有像 inference 一样刷新。

复现：每批目标函数休眠 0.04 秒，GA 三批评价，实际约 0.14 秒；verbose 开关两种情况均返回 result.runtime=0，SQLite run.runtime 和 snapshot.elapsed 也全为 0。

建议：在运行开始建立计时，在更新与最终出口读取 elapsed，展示层只消费该值。失败收尾已有独立计时，但不能替代成功路径的耗时记录。

### N20 · P2 · problem 的 __all__ 包含列表，公开星号导入失败

位置：[problem/__init__.py](../UQPyL/problem/__init__.py:27)。前两项是 sop/mop 列表而非名字字符串。

复现：`from UQPyL.problem import *` 报 `TypeError: Item in UQPyL.problem.__all__ must be str, not list`。

建议：用扁平字符串列表管理公开入口；顺便决定 benchmark 是否继续全部顶层导出，不要再增加另一层同名别名。

## 从精简角度，值得做的事

### 1. 优先删除不需要的计算与重复数据

- **重复模拟：** N14 一次评价、多处使用，直接减少真实模型调用；不用建通用缓存。
- **无条件方差：** [GPR.predict](../UQPyL/surrogate/gp/gaussian_process.py:110) 在只请求均值时仍求方差并生成 nPred×nPred 核矩阵；[KRG.predict](../UQPyL/surrogate/kriging/kriging.py:168) 也无条件求方差。只需均值时提前返回；真正请求方差时再考虑核对角和分批计算。
- **重复持久化：** [校准保存](../UQPyL/calibration/runtime.py:220) 存了完整 result，又将其中数组、history、diagnostics 分别存一遍。本轮 100 样本×30 时间点 GLUE 示例，result blob 28,825 字节，全部 artifact 共 57,906 字节，约 2.01 倍；这是 payload 比例，不是完整数据库磁盘占用比。
- **优化保存：** [完整 population JSON](../UQPyL/optimization/runtime/storage.py:84) 与 [逐成员记录](../UQPyL/optimization/runtime/storage.py:126) 重复存同一批决策/目标/约束。按实际 reader 查询需求选择一种主表示，元数据单独保留。
- **摘要读取：** CalReader.get_run_summary 现在调用 get_artifacts，把全部大对象反序列化后才读少量字段。将摘要字段独立查询，避免每次列详情都加载整段模拟。

### 2. 清理已经失去作用的兼容与占位

- analysis 外层 7 个转发文件可在统一导入路径时删掉。DeltaTest 还通过 sys.modules 查旧入口以适配补丁，见 [delta.py](../UQPyL/analysis/methods/delta.py:155)；测试应补丁实际使用处。
- Space 与 ProblemBase 各保留 4 个旧 transform 包装；不要求向后兼容时，确认文档/示例调用迁移后可删除。
- surrogate/ensemble 有两个空文件；weightEnsemble 导入不存在的 surrogateABC，且未接入正式导出。未形成可用功能的占位宜移出正式包。
- RBF 的 `_get_tail_matrix` 已由 kernel 的矩阵构造承担，生产路径不调用；`_kernelChoiceRegistered` 只有赋值没有读取。清理时也删除只为保留死代码而存在的分支测试。
- state/result、params/setting 的同对象别名可逐步收敛：运行态叫 state，返回快照叫 result；算法参数叫 params，代理模型的可调参数空间仍可叫 Setting。不能只做全局替换，需同步调用处。

### 3. 合并真正相同的小块逻辑

- Reader 连接建立、关闭、context manager、get_run、get_run_params 可放进已有 BaseReader；各领域 schema 和结果类型仍保持独立。
- GPR/KRG/RBF 的核模板安装、旧参数移除、Setting 合并可用小型辅助函数减少重复。三种核家族的数学运算不合并。
- analysis 公共输入整理、校准模拟数据复用、runId 生命周期，各保留一个负责点，分别解决本轮已复现问题。
- Evaluator/ModelEvaluator 对独立 obj/con 回调可按 target 决定执行，而不是两个都运行后再丢弃结果；不必增加新的配置层。

### 4. 保留有明确价值的抽象

保留 Eval、Space、领域基类、独立结果对象、核 clone、Scaler 和现有共享异常收尾。`__init_subclass__` 自动包装会增加追踪成本，但它目前承担入口校验和异常资源收尾，不建议在本轮为减少几行代码再推翻这一机制。

不为目录整齐把所有 runtime 合成一个大模块，也不把优化、MCMC、敏感性分析和校准硬塞进同一执行引擎。精简以减少重复执行、减少重复状态、减少错误表达方式为标准。

### 5. 测试和交付侧的调整

- 测试优先覆盖数学关系：似然优化方向、线性趋势再现、敏感度尺度不变性、有界提议、重复候选结果一致性、历史独立性、读写后配置和坐标对齐。
- 对同一契约可使用少量参数化案例；不为了行覆盖率保留旧别名和无调用方法。
- 现有 CI 远程六组矩阵仍需真正运行；发布流程还构建 Python 3.11/3.13 等组合，不能把本机 3.12 验证描述为所有声明平台已验证。
- 保持当前 docs_v2 为文档入口。暂不做新的文档体系或包装层迁移。

## 建议处理顺序

1. **数值和结果可靠性：** N01、N03、N05、N06、N07、N08、N09；N02/N04 与代理模型调参一起处理。
2. **边界契约：** N10—N13、N15、N20，补少量实际使用路径验证。
3. **减少模型成本：** N14，随后均值预测快路径与重复存储；先测调用次数/数组规模，再比较效果。
4. **运行与恢复：** N16—N19，统一一次运行的标识、耗时、所属领域和可恢复配置。
5. **小范围收尾：** 删除旧转发、死代码和无用别名，复用 reader 公共代码。独立小批次完成，不把它们捆成全包重写。

建议下一项从 **N01：GPR 超参数优化方向** 开始：错误明确，影响直接，修复边界小，也容易用数值对照验证。后续仍可按一个问题一个问题讨论和处理。
