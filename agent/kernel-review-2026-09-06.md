# 核函数专项检查

日期：2026-09-06。以当前本地实现为准。以下逐项保留初次检查证据；最新修复状态见下表。

范围：GP 5 个 Python 核（含未在包入口导出的 Constant、DotProduct）、Kriging 3 个核、RBF 5 个核及其基类/模型调用边界；另静态检查了 LIBSVM 的 linear、polynomial、rbf、sigmoid 公式分支。

环境：conda py312。现有 kernel、kernel_templates、gp_kriging 三组测试共 31 项通过。以下复现说明目前测试仍有功能覆盖缺口。

## 修复状态（2026-09-06）

用户授权后完成以下修复，原始复现内容保留用于追溯。

| 编号 | 当前状态 | 实现与验证 |
|---|---|---|
| K01 | 已修复 | Exp/Cubic 使用逐维绝对差；三种 KRG 均验证符号对称与训练点重建 |
| K02 | 已修复 | Cubic 改为 DACE 逐维乘积；验证明确数值、紧支撑与二维样本矩阵半正定 |
| K03 | 已修复 | 修正尾项局部维度，零距离直接返回 0，不改输入；验证只读数组、增广矩阵及实际拟合 |
| K04 | 已修复 | Constant 正确生成自相关和矩形交叉相关矩阵；保持原导出范围 |
| K05 | 已修复 | 构造时将物理 nu 编为候选区间中点；验证四候选读回、clone、调优值切换；拒绝非法 nu/非候选 |
| K06 | 已修复已复现的数值问题 | 零距离精确赋 1；普通阶数保留 Bessel 快速计算，nu>=50 或非有限中间结果改用经模态中心化/缩放的等价 Gamma 积分；近零仅在误差上界小于机器精度时赋 1。新增 70 位独立参考值、宽范围扫描、Gaussian 极限及真实 GP 拟合测试；性能代价是大 nu 路径较慢 |
| K07 | 已修复 | 固定参数独立规整维数并保留常量身份；GP/KRG 无可调参数时直接拟合；验证标量广播、克隆和所有固定参数下的公开 fit |
| K08 | 已修复本轮约定范围 | 13 个内置 Python 核加入参数域、有限性、标量/向量维度、连续调优边界及输入特征形状检查；构造、初始化及整批计算前校验；非法 Setting 更新在使用前拦截 |

新增回归测试：`tests/test_surrogate_kernel_regressions.py`，38 项通过。完整测试：conda py312 环境下 **366 passed, 4 warnings，14.20 秒**；4 条均为既有 FAST 采样频率警告。日志：`/tmp/uqpyl-kernel-fixes-full-tests.log`。

Gaussian 的 epsilon 定义、RBF 平滑项及 GP 默认长度下界未改，继续独立讨论。

## 逐核结论

| 系列 | 核 | 结论 |
|---|---|---|
| GP | RBF | 常规正长度尺度公式与交叉矩阵检查通过；固定参数和非法参数见 K07/K08 |
| GP | RationalQuadratic | 常规正 l/alpha 公式检查通过；文档字符串误写为 Constant Kernel；K07/K08 |
| GP | Matern | 常用 nu=0.5/1.5/2.5/inf 分支检查通过；可调 nu 与一般 nu 分支见 K05/K06 |
| GP | Constant | 分支写反，见 K04；当前未在 kernel/__init__.py 导出 |
| GP | DotProduct | X @ Y.T + sigma² 的常规输入检查通过；当前未在 kernel/__init__.py 导出 |
| Kriging | Guass | exp(-sum(theta * D²)) 符合当前系数定义；K07/K08 |
| Kriging | Exp | 非负距离下公式正常，真实预测路径有符号问题，见 K01 |
| Kriging | Cubic | 有符号处理与多维组合均有问题，见 K01/K02 |
| RBF | Linear | epsilon*r 与常数尾项未发现明显公式错误 |
| RBF | Cubic | (epsilon*r)³ 与线性多项式尾项未发现明显公式错误 |
| RBF | Gaussian | 当前定义为 exp(-epsilon*r²)，不同于将 epsilon 直接乘在 r 上再平方的约定；属于参数定义差异，不直接判为公式错误 |
| RBF | Multiquadric | sqrt(1+(epsilon*r)²) 与常数尾项未发现明显公式错误；不同库可能使用相反整体符号，不能仅据此判错 |
| RBF | ThinPlateSpline | 拟合入口报错，且修改调用方距离数组，见 K03 |
| SVR | 四类 LIBSVM 核 | 静态检查未发现明显公式抄写错误，未进行完整 C++ 数值/内存审计 |

“未发现明显错误”限于此次公式与小样本检查，不是任意参数和规模下的正确性保证。

## K01 [P1] Kriging Exp/Cubic 没有处理预测中的带符号差值

- 位置：`UQPyL/surrogate/kriging/kernel/exp_kernel.py::__call__()`、`cubic_kernel.py::__call__()`；`kriging.py::predict()`。
- 训练 D 来自逐维 pdist，非负；预测 D 来自 `xPred - xTrain`，可以为负。
- Exp 实测 theta=0.5、D=±0.2，返回约 0.9048 和 1.1052，相关值不对称且大于 1。
- Cubic 在相同差值下返回 0.972 和 0.968，也不对称。
- 实际 KRG 路径：4 个一维样本、sin(3x) 目标、固定 theta=0.5，Exp 训练点预测最大误差约 0.727，Cubic 约 2.958；Guass 对照约 1.3e-13。
- 建议：统一 D 契约；这两类平稳相关核按逐维绝对差计算。补 K(D)=K(-D)、训练/预测矩阵一致性和训练点重建测试。

## K02 [P1] Kriging Cubic 的多维组合方式错误

- 当前先 `sum(D*theta, axis=1)`，再计算一个三次函数。
- DACE 对应结构是逐维计算 `xi=min(1, theta_i*abs(D_i))`，再对 `1-3*xi²+2*xi³` 取乘积。
- 已复现：default_rng(123) 生成 20 个二维 [0,1] 样本，theta=1，当前相关矩阵最小特征值约 -0.331；不是浮点舍入级别误差。三维样本也复现负特征值。
- 建议：核对并采用逐维乘积公式；测试多维样本的相关矩阵对称性、半正定性和紧支撑边界。
- 来源：DACE 手册第 2.3 节、表 2.1： https://www.omicron.dk/dace/dace.pdf 。

## K03 [P1] ThinPlateSpline 拟合失败及距离数组副作用

- 位置：`UQPyL/surrogate/rbf/kernel/thin_plate_spline_kernel.py`。
- `get_Tail_Matrix()` 使用不存在的 `self.n_samples/self.n_features`，实际已取得局部 nSample/nFeature。
- 已复现：`RBF(kernel=ThinPlateSpline()).fit(X,Y)` 报 AttributeError。
- `evaluate()` 原地修改 `dist`，把零距离替换成机器 epsilon。已复现输入 `[0,1]` 被改写；零点值约 -1.8e-30，而非直接使用极限 0。
- 建议：正确使用局部维度；不修改输入，显式处理 r=0 的 r²log(r) 极限。补真实拟合、重复点、只读输入和输入不变性测试。

## K04 [P2] Constant 自相关/交叉相关分支写反

- 位置：`UQPyL/surrogate/gp/kernel/c_kernel_.py::__call__()`。
- 不传第二个矩阵时访问 None.shape，直接 AttributeError。
- 传入 n 行和 m 行矩阵时错误返回 n×n，正确形状应为 n×m。
- 已复现 n=3、m=2 返回 (3,3)，而非 (3,2)。
- 当前不是公开导出的默认核；现有测试只验证模块能够 import。
- 建议：修正矩阵维度分支，补 self/cross、不同样本数和数值常量检查。

## K05 [P1] Matern 可调 nu 的实际值与离散编码混淆

- 位置：`UQPyL/surrogate/gp/kernel/matern_kernel.py` 构造与 Setting 的离散编码交互。
- 候选值为 [0.5,1.5,2.5,inf]，编码区间是 [0,1]，却直接将物理 nu 值作为编码存入。
- 已复现：`Matern(optimize_nu=True)` 默认 nu=1.5，读取 nu 报 IndexError；显式 nu=0.5 被读成 1.5，2.5/inf 也越界。
- 建议：明确数值候选的实际值/编码转换，构造时保留用户指定的物理 nu。补四个候选的构造、读回、优化与 clone 测试。

## K06 [P2] 一般 Matern nu 的零距离与数值稳定性

- 位置：`matern_kernel.py` 的 Bessel 函数分支。
- 先将 scaled_dist 截断到至少 1e-10，随后 `scaled_dist == 0` 判断永远不会成立。
- 已复现 nu=0.01，同一点在 K(X) 对角线上为 1，在 K(X,X) 对角线上约为 0.3705；nu=0.1 时约为 0.9902。
- nu=50/100 时零距离交叉项出现 NaN。常用 0.5/1.5/2.5/inf 专门分支未复现此问题。
- 建议：在替换/运算前记录真实零距离，零点直接赋 1；非零部分采用稳定的 Gamma/Bessel 计算方式，并明确可支持的 nu 范围。

## K07 [P2] 固定长度/尺度参数不能正常初始化

- 位置：GP/Kriging BaseKernel.initialize()、`UQPyL/surrogate/setting.py::expandParam()`。
- `length_attr=None` 或 `theta_attr=None` 将参数登记为常量，但 initialize 仍调用只接受可调参数的 expandParam。
- 已复现 GP RBF/Matern 的固定长度初始化失败；Kriging Guass/Exp/Cubic 的固定 theta 初始化均报 KeyError。RQ 同样使用该基类路径。
- 建议：参数维数规整与“是否参与调优”分开处理。固定参数应保留常量身份，并支持所需的标量/逐维广播。

## K08 [P2] 参数和维度校验不足

- 已复现：GP RBF/Matern/RQ 接受 length_scale=0，随后输出 NaN 矩阵，而非明确错误。
- 待补：长度尺度/alpha/epsilon/theta 的合法域与有限性；nu 的合法值；各向异性参数长度；输入特征数一致性。
- 注意：不要把所有核都要求 K(x,x)=1 或矩阵正定；RBF 插值基函数存在带多项式尾项的条件正定/负定约定，DotProduct 的对角线也不是固定 1。
- Gaussian 的 epsilon 是平方距离前系数还是逆长度，应文档统一，不能不经确认直接改公式。

## 邻接实现问题（独立于核公式）

- `RBF.fitModel()` 将 C_smooth 加到整个增广矩阵的每个元素，包括多项式约束块，而不是仅作用于核块对角线。静态确认与常见平滑正则化形式不同，需按本项目设计单独确认与复现，不纳入上述核公式修复。
- 常用 GP 长度尺度默认下界为 1：这不构成公式错误，但在 [0,1] 输入空间可能限制短尺度拟合，应作为参数默认值讨论。
- 本次更正主记录 R08：KRG 普通 StandardScaler 已对 sigma2 做逆缩放；扩大目标 10 倍，方差比例实测 100。MinMaxScaler 实测仍为 1。GPR 已确认的问题不变。

## 参考依据

- DACE correlation model 定义与逐维乘积：https://www.omicron.dk/dace/dace.pdf
- GP 核定义：https://scikit-learn.org/stable/modules/gaussian_process.html
- RBF 定义、尾项与平滑矩阵：https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html

建议逐项处理顺序：K01 → K02 → K03 → K05 → K06 → K04 → K07 → K08。

## Matern 通用 nu 稳定性后续修复

- 保留固定任意正 nu 与 np.inf；可调 nu 仍只选 0.5/1.5/2.5/inf。
- 积分依据：NIST DLMF 10.32.10，https://dlmf.nist.gov/10.32.E10 。由其推导 Gamma 混合表示；对数坐标下居中于积分峰值，并按曲率缩放，避免大 nu 的峰值过窄。
- 大阶数的归一化使用 Stirling 余项，避免巨大相近数相减；无需新增依赖。
- 独立参考：在已有 base Python 的 mpmath 中以 70 位精度生成 Bessel 公式参考常量，测试自身仍在 conda py312 中运行，且不依赖 mpmath。
- 新增 30 项测试，覆盖 nu=1e-12 到 1e300 的代表值、距离 0 与 1e-12 到 1e3、单独高精度参考值、Gaussian 极限、矩阵半正定及 GP 拟合预测。核专项回归文件当前 68 项通过。
- 这是双精度实现和上述覆盖范围内的验证，不代表穷尽所有浮点输入；大 nu 的积分路径比常用闭式核慢，相同距离会复用积分结果。
- 本轮完整验证：conda py312，**396 passed, 4 warnings，15.90 秒**；4 条仍是既有 FAST 采样频率警告。日志：`/tmp/uqpyl-matern-stability-tests.log`。

## K08 参数校验修复

- 长度尺度、alpha、epsilon 为有限正数；theta、constant、sigma 为有限非负数，保留合法零值；nu 为正数或正无穷。
- 检查长度尺度/theta 与输入特征维数一致；GP self/cross 输入为二维且特征数相同；不扫描样本对或完整距离矩阵。
- 连续调优边界检查参数域、有限性、顺序及对数正值；不要求初值位于调优边界内。离散候选边界是编码坐标，不按物理参数域检查；读取前拒绝非有限/越界坐标。
- 不修改通用 Setting 协议，不新增依赖；调参或直接 Setting 变更后的非法值在初始化/计算前报错，不自动回滚参数。
- 新增 `tests/test_surrogate_kernel_validation.py`，涵盖全部核族、零值、非法构造值、维度、输入形状、调优边界、实际模型更新及可调 nu 非法坐标。
- 验证：新增校验测试 **110 项通过**；conda py312 完整测试 **506 passed, 4 warnings，14.36 秒**。4 条为既有 FAST 采样频率警告。日志：`/tmp/uqpyl-kernel-validation-tests.log`。
