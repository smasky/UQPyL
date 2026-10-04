# 2026-09-30 DeltaTest 距离缩放与带符号归一化

本轮按用户认可的方案修改 DeltaTest，解决输入单位改变近邻几何和负总分归一化反转排名的问题。MARS 验证阈值保持用户最新约定：R²<0.8 发出 warning 并继续返回结果。本轮未提交/推送。

## 实现

[DeltaTest 源码](../UQPyL/analysis/methods/delta.py) 的三个公共入口统一调用 `_scaleInputs`，对完整输入缩放一次，再删除变量或选取子集：

- 连续/整数维度按声明上下界 `(X-lb)/(ub-lb)` 计算坐标；数值离散变量按实际 `varSet` 数值范围缩放，避免使用离散编码的 lb/ub 误当物理范围。
- 固定维度映射为零，固定样本必须匹配声明值；样本中不变的维度在删变量敏感性里贡献为零。这也避免删去常数坐标后等距近邻重新选择造成虚假增量。
- 参数上下界须有限且有序；不裁剪越界值，避免将不同样本压到同一个边界。原始 X/Y 和参数边界不修改，评价和结果保存保持真实坐标。
- Delta 半均方近邻差公式、自身排除和邻居数约束保持前轮修复后的定义；原始 S1 仍为删变量 Delta 减完整 Delta，允许负值。
- `S1_norm = S1 / sum(abs(S1))`。实现先除以最大绝对分数，再做绝对值求和，避免归一化分母溢出。保留符号和原始排序，正负抵消也不清零；全零行仍为零。非零行的绝对值之和为 1，带符号和不一定为 1。
- 非恒定输出若没有正分数，发出 RuntimeWarning 并继续返回结果；恒定输出返回零且不触发这一提示。提示不是拟合 R² 门槛，也不是统计显著性判断。

这是开发阶段行为变更：默认距离从真实值变为参数范围缩放值，分数及子集搜索结果可能改变；负分数不会被取绝对值变成正的重要性。中英文 API、测试导航同步。

## 独立验证

新增 [test_analysis_delta_scaling.py](../tests/test_analysis_delta_scaling.py) 共 17 项：三种输入单位/平移的独立逐对距离参照、负总分/精确抵消/全负分数、固定和样本常数维度、非正分数告警及常数输出、暴力/EA 子集目标同一尺度、单位采样元数据与真实评价、数值离散实际范围和空样本错误。

EA 验证使用测试中的穷举 GA 替身逐一评价所有二元子集目标，与独立成对距离公式比较，既验证嵌入目标又避免将随机搜索结果当成数学参照。生产 GA 未替换。

修复前新测试最初 14 项中 **11 failed / 3 passed**：[首次复现](verification/0930-delta-before.txt)。修复后既有/新增 Delta 及科学参照专项 **43 passed，2.56 秒，零未处理警告**：[专项日志](verification/0930-delta-targeted.txt)。离散删除后有并列近邻时，独立 argsort 与 KDTree 可以选择不同的同距行；测试使用组内相同输出使参照不依赖任意 tie 选择，未放宽数值阈值。[原参照失败记录](verification/0930-delta-tied-reference.txt) 保留用于说明这一修正。

[验证脚本](verification/check_delta_scaling.py)、[JSON](verification/0930-delta-scaling.json)、[日志](verification/0930-delta-scaling.txt)：5 种子×3 种输入单位×2 种输出单位，共 **30 组**。模型为三个独立均匀输入下的 `Y=x0+2*x1`，每组 512 行；输入第一轴乘以 1/0.001/1000 并加平移，输出乘以 1/1e-6。所有分数匹配独立成对距离定义，原始/归一化第一名均为 x1；最大原始尺度复原误差 **4.44e-16**。

seed=17 的三种输入单位得到完全相同的分数：原始 `[0.2983373,1.3324615,-0.0224675]`，归一化 `[0.1804533,0.8059570,-0.0135897]`。此前同一输入放大 1000 倍产生负总分和排名翻转的反例已消除。

## 全量回归中发现的 MARS 初始化问题

第一次全量 **2 failed / 2053 passed**，两项 MARS 交互回归在原生初始化 `np.dot(weight.Q_t, wy)` 触发 invalid 警告：[失败日志](verification/0930-delta-full-before-initialization.txt)。此时 k=0，没有已填充的 Q 行，Q_t 来自 `np.empty`。因此读取的是未初始化工作区，表现依赖先前的内存内容；不是 Delta 分数公式变化造成 MARS 数学行为变化。

新增 [确定性回归](../tests/test_surrogate_mars_initialization.py) 将未使用的 Q_t 填为 inf，配合含零的输出可稳定触发非法投影，修复前 **1 failed**：[日志](verification/0930-mars-initialization-before.txt)。最小修复将初始 theta 建为零向量；后续只对已激活的 Q 行投影保持不变。还手算更新后 SSE=8.75，验证不是单纯屏蔽 warning。

修改 [原生源码](../UQPyL/surrogate/mars/core/_knot_search.pyx)，重建 py312 扩展，构建结果见 [日志](verification/0930-delta-native-build.txt)。保留此处额外修复以消除新测试序列触发的未初始化内存读取；不更改 MARS 0.8/warning 约定。

## 范围和限制

范围缩放给各输入采用相同的参数范围尺度，不保证高维、噪声、相关输入下的排名精度。浮点单位转换和等距近邻并列仍可能影响选择；无序类别没有新增专门距离模型。原始输出差平方仍受浮点上溢/下溢限制，1e-6 验证不能推广为任意量级。

重编译后的初始化/MARS/Delta 相关专项 **36 passed，12.01 秒**，见 [日志](verification/0930-delta-native-targeted.txt)；最终 py312 全量 **2056 passed，41.30 秒，零警告**，见 [日志](verification/0930-delta-full.txt)。触达 Python 文件 Ruff 检查/格式及 `git diff --check` 通过。

本轮未重建 3.14 wheel；此前 wheel 结果不能作为这些新修改的安装包验证。
