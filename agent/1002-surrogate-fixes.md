# 2026-10-02 替代模型 SR01–SR07 修复

后续用户要求尽量使用 warning：RBF 可恢复奇异系统已调整为告警后约束最小二乘近似，附残差诊断，见 [最新约定与 2594 项验证](1002-rbf-warning.md)。下方“奇异时报错”及 2579 项计数为首轮修复阶段记录；其他六项保持。

按用户要求依次处理 [七项审查问题](1002-surrogate-module-review.md)。七项代码修复与新增 89 项回归已落实；原 215 组独立审查已全部正常，最终 py312 全量 **2579 passed，53.22 秒，`-W error` 零未捕获警告**。

## 实现与验收

| 编号 | 修复 | 验证 |
|---|---|---|
| SR01 | RBF 对完整 LU 因子作三角求解，不再分别取伪逆截断趋势约束；奇异系统抛 LinAlgError 并提示重复输入/趋势/平滑设置 | Cubic 线性函数 X 范围 0～1000/10000 的误差从约 0.0209/0.0733 降至约 1e-15；非线性与 SciPy 自然样条一致，原五种核平滑参照保持 |
| SR02 | Lasso 用 X/Y 的共同浮点 dtype 建立独立工作副本；已有纯 float32 路径保持 | 整数/混合类型、两种回归模型、截距开关、只读数组与浮点参照一致，原数据不变 |
| SR03 | nu 默认搜索上界从 1000 改为 1；调用原生训练前检查有效 nu 和自定义边界 | 三个种子的真实 GA/AutoTuner 搜索均完成；非法 nu 未进入后端 |
| SR04 | R²/NSE 在按列二进制尺度上计算平方和，合并时保留输出权重；AutoTuner 用原始值判定是否恒定 | 正负 1e-200/1e160 缩放下手算 0.968 保持；混合量纲/大常数列对照保持；二次函数可正常调参并得 R²=1 |
| SR05 | StandardScaler 使用安全中心化和样本标准差，再恢复单位；仿射 Scaler 分开保存原始中心与目标中心 | ddof=1 手算、极端列与非默认目标中心的常数列往返通过，±1e308 数据可标准化/还原而不因中间减法溢出 |
| SR06 | returnStd 直接逆变换标准差；方差用尾数/指数乘法，避免先计算 scale² | GPR/KRG/容器极小与极大单位的可表示标准差正确；原单位方差真正越界时 warning，零/inf 的意义明确；可表示的方差不会因 scale² 中间量越界而丢失 |
| SR07 | GPR 原始/预处理拟合入口在 Scaler/搜索/后端之前检查数据；C 与搜索边界分别校验，候选目标计算前再次检查 C | 三个拟合入口的 NaN/inf 数据、负/NaN/inf C 被提前拒绝并使旧状态失效；C=0 的合法拟合保留 |

生产修改集中于 surrogate：新增小型 `_numeric.py` 共用二进制中心化/平方和表示，触达 base、metric、scaler、auto_tuner、GPR、RBF、Lasso 与 SVR。没有变更原生扩展，不需要重建本轮 Cython/C++ 代码。

## 明确保留的行为

- 单模型仍只接收单输出，多输出使用 MultiSurrogate；各列独立建模，不增加输出间协方差。
- R²/NSE 对真正恒定目标保留原 NumPy warning/errstate 约定：预测完全匹配为 NaN，否则为负无穷。AutoTuner 仍明确拒绝未定义的常数验证目标；普通非恒定目标不再因原始平方和超范围被拒绝。
- 无法表示的原单位方差发出 RuntimeWarning，浮点下溢返回零、上溢返回 inf；可改请求标准差。未将过大方差裁剪为看似正常的有限值。自定义 Scaler 的标准差与方差请求分别要求对应的逆变换方法。
- RBF 奇异系统会明确失败，正平滑可处理已有重复观测案例；不保证所有病态核矩阵都能可靠插值，亦未静默增加用户未设置的平滑。
- GPR 输入有限性检查只用于 GPR，不改变 MARS 已有的标记缺失输入协议。

## 回归与复核

新增 [求解/类型/搜索测试](../tests/test_surrogate_solver_boundaries.py) **27 项**及 [数值范围测试](../tests/test_surrogate_numeric_ranges.py) **62 项**，合计 **89 项**。测试使用独立自然样条、手算 R²/样本标准差、输出单位变换、真实搜索与后端前调用检查；不是只验证返回形状。

数值测试首次额外发现非默认目标中心下常数列丢失：极小常数逆变换为零、极大常数未映射到目标中心；已修复仿射运算顺序并补充 MinMaxScaler 常数列对照。首次全量另有 4 项因恒定目标被改成异常而失败，已恢复原 warning/errstate 行为，原测试没有删除或放宽。

| 验证 | 结果 | 日志 |
|---|---|---|
| SR01–SR03 与原相关测试 | 156 passed，1.01 秒 | [前三项](verification/1002-surrogate-sr01-sr03.txt) |
| 既有替代模型专项加前三项新测试 | 912 passed，4.22 秒 | [阶段专项](verification/1002-surrogate-fixes-targeted-initial.txt) |
| 首次数值范围测试 | 2 failed / 57 passed，0.86 秒 | [额外常数列复现](verification/1002-surrogate-sr04-sr07.txt) |
| 数值范围及 Scaler 回归 | 104 passed，0.89 秒 | [范围最终专项](verification/1002-surrogate-sr04-sr07-final.txt) |
| 首次全量 | 4 failed / 2575 passed，53.30 秒 | [常数评分协议检查](verification/1002-surrogate-fixes-full-initial.txt) |
| 恢复常数评分协议后 | 72 passed，1.15 秒 | [评分/绘图/范围](verification/1002-surrogate-constant-protocol.txt) |
| 最终全量 | 2579 passed，53.22 秒 | [全量](verification/1002-surrogate-fixes-full.txt) |

所有 pytest 命令使用 conda py312、`-W error`，数值进程限制 BLAS/OMP 单线程。预期范围 warning 由测试明确捕获。相关 12 个 Python 文件 Ruff 静态/格式检查及差异检查通过。中英文 API、测试导航和交接/TODO 同步。

独立 [审查脚本](verification/review_surrogate_module.py) 保留原 215 组案例与独立参照，新输出 [after-fixes JSON](verification/1002-surrogate-module-after-fixes.json) / [摘要](verification/1002-surrogate-module-after-fixes.txt)：**215 正常、0 异常**。原 [修复前 JSON](verification/1002-surrogate-module-review.json) 的 34 异常/181 正常保持不变。

未提交/推送/发布，未重建 Python 3.14 wheel。极小尺度校准 RMSE、SA09/SA11、MARS 高阶敏感性贡献局限及 C15/C17/C20 暂缓项不属于本轮七项，状态保持。
