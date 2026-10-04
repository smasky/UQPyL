# ES / IES 论文与独立实现对照（2026-10-02）

本轮完成修正原型及独立对照，**尚未替换正式 ES / IES**。此前“45 组矩阵参照正确”只说明实现符合原有更新式，不能证明该式正确保留后验不确定性；将此问题仅归为方法语义过于宽松。

## 参考与选择

- [Evensen 2003](https://www.ecmwf.int/sites/default/files/elibrary/2003/74424-ensemble-kalman-filter-theoretical-formulation-and-practical-implementation_0.pdf)：直接删除观测扰动会低估分析集合方差；确定性方法需要适当的平方根变换。
- [Raanes、Stordal、Evensen 2019](https://npg.copernicus.org/articles/26/325/2019/) 与 [Evensen 等 2019](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2019.00047/full)：随机迭代平滑器的先验约束、集合子空间实现及非线性近似边界。
- [Chen、Oliver 2013](https://link.springer.com/article/10.1007/s10596-013-9351-5)：LM-EnRML；不能将现有反复增益更新直接宣称为完整 LM-EnRML。
- [Equinor iterative_ensemble_smoother](https://github.com/equinor/iterative_ensemble_smoother)：找到可运行 SIES 参照。本轮固定使用 **0.2.7**，并非声称当前最新版的接口相同。该项目 GPL-3.0，仅隔离安装用于对照，不复制实现、不加入 UQPyL 生产依赖。
- [ERT](https://github.com/equinor/ert) 及 [Luo IES](https://github.com/lanhill/Iterative-Ensemble-Smoother) 也是相关实现；后者为另一种迭代方案，不能直接互换方法名。

ES-MDA 与 SIES/EnRML 不同，本轮没有用 ES-MDA 替换 IES。

## 原型与结果

[独立原型](verification/prototype_es_ies_reference.py) 由方程实现两条路径：

1. ES：均值使用 Kalman 增益，中心化成员使用对称平方根变换，使线性案例的样本均值和协方差符合以初始集合矩为先验的解析更新。
2. IES：固定原始集合、先验协方差及每个成员的扰动观测，以集合回归近似模型导数，进行带先验约束的 Gauss–Newton RML 更新。原型是小规模稠密实现，不是完整的自适应 LM 接受/拒绝算法。

py312、单线程 BLAS/OMP、`-W error` 运行通过，保存 **85 条记录**（不是 85 个独立案例）：

- 3 个随机种子 × 4 组维度/集合大小 × 5 次线性更新，含参数维度 20、成员 12 的秩亏先验；ES 同时与解析均值/协方差比较。
- 3 个种子的单参数非线性模型 `y=x+0.3*x^3`，各 8 次半步更新。
- 原标量反例：先验成员 `[-1,0,1]`、观测 1、噪声方差 1。原 ES 均值 0.5、方差 0.25；修正原型均值 0.5、方差 0.4999999999999999，理论方差 0.5。

IES 与固定版本 Equinor SIES 使用完全相同的扰动观测，逐成员最大差 **3.55e-15**；ES 解析协方差最大误差 **3.11e-15**。线性 IES 全步更新第一次即得到该扰动观测下的解，后续四次保持，不再因为重复使用同一数据而继续收缩。

数据：[JSON](verification/1002-es-ies-reference-prototype.json)、[运行日志](verification/1002-es-ies-reference-prototype.txt)、[安装日志](verification/1002-es-ies-reference-install.txt)。脚本头部记录隔离依赖安装方法。

## 尚未覆盖及正式接入工作

- 本轮观测协方差均正定，未验证零噪声、半正定奇异噪声及不一致硬观测。旧版零噪声薄 SVD 路径的内存保护需要保留或提供等效实现。
- 对照均无边界裁剪；裁剪会改变集合矩，不能继续宣称裁剪后满足无约束高斯公式。
- 非线性只验证一个单参数模型。与参考程序一致不等于非线性真实后验正确，也不证明高维非线性回归方案与子空间实现普遍等价。
- 随机 IES 有有限集合误差，不应要求其单次样本协方差严格等于解析总体后验协方差。
- 正式接入需要明确随机种子/固定扰动、步长、收敛及诊断；现有 `lam` 是观测空间附加 ridge，不能不改语义就称作论文的 LM 阻尼。
- 平方根原型构造成员数平方矩阵；大规模接入应考虑低秩实现。本轮不恢复已暂缓的一般大观测协方差优化任务。

结论：已验证可行的数学修正路径。正式模块仍有上述问题，不能标为已修复。此次未改生产或 pytest、未重复全量；此前 2671 passed 是上轮基线。未提交、推送或重建 wheel。
