# Problem / ModelProblem 专项复核（2026-10-03，只检测）

后续状态：PR01–PR03已修复，见[修复验证](1003-problem-fixes.md)。下文保留审查时证据。

本轮确认三项新的实现问题，尚未修改生产代码或pytest。conda py312既有Problem/ModelProblem专项 **168 passed，5.17秒，-W error**；另执行62条公共接口/独立参照记录，其中包含缺陷复现，不称为62项全部通过。3042为上一轮全量基线，本轮未重复全量、提交/推送/重建wheel。

## 确认问题

### PR01：列向量边界通过校验后，沿错误轴广播（P1）

`Space._check_bound`只检查ravel后的元素数，`_set_ub_lb`却通过atleast_2d保留原始二维形状。两变量边界下界[0,10]、上界[1,20]，传成(2,1)列向量会被接受。

单位坐标中点应每行得到[0.5,15]；实际：

- 一个样本竟返回两行 `[[0.5,0.5],[15,15]]`。
- 两个样本返回同一错误矩阵，形状看似正常但参数轴错位。
- 三个样本发生广播异常。
- 公共LHS两样本返回 `[[0.4225,0.6840],[17.7887,10.8049]]`，第一行第二维及第二行第一维均越过各自参数边界，无warning。

同数值一维向量和(1,2)行向量的控制均正确。建议在入口将支持的标量/向量/行列向量统一为(1,nInput)，对其他不明确二维布局拒绝；不能只补下游shape检查，因为样本数恰等于维数时shape仍匹配。

### PR02：singleFunc保存输出引用，复用缓冲区时全部行变成末行（P1）

装饰器循环中`np.atleast_1d`不一定复制数据，随后最后才vstack。用户单点评价为了减少分配复用一个长度2的ndarray，逐点写入[x,2*x]再返回；通过Problem.evaluate评价[[1],[2],[3]]。

理论结果 `[[1,2],[2,4],[3,6]]`；实际 `[[3,6],[3,6],[3,6]]`，无异常或warning。这不是原地修改输入导致的问题，而是适配器没有在每次返回时保留结果快照；同样适用于返回同一缓冲区切片。建议在每次单点评价返回时复制，再堆叠，并补 scalar/vector/view 的行对应回归。

### PR03：Space极大有限边界下单位转换溢出（P2）

边界[-1e308,1e308]均有限。unit_to_space直接形成`ub-lb`，结果为inf，再乘单位值和clip：

- 单位点[0,.25,.5,.75,1]本应[-1e308,-5e307,0,5e307,1e308]。
- 实际[nan,1e308,1e308,1e308,1e308]。
- 反向space_to_unit对上述真实点返回[0,0,0,0,nan]，同样错误。

只出现NumPy通用溢出/无效运算warning，结果仍返回。建议正向采用安全插值，反向使用安全范围比值，补端点/中点/往返/原数组不变与普通尺度保持。此处是公共Space转换，与已修复DeltaTest内部安全距离换算是不同入口，不能把前者视为后者修复失败。

## 正常对照

- 27组Problem：1/2/7行，完整/仅目标/仅约束，min/max/混合方向。独立解析目标和约束逐行一致；只调用请求的回调；公开Eval保留真实目标方向，未擅自翻转最大化值。
- 12组ModelProblem：1/2/7行×完整/objs/cons/sims。模型每次恰调用一次，目标/约束按请求调用；仿真数组、派生目标/约束逐行对应，输出块裁剪正确。
- 3组混合域：连续/整数/离散小数/固定值的独立解码、真实值往返和单位坐标规范化正确。
- 普通行向量/一维边界与1/2/3行样本、LHS对照正常。
- ModelProblem仅在已掩码的观测位置允许NaN，未掩码NaN明确拒绝。
- 错误输出行数、列数、字符串类型均被拒绝。

这些不覆盖用户回调所有副作用、自定义Space/外部模型适配器、全部大规模模型或公共存储/可视化模块。没有把缺少新专项验证说成这些模块完全没有测试。

## 证据与下一步

优先PR01 → PR02 → PR03，先处理普通范围下可能静默错算的两个问题，再补极端范围。

- [审计脚本](verification/check_problem_review.py)
- [62条原始记录](verification/1003-problem-review.json)（非有限值以字符串保留）
- [审计日志](verification/1003-problem-review.txt)
- [既有专项](verification/1003-problem-review-pytest.txt)

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run --no-capture-output -n py312 python agent/verification/check_problem_review.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 conda run --no-capture-output -n py312 pytest -q tests/test_problem*.py tests/test_model_problem.py -W error --basetemp=.cache/pytest/problem-review-final
```
