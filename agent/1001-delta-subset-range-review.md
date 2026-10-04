# 2026-10-01 DeltaTest 组合搜索输出范围复核

用户询问还有哪些问题。本轮只读复核和保存证据，没有修改生产代码/测试；不要将本轮问题与已完成的 `analyze` 输出归一化修复混淆。

## 新确认的问题

DeltaTest 的 `findCombVio` / `findCombEA` 仍直接将原始 Y 交给 `_cal_delta`，组合目标中的输出差平方没有安全缩放。此前修复只为 `analyze` 增加了每输出预缩放及尺度恢复前归一化，两个组合搜索入口只同步了等距近邻规则。

实测模型 `Y=3*x0+1`，两个独立 U[0,1] 输入、128 行 LHS classic、seed=17、默认 k=2。三个非空组合依次是只保留活跃 x0、只保留无关 x1、保留两者。

| 输出倍率 | 三个非空组合的目标 | findCombVio 结果 |
|---|---|---|
| 1 | `[0.0003021995, 0.7598890, 0.009802545]` | `['x_1']`，正确活跃变量 |
| 1e-200 | `[0, 0, 0]` | `['x_2']`，错误选中无关变量，没有警告 |
| 1e200 | `[inf, inf, inf]` | `[]`，空组合；只有 NumPy 平方 overflow warning |

极小输出使全部非空目标下溢为零，`argmin` 按枚举顺序选出第一个非空组合。极大输出使所有目标连同定义为 inf 的空组合均为 inf，`argmin` 选中了空组合。当前代码中 EA 的目标回调走同一个原始平方路径，故目标同样会退化；本轮没有实际运行随机 GA，不把目标退化描述为某个已实测的 GA 最终选择。

这已是明确的数值边界问题，而不仅是“极端尺度未覆盖”。原始目标平方不可表示本身有浮点限制，但不能因此继续输出误导性的最优组合。后续需明确安全的比较尺度，以及 EA 结果/历史中原始目标的范围和诊断协议；多输出情况下不得逐列任意标准化而静默改变各输出的相对权重。未在本轮实施。

证据：[三组输出对照](verification/1001-delta-subset-output-range.json)。复现：

```python
from UQPyL.analysis import DeltaTest
from UQPyL.doe import LHS
from UQPyL.problem import Problem

problem = Problem(nInput=2, nObj=1, lb=0, ub=1,
                  objFunc=lambda x: (3*x[:, 0]+1)[:, None])
x = LHS('classic').sample(problem, 128, seed=17)
y = problem.evaluate(x).objs
method = DeltaTest(verboseFlag=False, logFlag=False, saveFlag=False)
for factor in (1, 1e-200, 1e200):
    print(factor, method.findCombVio(problem, x, y*factor))
```

## 另外两项既有待处理内容

- Morris 当前为物理步长 `ΔY/ΔX`，换输入单位会改变跨变量排名。此行为符合已实现的物理斜率定义，但若希望比较整个参数范围的影响，需要明确并增加单位区间/无量纲模式；不应把原有结果直接定性为算术错误。
- MARS 高阶交互可被低估，甚至高留出 R² 下仍有明显偏差。这是容量/自适应搜索和 GCV 筛选指标的局限。用户已确认默认二阶，后续应侧重分数随划分/样本的稳定性和实际模型的独立参照，不能只凭更高 R² 改默认或宣称贡献准确。见 [阶数对照](0930-mars-degree-accuracy.md)。

当前 2099 项通过的结果属于前轮生产修复验证；本轮没有新增 pytest、重跑全量或重建 wheel。没有提交/推送，既有暂缓项继续保持暂缓。
