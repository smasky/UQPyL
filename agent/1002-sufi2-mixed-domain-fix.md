# SUFI2 内部混合域修复（2026-10-02）

CAL01 已实现：内部临时 Problem 保留 varType/varSet，LHS 通过正式 unit_to_space 解码出真实合法值；整数遵守 ceil(lb)～floor(ub)，离散值只来自允许集合。

首轮采用完整 varSet，编码边界不裁剪真实离散选项。后续与原精英 min/max 收缩规则一致：保留当前选项中位于精英真实数值包络内的值、保持原选项顺序；可保留本轮精英未实际出现的中间合法选项。退化为单值时继续正常运行。原问题与选项列表不被修改。

每轮 history 和最终 diagnostics 新增 updatedVarSet；updatedLb/updatedUb 仍是精英真实值范围。外部给定 X 用原问题 space_to_unit 做合法性检查，但传给模拟器的仍是原真实值，不做取整/离散二次映射。非法配置/非法样本在模拟前拒绝，未以 warning 继续评价无效参数。

## 验证

- 新增 9 项 `tests/test_sufi2_mixed_domains.py`：三个种子×单精英/多精英的四轮混合采样；无序离散选项、负数/小数且超出编码边界；非整数整数边界；单值域；中间选项保留；原问题保护；外部非法整数/离散值模拟前拒绝。
- 校准专项 **268 passed，2.60 秒，`-W error`**，见 `verification/1002-sufi2-domain-targeted.txt`。
- 原 63 组审查重新运行，历史结果保留，新文件 `verification/1002-calibration-science-after-sufi2.json/.txt`。原两例各 24 个非法样本均变成 **0**，外部合法输入、独立更新和正常分位数参照保持。RMSE 缺陷仍如实复现，CAL02 尚未处理。
- SQLite 手动往返检查通过：updatedVarSet 和三轮历史离散集合完整保留。
- Ruff、格式、差异检查通过，中英文校准 API 与测试导航同步。最终 py312 全量 **2645 passed，86.01 秒，`-W error` 零未捕获警告**，见 `verification/1002-sufi2-domain-full.txt`。

本轮只处理 SUFI2，不改 ES/IES 后验语义，不把连续与离散参数混合情况下的筛选算法宣称为最优性保证。未提交/推送或重建 wheel；原其他待处理/暂缓项保持。
