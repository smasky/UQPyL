import sys
sys.path.append('.')

import os
uqPath=os.path.dirname("../UQPyL/")
sys.path.insert(0, uqPath)

import numpy as np
import pandas as pd
from UQPyL.problems import ProblemABC
# ----------------------
# 数据读取
# ----------------------

# 从 TXT 文件读取每日入库流量数据
file_path = "./examples/ob.txt"
data = pd.read_csv(file_path)

# 提取每日入库流量（单位：立方米/秒）
inflow_xfj = data["inflow_XFJ"].values  # 新丰江水库每日入库流量
inflow_fsb = data["inflow_FSB"].values  # 枫树坝水库每日入库流量

# ----------------------
# 调度参数
# ----------------------

delta_t = 86400  # 调度周期（单位：秒，每天）
days = len(data)  # 调度总天数

# 初始库容（单位：立方米）
initial_volume_xfj = np.exp((112.44 + 2059.5) / 456.12)
initial_volume_fsb = np.exp((162.33 - 88.554) / 28.369)

# 安全水位
safe_level_xfj = [114, 115, 116]  # 新丰江（7、8、9 月）
safe_level_fsb = [162, 163, 164]  # 枫树坝（7、8、9 月）

# 死水位
dead_level_xfj = 93
dead_level_fsb = 128

# 出库流量上下限
outflow_limits_xfj = [0, 5950]  # 新丰江
outflow_limits_fsb = [0, 9962]  # 枫树坝

# 适宜生态流量
eco_flow_xfj = [165, 120, 109]  # 新丰江
eco_flow_fsb = [171, 171, 154]  # 枫树坝

# ----------------------
# 水位-库容曲线
# ----------------------

def level_to_volume_xfj(level):
    """新丰江水库水位转库容"""
    return np.exp((level + 2059.5) / 456.12)

def volume_to_level_xfj(volume):
    """新丰江水库库容转水位"""
    return 456.12 * np.log(volume) - 2059.5

def level_to_volume_fsb(level):
    """枫树坝水库水位转库容"""
    return np.exp((level - 88.554) / 28.369)

def volume_to_level_fsb(volume):
    """枫树坝水库库容转水位"""
    return 28.369 * np.log(volume) + 88.554

def calculate_discharge_xfj(X1, X2, E):
    """新丰江出库流量计算"""
    return 450 + (1832.3 * np.log(E) - 7062.8) + X1 * 1267 * (X2 / 24)

def calculate_discharge_fsb(X3, X4, E):
    """枫树坝出库流量计算"""
    return 292 + (607.91 * np.log(E) - 2641.6) + X3 * 1530 * (X4 / 24)

# ----------------------
# 马斯京根法：枫树坝至河源站流量演算
# ----------------------

def calculate_downstream_flow(w1, w2):
    """
    使用马斯京根法计算枫树坝出库流量到达河源站的流量。
    
    参数：
        w1 (float): 枫树坝调度时段初的出库流量 (m³/s)。
        w2 (float): 枫树坝调度时段末的出库流量 (m³/s)。
        
    返回：
        float: 河源站末流量 W4 (m³/s)。
    """
    return 0.2558 * w2 + 0.4884 * w1 + 0.2558 * 693  # 693 为河源站时段初出流量
class YSProblem(ProblemABC):
    def __init__(self, nInput, nOutput, ub, lb, var_type=None, var_set=None, x_labels=None, y_labels=None):
        
        super().__init__(nInput, nOutput, ub, lb, var_type, var_set, x_labels, y_labels)

    def single_evaluate(self, x):
        # 解析决策变量
        # 决策变量
        X1, X2, X3, X4 = x
        
        # 初始化水位和库容
        volume_xfj = initial_volume_xfj
        volume_fsb = initial_volume_fsb
        
        # 目标函数
        flood_risk = 0.0
        total_spill = 0.0
        ecological_change = 0.0

        # 记录约束值
        water_balance_violation = 0.0
        max_downstream_violation = 0.0
        
        vol_constrain_xfj = 0.0
        vol_constrain_fsb = 0.0

        # 逐日迭代计算
        for day in range(days):
            # 当前入库流量
            inflow_day_xfj = inflow_xfj[day] * delta_t
            inflow_day_fsb = inflow_fsb[day] * delta_t

            # 当前水位
            level_xfj = volume_to_level_xfj(volume_xfj)
            level_fsb = volume_to_level_fsb(volume_fsb)

            # 出库流量
            outflow_day_xfj = max(0, calculate_discharge_xfj(X1, X2, level_xfj))  # 保证非负
            outflow_day_fsb_start = max(0, calculate_discharge_fsb(X3, X4, level_fsb))  # 保证非负
            outflow_day_fsb_end = outflow_day_fsb_start

            # 更新库容（水量平衡）
            next_volume_xfj = max(0, volume_xfj + (inflow_day_xfj - outflow_day_xfj))  # 保证非负
            next_volume_fsb = max(0, volume_fsb + (inflow_day_fsb - outflow_day_fsb_start))  # 保证非负

            # 更新水位
            next_level_xfj = volume_to_level_xfj(next_volume_xfj)
            next_level_fsb = volume_to_level_fsb(next_volume_fsb)

            # 河源站下游流量
            downstream_flow = calculate_downstream_flow(outflow_day_fsb_start, outflow_day_fsb_end)

            # 检查下游控制断面流量 TODO
            max_downstream_violation += abs(outflow_day_xfj + downstream_flow - 7900)
            # max_downstream_violation = max(
            #     max_downstream_violation,
            #     outflow_day_xfj + downstream_flow - 7900
            # )
            
            # 检查水库运行水位约束
            if not (dead_level_xfj <= next_level_xfj <= safe_level_xfj[min(day // 31, 2)]):
                flood_risk += 1
            if not (dead_level_fsb <= next_level_fsb <= safe_level_fsb[min(day // 31, 2)]):
                flood_risk += 1

            # 累积水量平衡约束违背值
            bal_xfj = abs(volume_xfj - next_volume_xfj - (inflow_day_xfj - outflow_day_xfj))
            if bal_xfj > abs(volume_xfj - next_volume_xfj)*0.1:
                water_balance_violation += bal_xfj
            
            bal_fsb = abs(volume_fsb - next_volume_fsb - (inflow_day_fsb - outflow_day_fsb_start))
            if bal_fsb > abs(volume_fsb - next_volume_fsb)*0.1:
                water_balance_violation += bal_fsb

            # 目标函数计算
            total_spill += max(0, outflow_day_xfj - eco_flow_xfj[min(day // 31, 2)])
            ecological_change += abs(outflow_day_xfj - eco_flow_xfj[min(day // 31, 2)]) / eco_flow_xfj[min(day // 31, 2)]

            # 更新库容
            volume_xfj = next_volume_xfj
            volume_fsb = next_volume_fsb

            #TODO 库容约束
            vol_constrain_xfj += max(0, initial_volume_xfj - volume_xfj)
            vol_constrain_fsb += max(0, initial_volume_fsb - volume_fsb)
        
        # 设置目标函数
        obj = np.array([
            flood_risk / days * 100,  # 防洪风险
            total_spill,  # 弃水量
            ecological_change / days  # 平均生态改变度
        ])

        # 设置约束（总违背值）
        con = np.array([
            max(0, water_balance_violation),       # 水量平衡约束，确保非负
            max(0, max_downstream_violation),      # 下游断面安全流量约束，确保非负
            max(0, vol_constrain_xfj),  # 最终库容与初始库容一致性，确保非负
            max(0, vol_constrain_fsb)   # 最终库容与初始库容一致性，确保非负
        ])

        return obj, con


    def evaluate(self, X):
        
        objs = np.zeros((X.shape[0], self.nOutput))
        cons = np.zeros((X.shape[0], 4))
        
        for i, x in enumerate(X):
            objs[i], cons[i] = self.single_evaluate(x)
        
        res = {}
        res['objs'] = objs
        res['cons'] = cons
        
        return res

from UQPyL.optimization import NSGAII

lb = np.array([0, 0, 0, 0])
ub = np.array([3, 24, 6, 24])
v_type = [1, 0, 1, 0]
ysProblem = YSProblem(4, 3, ub = ub, lb=lb, var_type=v_type)

# nsgaii = NSGAII()
# res = nsgaii.run(ysProblem)

#单次跑
x = np.array([2, 20, 3, 15])
obj, con = ysProblem.single_evaluate(x)
print("obj:", obj)
print("con:", con)