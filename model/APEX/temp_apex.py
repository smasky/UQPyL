from apex import APEX
import numpy as np

cfgPath = "D:/UQPyL/model/APEX/cfg.yaml"

apex = APEX(cfgPath)

from UQPyL.doe import LHS

lhs = LHS()
X = lhs.sample(apex, 5000)
# X = np.array([41.8007,268.02091,46, 7.48015,0.55689,0.32735,0.8633,
#               0.00258,0.56848,0.53609,0.24591,5.37041,0.2058,4.86825,
#               1.39641,1.65329,0.1022,2.78815,0.09235,0.09426,0.43786,
#               0.95824,0.80292,0.15364,0.74659,1.02759,1.70902,1.85659,
#               0.70676,19.63727,0.02913]).reshape(1, -1)

objs = apex.evaluate(X)

# sim = records[0]['cache']['s1']['sim']   
# obs = records[0]['cache']['s1']['obs']

# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates

# 2013-01 到 2021-12 共 108 个月
# dates = pd.date_range(start="2013-01-01", periods=len(sim), freq="MS")  # MS=每月月初

# fig, ax = plt.subplots(figsize=(10, 4))

# ax.plot(dates, sim, label='sim')
# ax.plot(dates, obs, label='obs')

# ax.xaxis.set_major_locator(mdates.YearLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# ax.xaxis.set_minor_locator(mdates.MonthLocator())

# ax.legend()
# ax.grid(True, alpha=0.3)
# fig.autofmt_xdate()  # 自动旋转日期标签防重叠
# plt.show()
