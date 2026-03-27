import arviz as az
import xarray as xr
import numpy as np

path = "D:/UQ/Result/Data/AMH_Problem_D5_M1_2.nc"

post_file = path          # ← 最重要！包含 decs 样本的那个

post_ds = xr.open_dataset(post_file, group ="posterior")

posterior_ds = post_ds

# 2. 转 ArviZ 格式
posterior_ds = posterior_ds.rename_vars({"decs": "theta"})
posterior_ds = posterior_ds.rename_dims({"decsDim": "theta_dim"})

inf_data = az.InferenceData(posterior=posterior_ds)

# 3. 计算 ESS 和 R-hat
ess  = az.ess(inf_data, var_names="theta")
rhat = az.rhat(inf_data, var_names="theta")

print("\n=== 原始 ESS（每维参数） ===")
print(ess)


# === 关键修复：正确取平均值 ===
mean_ess = float(ess["theta"].mean())           # 先取变量，再 .mean() → 标量

print("\n=== Mean ESS (Banana) ===")
print(f"Mean ESS (Banana): {mean_ess:.0f}   ← 直接抄进 Table 3！")

print("\n=== R-hat ===")
print(rhat)


# 如果你只想要干净的 10 维均值向量（直接抄进论文/Table）
post_mean = inf_data.posterior["theta"].mean(dim=["chain", "draw"]).values
print("\n=== Posterior Mean（10维向量，干净数值） ===")
print(np.round(post_mean, 4))   # 保留4位小数，便于复制