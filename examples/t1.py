import pymc as pm
import arviz as az
from multiprocessing import freeze_support   # ← 新增这行！

# ==================== 【你的 Banana log density】 ====================
# 必须和你的自定义 MH 里 logp **一模一样**！（系数不对 ESS 会差）
def banana_logp(theta):
    x = theta[0]
    y = theta[1]
    # ← 这里改成你程序里**完全一致**的公式
    return -0.5 * x**2 - 0.5 * ((y - x**2) ** 2)   # 标准 Banana

# =====================================================================

if __name__ == "__main__":                    # ← 关键保护（必须加）
    freeze_support()                          # ← 解决 Windows multiprocessing 报错

    with pm.Model() as model:
        theta = pm.Flat("theta", shape=2)
        pm.Potential("banana", banana_logp(theta))

        # ==================== 和你完全一样的采样设置 ====================
        idata = pm.sample(
            draws=10000,                  # 每链后验样本数（和你一模一样）
            tune=1000,                    # 调优步
            chains=4,
            step=pm.Metropolis(),         # 标准随机游走 MH
            random_seed=42,
            return_inferencedata=True,
            progressbar=True
        )

    # 保存成 nc（和你之前一样）
    # idata.to_netcdf("pymc_banana_MH.nc")

    # ==================== 计算 ESS（和你之前代码 100% 一样） ====================
    ess  = az.ess(idata, var_names="theta")
    rhat = az.rhat(idata, var_names="theta")

    print("\n=== PyMC MH 的原始 ESS ===")
    print(ess)

    mean_ess = float(ess["theta"].mean())
    print(f"\n=== Mean ESS (PyMC MH Banana): {mean_ess:.0f}   ← 直接和你 1019 对比！")

    print("\n=== R-hat ===")
    print(rhat)
