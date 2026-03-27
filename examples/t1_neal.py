import pymc as pm
import arviz as az
import pytensor.tensor as pt
import numpy as np
from multiprocessing import freeze_support

# ==================== 【5D 中等相关 Gaussian log density】 ====================
# 和你自定义的 gauss5d_objFunc 完全一致（potential = -logp）

# 预先计算协方差逆矩阵（常量，只算一次）
_mu = np.array([1.0, -1.0, 0.5, 2.0, -0.5])
_sigma = np.array([1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0])
_corr = np.array([
    [1.0,  0.5,  0.2,  0.0,  0.0],
    [0.5,  1.0,  0.5,  0.2,  0.0],
    [0.2,  0.5,  1.0,  0.5,  0.2],
    [0.0,  0.2,  0.5,  1.0,  0.5],
    [0.0,  0.0,  0.2,  0.5,  1.0],
])
_cov = np.diag(_sigma) @ _corr @ np.diag(_sigma)
_cov_inv = np.linalg.inv(_cov)
_log_det = np.log(np.linalg.det(_cov))   # 常数项（可选，不影响采样）

def gauss5d_logp(theta):
    """
    5 维中等相关 Gaussian 的 logp
    theta: shape (5,)
    返回 logp = -0.5 * (x-μ)^T Σ⁻¹ (x-μ)  （忽略常数项）
    """
    diff = theta - pt.as_tensor_variable(_mu)
    cov_inv_tensor = pt.as_tensor_variable(_cov_inv)

    # -0.5 * diff^T @ Σ⁻¹ @ diff
    logp = -0.5 * pt.dot(diff, pt.dot(cov_inv_tensor, diff))
    return logp

# =====================================================================

if __name__ == "__main__":
    freeze_support()

    with pm.Model() as model:
        theta = pm.Flat("theta", shape=5)             # ← 5 维
        pm.Potential("gauss5d", gauss5d_logp(theta))

        # ==================== 采样设置（和之前一样） ====================
        idata = pm.sample(
            draws=10000,
            tune=1000,
            chains=4,
            step=pm.Metropolis(),         # MH 基准
            random_seed=42,
            return_inferencedata=True,
            progressbar=True
        )

    # 保存（可选）
    # idata.to_netcdf("pymc_gauss5d_MH.nc")

    # ==================== ESS + R-hat ====================
    ess  = az.ess(idata, var_names="theta")
    rhat = az.rhat(idata, var_names="theta")

    print("\n=== PyMC MH Gauss 5D（中等相关）的原始 ESS ===")
    print(ess)

    mean_ess = float(ess["theta"].mean())
    print(f"\n=== Mean ESS (PyMC MH Gauss 5D): {mean_ess:.0f}   ← 直接和你自定义对比！")

    print("\n=== R-hat ===")
    print(rhat)

    # ==================== 后验均值（验证是否收敛到真实 μ） ====================
    post_mean = idata.posterior["theta"].mean(dim=["chain", "draw"]).values
    print(f"\n=== Posterior Mean（应 ≈ [1, -1, 0.5, 2, -0.5]） ===")
    print(np.round(post_mean, 4))

    post_std = idata.posterior["theta"].std(dim=["chain", "draw"]).values
    print(f"\n=== Posterior Std（应 ≈ [1, 1.41, 1, 1.41, 1]） ===")
    print(np.round(post_std, 4))
