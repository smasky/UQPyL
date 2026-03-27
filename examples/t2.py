import pymc as pm
import arviz as az
from multiprocessing import freeze_support
import numpy as np 
import pytensor.tensor as pt
# ==================== 【你的 Banana log density】 ====================
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
        theta = pm.Flat("theta", shape=5)
        pm.Potential("banana", gauss5d_logp(theta))

        # ==================== AMH 专用设置（和之前完全一致，只是改标签） ====================
        idata = pm.sample(
            draws=10000,                  # 和你完全一样
            tune=1000,                    # ← 这就是自适应关键！（AMH 的核心）
            chains=4,
            step=pm.Metropolis(),         # PyMC 的 Adaptive MH
            random_seed=42,
            return_inferencedata=True,
            progressbar=True
        )

    # 保存成 AMH 专属文件（方便和你的自定义 Adaptive MH .nc 对比）
    # idata.to_netcdf("pymc_banana_AMH.nc")

    # ==================== 计算 ESS（和你之前代码 100% 一样） ====================
    ess  = az.ess(idata, var_names="theta")
    rhat = az.rhat(idata, var_names="theta")

    print("\n=== PyMC AMH 的原始 ESS ===")
    print(ess)

    mean_ess = float(ess["theta"].mean())
    print(f"\n=== Mean ESS (PyMC AMH Banana): {mean_ess:.0f}   ← 直接和你自定义 AMH 对比！")

    print("\n=== R-hat ===")
    print(rhat)
