from typing import Optional, Union
from scipy.spatial.distance import pdist, squareform, cdist
import scipy.special as sp
from scipy.integrate import quad
import numpy as np

from .base_kernel import BaseKernel


class Matern(BaseKernel):
    """Matérn correlation with a positive length scale.

    Fixed ``nu`` accepts any positive real value, or ``np.inf`` for the
    Gaussian limit. With ``optimize_nu=True``, optimization chooses from
    0.5, 1.5, 2.5 and np.inf. General orders use the Bessel formula with
    a slower integral fallback for large orders or intermediate overflow.
    """
    name = "Matern"

    def __init__(self, length_scale: Union[float, np.ndarray] = 1.0,
                 length_attr: dict = {'ub': 1e5, 'lb': 1, 'type': 'float', 'log': True},
                 nu: float = 1.5,
                 optimize_nu: bool = False,
                 heterogeneous: bool = False):

        super().__init__()
        
        self.heterogeneous = heterogeneous
        
        self._setKernelParameter("l", length_scale, length_attr)

        self._validateParameter("nu", nu)
        nu = float(np.asarray(nu).item())

        if optimize_nu:
            choices = [0.5, 1.5, 2.5, np.inf]
            if nu not in choices:
                raise ValueError("Optimized nu must be one of 0.5, 1.5, 2.5, np.inf.")
            nu_attr = {'ub': 1, 'lb': 0, 'type': 'discrete', 'log': False, 'set': choices}
            # Setting stores numeric discrete values as bin coordinates.
            nu = (choices.index(nu) + 0.5) / len(choices)
        else:
            nu_attr = None
            
        self.setting.set("nu", nu, nu_attr)
        
    def diag(self, X):
        self._validateInputs(X)
        return np.ones(len(X))

    def __call__(self, xTrain1: np.ndarray, xTrain2: Optional[np.ndarray]=None):
        self._validateInputs(xTrain1, xTrain2)
        
        length_scale = self.setting.get("l")
        
        nu = self.setting.get("nu")
        
        if xTrain2 is None:
            dists = pdist(xTrain1/length_scale, metric="euclidean")
        else:
            dists = cdist(xTrain1/length_scale, xTrain2/length_scale, metric="euclidean")
        
        if nu==0.5:
            
            K=np.exp(-dists)
            
        elif nu==1.5:
            
            K=dists*np.sqrt(3)
            K=(1.0+K)* np.exp(-K)
            
        elif nu==2.5:
            
            K=dists*np.sqrt(5)
            K=(1.0+K+K**2/3.0) * np.exp(-K)
            
        elif nu==np.inf:
            
            K=np.exp(-0.5*dists**2)
            
        else:
            
            K = self._generalCorrelation(dists, nu)

        if xTrain2 is None:
            
            K = squareform(K)
            np.fill_diagonal(K,1.0)
        
        return K

    @staticmethod
    def _generalCorrelation(dists, nu):
        result = np.ones_like(dists)
        active = (dists > 0) & np.isfinite(dists)
        result[np.isinf(dists)] = 0.0
        # From the Gamma-mixture representation, for nu > 1:
        # 0 <= 1-k(d) <= nu*d**2 / (2*(nu-1)). Only round to 1 when
        # this error is below machine precision, never on Bessel overflow.
        if nu > 1:
            threshold = np.sqrt(2 * np.finfo(float).eps * ((nu - 1) / nu))
            active &= dists > threshold
        distances = dists[active]
        values = np.full(distances.shape, np.nan)
        if nu < 50:
            scaledDist = np.sqrt(nu) * np.sqrt(2.0) * distances
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                values = np.exp((1 - nu) * np.log(2.0) - sp.gammaln(nu)
                                + nu * np.log(scaledDist)
                                + np.log(sp.kve(nu, scaledDist)) - scaledDist)
        fallback = ~np.isfinite(values)
        # Repeated distances share the relatively expensive quadrature.
        uniqueDist, inverse = np.unique(distances[fallback], return_inverse=True)
        stableValues = np.array([Matern._integralCorrelation(d, nu) for d in uniqueDist])
        values[fallback] = stableValues[inverse]
        # Both formulas can exceed one by a few ulps near the origin.
        result[active] = np.minimum(values, 1.0)
        result[np.isnan(dists)] = np.nan
        return result

    @staticmethod
    def _integralCorrelation(distance, nu):
        """Evaluate the Gamma-mixture integral around its log-space mode.

        k(d) = integral t**(nu-1)*exp(-t-nu*d**2/(2*t)) dt / Gamma(nu).
        This follows from DLMF 10.32.10. Set t=(nu+b)*exp(v/width),
        where b*(nu+b)=nu*d**2/2 and width**2=nu+2*b. Centering and
        scaling keep the integrand resolvable even for very large nu.
        """
        with np.errstate(over="ignore"):
            ratio = np.hypot(1.0, (distance / np.sqrt(nu)) * np.sqrt(2.0))
            b = (distance / (ratio + 1.0)) * distance
        if not np.isfinite(b):
            return 0.0
        width = np.sqrt(nu) * np.sqrt(ratio)
        if nu >= 16:
            # Stirling's remainder avoids subtracting O(nu*log(nu)) terms.
            invNu = 1.0 / nu
            correction = invNu * (1/12 + invNu**2 *
                                  (-1/360 + invNu**2 * (1/1260 - invNu**2/1680)))
            relativeB = b / nu
            logTerm = (b * (1 - relativeB/2 + relativeB**2/3)
                       if relativeB < 1e-5 else nu * np.log1p(relativeB))
            logPrefactor = (logTerm - 2*b - 0.5*np.log(ratio)
                            - 0.5*np.log(2*np.pi) - correction)
        else:
            logPrefactor = (nu*np.log(nu+b) - nu - 2*b
                            - sp.gammaln(nu) - np.log(width))

        def expResidual(value):
            # exp(value)-1-value without cancellation at small arguments.
            if abs(value) < 0.01:
                return value**2 * (0.5 + value * (1/6 + value *
                       (1/24 + value * (1/120 + value * (1/720 + value/5040)))))
            return np.expm1(value) - value

        def integrand(value):
            u = value / width
            if abs(u) > 700:
                return 0.0
            positive = expResidual(u)
            negative = expResidual(-u)
            with np.errstate(over="ignore", under="ignore"):
                return np.exp(-nu*positive - b*(positive + negative))

        left = quad(integrand, -np.inf, 0.0, epsabs=1e-12, epsrel=1e-12)[0]
        right = quad(integrand, 0.0, np.inf, epsabs=1e-12, epsrel=1e-12)[0]
        return np.exp(logPrefactor + np.log(left + right))
