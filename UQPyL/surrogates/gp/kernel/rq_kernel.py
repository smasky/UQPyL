from typing import Optional, Union
from scipy.spatial.distance import pdist, squareform, cdist
import numpy as np

from .base_kernel import BaseKernel


class RationalQuadratic(BaseKernel):
    """
    Constant Kernel
    
    Attribute:
    
    theta: the set of unknown parameters. np.vstack(length_scale, alpha)

    """
    def __init__(self, length_scale: Union[float, np.ndarray]=1.0, length_ub: Union[float, np.ndarray]=10.0, length_lb: Union[float, np.ndarray]=0.0,
                 heterogeneous: bool=False,
                 alpha: float=1.0, alpha_ub: float=1e5, alpha_lb: float=1e-5):
        
        super().__init__()
        self.setPara("l", length_scale, length_lb, length_ub)
        self.setPara("alpha", alpha, alpha_lb, alpha_ub)
        self.heterogeneous=heterogeneous
       
    def __call__(self, xTrain1: np.ndarray, xTrain2: Optional[np.ndarray]=None):
        
        length_scale=self.getPara("l")
        alpha=self.getPara("alpha")
        
        if xTrain2 is None:
            dists=squareform(pdist(xTrain1/length_scale, metric="sqeuclidean"))
            tmp= dists / (2*alpha)
            base=1 + tmp
            K=base**-alpha
            np.fill_diagonal(K,1)
        else:
            dists=cdist(xTrain1, xTrain2, metric="sqeuclidean")
            dists=dists/length_scale
            K= (1+dists / (2* alpha )) ** -alpha
        
        return K