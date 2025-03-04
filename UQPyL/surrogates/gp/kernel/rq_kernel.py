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
    def __init__(self, length_scale: Union[float, np.ndarray]=1.0, 
                 length_attr: dict = {'ub': 1e5, 'lb': 1, 'type': 'float', 'log': True},
                 alpha: float=1.0, alpha_attr: dict = {'ub': 1e5, 'lb': 1e-5, 'type': 'float', 'log': True},
                 heterogeneous: bool=False):
        
        super().__init__()
        self.setting.setPara("l", length_scale, length_attr)
        self.setting.setPara("alpha", alpha, alpha_attr)
        self.heterogeneous=heterogeneous
       
    def __call__(self, xTrain1: np.ndarray, xTrain2: Optional[np.ndarray]=None):
        
        length_scale=self.setting.getVals("l")
        alpha=self.setting.getVals("alpha")
        
        if xTrain2 is None:
            dists=squareform(pdist(xTrain1/length_scale, metric="sqeuclidean"))
            np.fill_diagonal(dists,1)
            tmp= dists / (2*alpha)
            base=1 + tmp
            K=base**-alpha
        else:
            dists=cdist(xTrain1/length_scale, xTrain2/length_scale, metric="sqeuclidean")
            K= (1+dists / (2* alpha )) ** -alpha
        
        return K