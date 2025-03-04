from typing import Optional, Union
from scipy.spatial.distance import pdist, squareform, cdist
import numpy as np

from .base_kernel import BaseKernel

class RBF(BaseKernel):
    """
        RBF kernel
    """
    def __init__(self, length_scale: Union[float, np.ndarray]=1.0,
                 length_attr: dict = {'ub': 1e5, 'lb': 1, 'type': 'float', 'log': True},
                 heterogeneous: bool=False):
        
        super().__init__()
        
        self.setting.setPara("l", length_scale, length_attr)
        
        self.heterogeneous = heterogeneous
        
    def __call__(self, xTrain1: np.ndarray, xTrain2: Optional[np.ndarray]=None):
        
        length_scale=self.setting.getVals("l")

        if xTrain2 is None:
            dists=pdist(xTrain1/length_scale, metric="sqeuclidean")
            K=squareform(np.exp(-0.5*dists))
            np.fill_diagonal(K,1.0)
        else:
            dists=cdist(xTrain1/length_scale, xTrain2/length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
        return K