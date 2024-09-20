from typing import Optional, Union
from scipy.spatial.distance import pdist, squareform, cdist
import numpy as np

from .base_kernel import BaseKernel

class RBF(BaseKernel):
    """
        RBF kernel
    """
    def __init__(self, length_scale: Union[float, np.ndarray]=1.0,
                 length_ub: Union[float, np.ndarray]=1e5, length_lb: Union[float, np.ndarray]=1,
                 heterogeneous: bool=False):
        
        super().__init__()
        
        self.setPara("l", length_scale, length_lb, length_ub)
        self.heterogeneous=heterogeneous
        
    def __call__(self, xTrain1: np.ndarray, xTrain2: Optional[np.ndarray]=None):
        
        length_scale=self.getPara("l")

        if xTrain2 is None:
            dists=pdist(xTrain1/length_scale, metric="sqeuclidean")
            K=squareform(np.exp(-0.5*dists))
            np.fill_diagonal(K,1.0)
        else:
            dists=cdist(xTrain1/length_scale, xTrain2/length_scale, metric="sqeuclidean")
            K = np.exp(-0.5 * dists)
        return K
    
    def initialize(self, nInput):
        
        length_scale=self.getPara("l")
        length_ub=self.setting.paras_ub["l"]
        length_lb=self.setting.paras_lb["l"]
        
        if self.heterogeneous:
            if isinstance(length_scale, float):
                length_scale=np.ones(nInput)*length_scale
            elif length_scale.size==1:
                length_scale=np.repeat(length_scale, nInput)
            elif length_scale.size!=nInput:
                raise ValueError("the dimension of length_scale is not consistent with the number of input")
            
            if isinstance(length_ub, float):
                length_ub=np.ones(nInput)*length_ub
            elif length_ub.size==1:
                length_ub=np.repeat(length_ub, nInput)
            elif length_ub.size!=nInput:
                raise ValueError("the dimension of length_ub is not consistent with the number of input")
            
            if isinstance(length_lb, float):
                length_lb=np.ones(nInput)*length_lb
            elif length_lb.size==1:
                length_lb=np.repeat(length_lb, nInput)
            elif length_lb.size!=nInput:
                raise ValueError("the dimension of length_lb is not consistent with the number of input")
        
        self.setPara("l", length_scale, length_lb, length_ub)