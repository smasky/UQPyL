from typing import Optional, Union
from scipy.spatial.distance import pdist, squareform, cdist
import scipy.special as sp
import numpy as np

from .base_kernel import BaseKernel


class Matern(BaseKernel):
    """
       Matern Kernel
       
       Parameters:
       
       nu: this parameter determine the smooth of the prediction
       
       length_scale:  a scaler or vector to determine the correlation of the input data
       
       ub, lb: the upper or lower bound of the length_scale
       
       Attribute:
       
       theta: the set of unknown parameters 
 
    """
    def __init__(self, length_scale: Union[float, np.ndarray]=1.0,
                 length_ub: Union[float, np.ndarray]=1e5, length_lb: Union[float, np.ndarray]=1e-5,
                 heterogeneous: bool=False,
                 nu: float=1.5):
                
        super().__init__()
        
        self.nu=nu
        
        self.setPara("length_scale", length_scale, length_ub, length_lb)
        self.nu = nu #TODO
        
    def __call__(self, xTrain1: np.ndarray, xTrain2: Optional[np.ndarray]=None):
        
        length_scale=self.getPara("length_scale")
        nu=self.nu
        if xTrain2 is None:
            dists=pdist(xTrain1/length_scale, metric="euclidean")
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
            
            K=np.exp(-(dists**2)/2.0)
            
        else:
            
            factor = (2 ** (1 - nu)) / sp.gamma(nu)
            # Argument for the Bessel function
            scaled_dist = np.sqrt(2 * nu) * dists / length_scale
            # Matérn kernel formula
            K = factor * (scaled_dist ** nu) * sp.kv(nu, scaled_dist)
            
        if xTrain2 is None:
            K=squareform(K)
            np.fill_diagonal(K,1.0)
        
        return K