from .base_kernel import BaseKernel
import numpy as np

class ThinPlateSpline(BaseKernel):
    
    name="Thin_plate_spline"
    
    def __init__(self, epsilon: float = 1.0, 
                 epsilon_attr: dict = {'ub': 1e5, 'lb': 1e-5, 'type': 'float', 'log': True}):
        
        super().__init__()
        
        self._setKernelParameter("epsilon", epsilon, epsilon_attr)
        
    def evaluate(self, dist):
        self.validateParameters()
        
        epsilon = self.setting.get("epsilon")
        
        scaledDist = np.asarray(dist, dtype=float) * epsilon
        result = np.zeros_like(scaledDist)
        nonzero = scaledDist != 0
        result[nonzero] = scaledDist[nonzero]**2 * np.log(scaledDist[nonzero])
        return result
    
    def get_Tail_Matrix(self, xTrain):
        
        nSample, nFeature = xTrain.shape
        Tail = np.ones((nSample, nFeature+1))
        Tail[:, :nFeature] = xTrain
        
        return (True,Tail)
    
    def get_degree(self,n_samples):
        
        return n_samples+1
    
