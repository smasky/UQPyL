from .base_kernel import BaseKernel
import numpy as np
class Cubic(BaseKernel):
    
    name="Cubic"
    
    def __init__(self, epsilon: float=1.0, epsilon_ub: float=1e5, epsilon_lb: float=1e-5):
        
        super().__init__()
        self.setParameters("epsilon", epsilon, epsilon_lb, epsilon_ub)
        
    def evaluate(self, dist):
        
        epsilon=self.getParaValue("epsilon")
    
        return np.power(dist*epsilon,3)
    
    def get_Tail_Matrix(self, xTrain):
        
        nSample, nFeature = xTrain.shape
        Tail = np.ones((nSample, nFeature+1))
        Tail[:nSample, :nFeature] = xTrain
        
        return ( True , Tail )
    
    def get_degree(self, nSample):
        
        return nSample+1