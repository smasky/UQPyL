from .base_kernel import BaseKernel
import numpy as np

class Gaussian(BaseKernel):
    
    name="Gaussian"
    
    def __init__(self, epsilon: float=1.0, epsilon_ub: float=1e5, epsilon_lb: float=1e-5):
        
        super().__init__()
        self.setPara("epsilon", epsilon, epsilon_lb, epsilon_ub)
        
    def evaluate(self, dist):
        
        epsilon=self.getPara("epsilon")
        
        return np.exp(-1*epsilon*np.power(dist,2))