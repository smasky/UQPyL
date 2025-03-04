from .base_kernel import BaseKernel
import numpy as np

class Gaussian(BaseKernel):
    
    name = "Gaussian"
    
    def __init__(self, epsilon: float = 1.0, 
                 epsilon_attr: dict = {'ub': 1e5, 'lb': 1e-5, 'type': 'float', 'log': True}):
        
        super().__init__()
        
        self.setting.setPara("epsilon", epsilon, epsilon_attr)
        
    def evaluate(self, dist):
        
        epsilon=self.setting.getVals("epsilon")
        
        return np.exp(-1*epsilon*np.power(dist,2))