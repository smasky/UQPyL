from .base_kernel import BaseKernel
import numpy as np
class Cubic(BaseKernel):
    
    name = "Cubic"
    
    def __init__(self, epsilon: float = 1.0, 
                 epsilon_attr: dict = {'ub': 1e5, 'lb': 1e-5, 'type': 'float', 'log': True}):
        
        super().__init__()
        
        self.setting.setPara("epsilon", epsilon, epsilon_attr)
        
    def evaluate(self, dist):
        
        epsilon = self.setting.getVals("epsilon")
    
        return np.power(dist*epsilon,3)
    
    def get_Tail_Matrix(self, xTrain):
        
        nSample, nFeature = xTrain.shape
        Tail = np.ones((nSample, nFeature+1))
        Tail[:nSample, :nFeature] = xTrain
        
        return ( True , Tail )
    
    def get_degree(self, nSample):
        
        return nSample+1