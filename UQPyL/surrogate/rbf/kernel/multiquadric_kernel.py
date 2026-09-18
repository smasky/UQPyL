from .base_kernel import BaseKernel
import numpy as np

class Multiquadric(BaseKernel):
    
    name="Multiquadric"
    # The positive multiquadric kernel is conditionally negative definite.
    smoothingSign = -1.0
    
    def __init__(self, epsilon: float = 1.0, 
                 epsilon_attr: dict = {'ub': 1e5, 'lb': 1e-5, 'type': 'float', 'log': True}):
        
        super().__init__()
        
        self._setKernelParameter("epsilon", epsilon, epsilon_attr)
        
    def evaluate(self, dist):
        self.validateParameters()
        
        epsilon=self.setting.get("epsilon")
        
        return np.sqrt(np.power(dist*epsilon, 2)+1)

    def get_degree(self,nSample):
        
        return 1
    
    def get_Tail_Matrix(self, xTrain):
        
        return (True, np.ones((xTrain.shape[0],1)))
