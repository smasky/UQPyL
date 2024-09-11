from .base_kernel import BaseKernel
import numpy as np

class Linear(BaseKernel):
    
    name="Linear"
    
    def __init__(self, epsilon: float=1.0, epsilon_ub: float=1e5, epsilon_lb: float=1e-5):
        
        super().__init__()
        self.setParameters("epsilon", epsilon, epsilon_lb, epsilon_ub)
    
    def evaluate(self, dist):
        
        epsilon=self.getParaValue("epsilon")
        
        return dist*epsilon
    
    def get_degree(self, nSample):
        
        return 1
    
    def get_Tail_Matrix(self, xTrain):
        
        return (True,np.ones((xTrain.shape[0],1)))