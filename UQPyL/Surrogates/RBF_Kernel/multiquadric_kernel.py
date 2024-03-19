from .base_kernel import Kernel
import numpy as np

class Multiquadric(Kernel):
    name="Multiquadric"
    def __init__(self,gamma=1):
        self.gamma=gamma
    def evaluate(self,dist):
        
        return np.sqrt(np.power(dist,2)+self.gamma*self.gamma)

    def get_degree(self,n_samples):
        
        return 1
    
    def get_Tail_Matrix(self,train_X):
        
        return (True,np.ones((train_X.shape[0],1)))