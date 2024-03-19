from .base_kernel import Kernel
import numpy as np

class Linear(Kernel):
    name="Linear"
    def evaluate(self,dist):
        return dist
    
    def get_degree(self,n_samples):
        
        return 1
    
    def get_Tail_Matrix(self,train_X):
        
        return (True,np.ones((train_X.shape[0],1)))