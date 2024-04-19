from .base_kernel import Kernel
import numpy as np

class Thin_plate_spline(Kernel):
    name="Thin_plate_spline"
    def evaluate(self,dist):
        
        dist[dist < np.finfo(float).eps] = np.finfo(float).eps
        return np.power(dist,2)*np.log(dist)
    
    def get_Tail_Matrix(self,train_X):
        
        Tail=np.ones((self.n_samples,self.n_features+1))
        Tail[:self.n_samples,:self.n_features]=train_X.copy()
        return (True,Tail)
    
    def get_degree(self,n_samples):
        
        return n_samples+1
    