from .base_kernel import Kernel
import numpy as np
class Cubic(Kernel):
    name="Cubic"
    def evaluate(self, dist):
        return np.power(dist,3)
    
    def get_Tail_Matrix(self, train_X):
        Tail=np.ones((self.n_samples,self.n_features+1))
        Tail[:self.n_samples,:self.n_features]=train_X.copy()
        return (True,Tail)
    
    def get_degree(self, n_samples):
        return n_samples+1