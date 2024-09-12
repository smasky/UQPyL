from .base_kernel import BaseKernel
import numpy as np

class ThinPlateSpline(BaseKernel):
    
    name="Thin_plate_spline"
    
    def __init__(self, epsilon: float=1.0, epsilon_ub: float=1e5, epsilon_lb: float=1e-5):
        
        super().__init__()
        self.setPara("epsilon", epsilon, epsilon_lb, epsilon_ub)
        
    def evaluate(self, dist):
        
        epsilon=self.getPara("epsilon")
        
        dist[dist < np.finfo(float).eps] = np.finfo(float).eps
        
        return np.power(dist*epsilon,2)*np.log(dist*epsilon)
    
    def get_Tail_Matrix(self, xTrain):
        
        nSample, nFeature = xTrain.shape
        Tail=np.ones((nSample, nFeature+1))
        Tail[:self.n_samples,:self.n_features]=xTrain
        
        return (True,Tail)
    
    def get_degree(self,n_samples):
        
        return n_samples+1
    