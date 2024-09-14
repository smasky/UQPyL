import numpy as np
from scipy.spatial.distance import pdist,squareform
import abc

class BaseKernel(metaclass=abc.ABCMeta):
    name="kernel"
    def __init__(self):
        
        self.setting=Setting(self.name)
        
    def __check_array__(self, value):
            
        if isinstance(value, float):
            value=np.array([value])
        elif isinstance(value, np.ndarray):
            if value.ndim>1:
                value=value.ravel()
        else:
            raise ValueError("Please make sure the used type (float or np.ndarray) of value")
        
        return value
    
    
    def evaluate(self, pdist):
        pass
    
    def get_A_Matrix(self, xTrain):
        dist=squareform(pdist(xTrain,'euclidean'))
        Phi=self.evaluate(dist)
        
        _, nFeature=xTrain.shape
        
        S, Tail=self.get_Tail_Matrix(xTrain)
        if(S):
            temp1=np.hstack((Phi, Tail))
            # t1=Tail.transpose()
            # t2=np.zeros((self.get_degree(nFeature),self.get_degree(nFeature)))
            temp2=np.hstack((Tail.transpose(), np.zeros((self.get_degree(nFeature), self.get_degree(nFeature)))))
            A_Matrix=np.vstack((temp1, temp2))
        else:
            A_Matrix=Phi
        return A_Matrix
            
    def get_Tail_Matrix(self, xTrain): 
        return (False,None)
    
    def get_degree(self, nSamples):
        return None
    
    def setPara(self, key, value, lb, ub):
        
        self.setting.setPara(key, value, lb, ub)
    
    def getPara(self, *args):
        
        return self.setting.getPara(*args) 
class Setting():
    
    def __init__(self, prefix):
        
        self.prefix=prefix
        self.paras={}
        self.paras_ub={}
        self.paras_lb={}
    
    def setPara(self, key, value, lb, ub):
        
        self.paras[key]=value
        self.paras_lb[key]=lb
        self.paras_ub[key]=ub

    def getPara(self, *args):
        
        values=[]
        for arg in args:
            values.append(self.paras[arg])
        
        if len(args)>1:
            return tuple(values)
        else:
            return values[0]
        