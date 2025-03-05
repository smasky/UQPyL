import numpy as np
from typing import Literal, Tuple
import math
class RandSelect():
    def __init__(self, pTest: int=5):
        self.pTest = pTest/100
        
    def split(self, X: np.ndarray) -> Tuple[list, list]:
        nSample,_ = X.shape
        
        nTest = int(nSample*self.pTest)
        nSets = math.floor(nSample/nTest)
        
        index = np.arange(nSample)
        np.random.shuffle(index)
        
        nn = np.floor(nSample/nSets)*nSets
        split_bound = np.linspace(0,nn,nSets+1,dtype=np.int32)
        signal = np.ones(nSample, dtype=np.bool_)
        signal[split_bound[0]:split_bound[1]] = 0
        test = index[split_bound[0]:split_bound[1]].copy()
        train = index[signal].copy()
        
        return train, test

class KFold():
    def __init__(self,n_splits: int=5):
        
        self.n_splits = n_splits
    
    def split(self, X: np.ndarray, mode: Literal['full','single'] ='full') -> Tuple[list, list]:
        
        train = [];test = []
        
        n_sample,_ = X.shape
        index = np.arange(n_sample)
        np.random.shuffle(index)
        
        nn = np.floor(n_sample/self.n_splits)*self.n_splits
        split_bound = np.linspace(0,nn,self.n_splits+1,dtype=np.int32)
        if(mode == 'full'):
            for i in range(self.n_splits):
                signal = np.ones(n_sample,dtype=np.bool_)
                signal[split_bound[i]:split_bound[i+1]] = 0
                test.append(index[split_bound[i]:split_bound[i+1]].copy())
                train.append(index[signal].copy())
        elif(mode == 'single'):
            signal = np.ones(n_sample,dtype=np.bool_)
            signal[split_bound[0]:split_bound[1]] = 0
            test.append(index[split_bound[0]:split_bound[1]].copy())
            train.append(index[signal].copy())
            
        return train, test


            

        
        
        
        
        
        
        