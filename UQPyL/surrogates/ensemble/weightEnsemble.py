import numpy as np
from ..surrogateABC import Surrogate

class weightEnsemble(Surrogate):
    def __init__(self):
        self.srgList = []
        self.weight = []
    
    def fit(self, trainX, trainY):
        
        for srg in self.srgList:
            srg.fit(trainX, trainY)
    
    def predict(self, xPred):
        
        Y = np.zeros(len(self.srgList))
        
        for i, srg in enumerate(self.srgList):
            Y[i] = srg.predict(xPred)
        
        return np.sum(np.array(self.weight) * Y)
    
    def addSrg(self, srg):
        
        self.srgList.append(srg)