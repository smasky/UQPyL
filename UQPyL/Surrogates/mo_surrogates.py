import numpy as np
from .surrogate_ABC import Surrogate
class MO_Surrogates():
    def __init__(self, N_Surrogates, Models_list=[]):
        self.N_Surrogates=N_Surrogates
        for model in Models_list:
            if model is not Surrogate:
                ValueError("Please append the type of surrogate!")           
        self.models_list=Models_list
    def append(self, model: Surrogate):
        if model is not Surrogate:
            ValueError("Please append the type of surrogate!")
        self.models_list.append(model)
    
    def fit(self, trainX, trainY):
        for i, model in enumerate(self.models_list):
            model.fit(trainX, trainY[:, i])
    
    def predict(self, testX):
        M,_=testX.shape
        res=[]
        for model in self.models_list:
            res.append(model.predict(testX))
        pre_Y=np.hstack(res)
        return pre_Y