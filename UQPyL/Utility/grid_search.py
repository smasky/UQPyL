from .metrics import r2_score, mse, rank_score
from .model_selections import KFold
from typing import Dict, Literal
import numpy as np
import itertools

class GridSearch():
    def __init__(self, para_grid: Dict, Evaluator, 
                 CV: int=5, Metric: Literal["r2_score", "mse", "rank_score"]="r2_score"):
        
        self.Evaluator=Evaluator
        self.para_grid=para_grid
        self.CV=CV
        self.Metric=eval(Metric)
    
    def start(self, dataX, dataY):
        
        kFold=KFold(self.CV)
        train_sets, test_sets=kFold.split(dataX)
        
        combos=itertools.product(*self.para_grid.values())
        combinations= [dict(zip(self.para_grid.keys(), combo)) for combo in combos]
        
        for para in combinations:
            self.Evaluator.set_Paras(para)
            res=[]
            for train_index, test_index in zip(train_sets, test_sets):
                trainX=dataX[train_index,:];trainY=dataY[train_index,:]
                testX=dataX[test_index,:];testY=dataY[test_index,:]
                self.Evaluator.fit(trainX,trainY)
                Predict_Y=self.Evaluator.predict(testX)
                res.append(self.Metric(testY,Predict_Y))
            print(para)
            print(res)
                
                
                
                
                
        
        
        
        
            
            
        
        
        
        
        
        
        
    
    