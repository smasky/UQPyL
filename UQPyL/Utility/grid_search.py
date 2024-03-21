from .metrics import r2_score, mse, rank_score
from .model_selections import KFold
from typing import Dict, Literal
import numpy as np
import itertools
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def fit_predict(evaluator,dataX, dataY, train_sets, test_sets, metric):
    
    values=[]
    for train_set, test_set in zip(train_sets, test_sets):
        trainX=dataX[train_set,:];trainY=dataY[train_set,:]
        testX=dataX[test_set,:];testY=dataY[test_set,:]
        
        evaluator.fit(trainX, trainY)
        P_Y,_=evaluator.predict(testX)
        value=eval(metric)(testY,P_Y)
        values.append(value)
    
    return np.mean(values)

class GridSearch():
    def __init__(self, para_grid: Dict, Evaluator, 
                 CV: int=5, Metric: Literal["r2_score", "mse", "rank_score"]="r2_score",
                 workers: int=8):
        
        self.Evaluator=Evaluator
        self.para_grid=para_grid
        self.CV=CV
        self.Metric=eval(Metric)
        self.workers=workers
    
    def start(self, dataX, dataY):
        
        kFold=KFold(self.CV)
        train_sets, test_sets=kFold.split(dataX)
        
        combos=itertools.product(*self.para_grid.values())
        combinations= [dict(zip(self.para_grid.keys(), combo)) for combo in combos]
        
        # trainX=dataX[train_sets[0],:];trainY=dataY[train_sets[0],:]
        # testX=dataX[test_sets[0],:];testY=dataY[test_sets[0],:]
        
        with ThreadPoolExecutor(max_workers=self.workers) as exe:
            futures={}
            i=0
            for para in combinations:
                tempEvaluator=copy.deepcopy(self.Evaluator)
                tempEvaluator.set_Paras(para)
                future=exe.submit(fit_predict, tempEvaluator,dataX, dataY, train_sets, test_sets, "r2_score")
                futures[future]=para

            bestValue=-np.inf
            bestPara=None
            
            for future in as_completed(futures):
                res=future.result()
                para=futures[future]
                if res>bestValue:
                    bestPara=para   
                    bestValue=res 
            return bestPara, bestValue 
        
                
                
                
                
                
        
        
        
        
            
            
        
        
        
        
        
        
        
    
    