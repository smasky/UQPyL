from .metrics import r2_score, mse, rank_score
from .model_selections import KFold
from typing import Dict, Literal
import numpy as np
import itertools
import copy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time

def fit_predict(evaluator, trainX, trainY, testX, testY, metric,i):
    print("Begin{}".format(i))
    evaluator.fit(trainX, trainY)
    P_Y=evaluator.predict(testX)
    time.sleep(1)
    value=eval(metric)(testY,P_Y)
    print("End{}".format(i))
    return value

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
        
        Results=[]
        trainX=dataX[train_sets[0],:];trainY=dataY[train_sets[0],:]
        testX=dataX[test_sets[0],:];testY=dataY[test_sets[0],:]
        a=time.time()
        with ThreadPoolExecutor(max_workers=12) as exe:
            futures={}
            i=0
            for para in combinations:
                tempEvaluator=copy.deepcopy(self.Evaluator)
                tempEvaluator.set_Paras(para)
                future=exe.submit(fit_predict, tempEvaluator, trainX, trainY, testX, testY, "r2_score",i)
                futures[future]=para
                i+=1
            Res=[]
            for future in as_completed(futures):
                res=future.result()
                para=futures[future]
                Res.append((para,res))
        # for para in combinations:
        #     tempEvaluator=copy.deepcopy(self.Evaluator)
        #     tempEvaluator.set_Paras(para)
        #     fit_predict(tempEvaluator,trainX,trainY,testX,testY, "r2_score", 0)
        b=time.time()
        print(b-a)
        a=1            
            # for train_index, test_index in zip(train_sets, test_sets):
            #     trainX=dataX[train_index,:];trainY=dataY[train_index,:]
            #     testX=dataX[test_index,:];testY=dataY[test_index,:]
            #     self.Evaluator.fit(trainX,trainY)
            #     Predict_Y=self.Evaluator.predict(testX)
            #     res.append(self.Metric(testY,Predict_Y))
            # Results.append(np.mean(res))
        
        # Index=np.argmax(np.array(Results))
        # print(combinations[Index])
        # print(Results[Index])
        
        # return combinations[Index]
    
        
                
                
                
                
                
        
        
        
        
            
            
        
        
        
        
        
        
        
    
    