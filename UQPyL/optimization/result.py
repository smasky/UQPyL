import numpy as np 

from .population import Population
from .metric import HV, IGD
class Result():
    
    def __init__(self, algorithm):
        
        self.bestDec=None
        self.bestObj=None
        self.appearFEs=None
        self.appearIters=None
        self.historyBestDecs={}
        self.historyBestObjs={}
        self.historyDecs={}
        self.historyObjs={}
        self.historyFEs={}
        self.historyBestMetrics={}
        
        self.algorithm=algorithm
        
    def update(self, pop: Population, FEs, iter, type=0):
    
        decs=np.copy(pop.decs); objs=np.copy(pop.objs)
        if type==0:
            
            if self.bestObj==None or np.min(objs)<self.bestObj:
                ind=np.where(objs==np.min(objs))
                self.bestDec=decs[ind[0][0], :]
                self.bestObj=objs[ind[0][0], :]
                self.appearFEs=FEs
                self.appearIters=iter
                
            self.historyFEs[FEs]=iter
            self.historyDecs[FEs]=decs
            self.historyObjs[FEs]=objs
            self.historyBestDecs[FEs]=self.bestDec
            self.historyBestObjs[FEs]=self.bestObj
            
        else:
            
            bests=pop.getBest()
            optimum=self.algorithm.problem.getOptimum()
            
            igdValue=None; hvValue=None
            if optimum is not None:
                optimum=optimum[~np.isnan(optimum).any(axis=1)]
                igdValue=IGD(bests, optimum)
            
            hvValue=HV(bests)
            self.historyDecs[FEs]=pop.decs
            self.historyObjs[FEs]=pop.objs
            self.historyBestDecs[FEs]= bests.decs
            self.historyBestMetrics[FEs]= [(hvValue, igdValue) if igdValue is not None else (hvValue)]
            self.historyBestObjs[FEs]= bests.objs
            self.historyFEs[FEs]=iter
            self.bestDec=bests.decs
            self.bestObj=bests.objs
            self.bestMetric=(hvValue, igdValue) if igdValue is not None else (hvValue)
            self.appearFEs=FEs
            self.appearIters=iter

    def generateHDF5(self):
        
        type = 1 if self.algorithm.problem.nOutput>1 else 0
        
        historyPopulation={}
        
        digit=len(str(abs(self.algorithm.iters)))
        
        for key in self.historyDecs.keys():
            
            decs=self.historyDecs[key]
            objs=self.historyObjs[key]
            iter=self.historyFEs[key]
            
            item={"FEs" : key , "Decisions" : decs, "Objectives" : objs}

            historyPopulation[f"iter "+str(iter).zfill(digit)]=item
        
        historyBest={}
        for key in self.historyBestDecs.keys():
            
            bestDecs=self.historyBestDecs[key]
            bestObjs=self.historyBestObjs[key]
            iter=self.historyFEs[key]
            
            if type==0:
                item={"FEs" : key, "Best Decisions" : bestDecs, "Best Objectives" : bestObjs}
            else:
                metrics=self.historyBestMetrics[key]
                if isinstance(metrics[0], tuple):
                    item={"FEs" : key, "Best Decisions" : bestDecs, "Best Objectives" : bestObjs, "HV": metrics[0][0], "IGD": metrics[0][1]}
                else:
                    item={"FEs" : key, "Best Decisions" : bestDecs, "Best Objectives" : bestObjs, "HV": metrics}
                    
            historyBest[f"iter "+str(iter).zfill(digit)]=item
        
        globalBest={}
        globalBest["Best Decisions"]=self.bestDec
        globalBest["Best Objectives"]=self.bestObj
        globalBest["FEs"]=self.appearFEs
        globalBest["Iter"]=self.appearIters
        
        if type==1:
            if isinstance(self.bestMetric, tuple):
                globalBest["HV"]=self.bestMetric[0]
                globalBest["IGD"]=self.bestMetric[1]
            else:
                globalBest["HV"]=self.bestMetric
        
        result={ "History_Population" : historyPopulation,
                 "History_Best" : historyBest,
                 "Global_Best" : globalBest,
                 "Max_Iter" : self.algorithm.iters,
                 "Max_FEs" : self.algorithm.FEs }
        
        return result
        
    def reset(self):
        self.bestDec=None; self.bestObj=None
        self.appearFEs=None; self.appearIters=None
        self.historyBestDecs={}; self.historyBestObjs={}
        self.historyDecs={}; self.historyObjs={}
        self.historyFEs={}; self.historyMetrics={}