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
        self.algorithm=algorithm
        self.historyMetrics={}
    def update(self, pop: Population, FEs, Iters, type=0):
    
        decs=np.copy(pop.decs); objs=np.copy(pop.objs)
        if type==0:
            
            if self.bestObj==None or np.min(objs)<self.bestObj:
                ind=np.where(objs==np.min(objs))
                self.bestDec=decs[ind[0][0], :]
                self.bestObj=objs[ind[0][0], :]
                self.appearFEs=FEs
                self.appearIters=Iters
                
            self.historyFEs[FEs]=Iters
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
            self.historyDecs[FEs]= bests.decs
            self.historyMetrics[FEs]= [(hvValue, igdValue) if igdValue is not None else (hvValue)]
            self.historyObjs[FEs]= bests.objs
            self.bestDec=bests.decs
            self.bestObj=bests.objs
            self.bestMetric=(hvValue, igdValue) if igdValue is not None else (hvValue)
            self.appearFEs=FEs
            self.appearIters=Iters

    def reset(self):
        
        self.bestDec=None; self.bestObj=None
        self.appearFEs=None; self.appearIters=None
        self.historyBestDecs={}; self.historyBestObjs={}
        self.historyDecs={}; self.historyObjs={}
        self.historyFEs={}