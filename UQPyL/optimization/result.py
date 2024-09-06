import numpy as np

from .population import Population

class Result():
    def __init__(self):
        
        self.bestDec=None
        self.bestObj=None
        self.appearFEs=None
        self.appearIters=None
        self.historyBestDecs={}
        self.historyBestObjs={}
        self.historyDecs={}
        self.historyObjs={}
        self.historyFEs={}
    
    def update(self, pop: Population, FEs, Iters):
        
        decs=np.copy(pop.decs); objs=np.copy(pop.objs)
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
    
    def reset(self):
        
        self.bestDec=None; self.bestObj=None
        self.appearFEs=None; self.appearIters=None
        self.historyBestDecs={}; self.historyBestObjs={}
        self.historyDecs={}; self.historyObjs={}
        self.historyFEs={}