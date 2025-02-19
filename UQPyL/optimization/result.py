import numpy as np 

from .population import Population
from .metric import HV
class Result():
    
    def __init__(self, algorithm):
        
        #Current best solution
        self.bestDecs = None
        self.bestObjs = None
        self.bestCons = None
        self.bestMetric = None
        self.bestFeasible = False
        
        #Current best solution appearance
        self.appearFEs = None
        self.appearIters = None
        
        #History Records
        self.historyBestDecs = {}
        self.historyBestObjs = {}
        self.historyBestCons = {}
        self.historyBestMetrics = {}
        
        self.historyDecs = {}
        self.historyObjs = {}
        self.historyCons = {}
        self.historyFEs = {}
        
        self.algorithm = algorithm
        
    def update(self, pop: Population, problem, FEs, iter, algType):
        
        decs = np.copy(pop.decs)
        
        opt = problem.opt
        
        if problem.encoding == 'mix':
            decs = problem._transform_discrete_var(decs)
        
        if algType == 'EA':
            self._update_EA(pop, FEs, iter, opt)
        else:
            self._update_MOEA(pop, FEs, iter, opt)
        
        self._update_history(pop, FEs, iter, opt)
      
    def _update_EA(self, pop, FEs, iter, opt):
        
        #Obtain local optima solutions
        bestPop = pop.getBest(k=1)
        localBestDecs = bestPop.decs[0]
        localBestObjs = bestPop.objs[0]
        localBestCons = bestPop.cons[0] if bestPop.cons is not None else None
        localBestFeasible = True if localBestCons is None else np.all(np.maximum(0, localBestCons) <= 0)
        
        # update global optima solutions
        if self.bestObjs is None or (
            (localBestFeasible and not self.bestFeasible) or
            (localBestFeasible == self.bestFeasible and localBestObjs < self.bestObjs)
        ):
            self.bestDecs = localBestDecs
            self.bestObjs = localBestObjs
            self.bestTrueObjs = localBestObjs * opt
            self.bestCons = localBestCons
            self.bestFeasible = localBestFeasible
            self.appearFEs = FEs
            self.appearIters = iter
    
    def _update_MOEA(self, pop, FEs, iter, opt):
        
        bestPop = pop.getBest()
        localBestDecs = bestPop.decs
        localBestObjs = bestPop.objs
        localBestCons = bestPop.cons if bestPop.cons is not None else None
        localBestFeasible = True if localBestCons is None else np.all(np.maximum(0, localBestCons) <= 0)
        
        self.bestDecs = localBestDecs
        self.bestObjs = localBestObjs
        self.bestTrueObjs = localBestObjs * opt
        self.bestCons = localBestCons
        self.bestFeasible = localBestFeasible
        
        self.bestMetric = HV(pop, refPoint = np.max(pop.objs, axis=0) * 1.1)
        self.historyBestMetrics[FEs] = self.bestMetric
        
        self.appearFEs = FEs
        self.appearIters = iter
        
    def _update_history(self, pop, FEs, iters, opt):
        
        self.historyDecs[FEs] = pop.decs
        self.historyObjs[FEs] = pop.objs * opt
        self.historyCons[FEs] = pop.cons
        self.historyFEs[FEs] = iters
        
        self.historyBestDecs[FEs] = self.bestDecs
        self.historyBestObjs[FEs] = self.bestTrueObjs
        self.historyBestCons[FEs] = self.bestCons
    
    def generateHDF5(self):
        
        alghType = 1 if self.algorithm.problem.nOutput>1 else 0
        
        historyPopulation = {}
        
        digit = len(str(abs(self.algorithm.iters)))
        
        for key in self.historyDecs.keys():
            
            decs = self.historyDecs[key]
            objs = self.historyObjs[key]
            iter = self.historyFEs[key]
            
            item = {"FEs" : key , "Decisions" : decs, "Objectives" : objs}

            if self.historyBestCons[key] is not None:
                item['Constrains'] = self.historyBestCons[key]
            
            historyPopulation[f"iter "+str(iter).zfill(digit)]=item
        
        historyBest = {}
        for key in self.historyBestDecs.keys():
            
            bestDecs = self.historyBestDecs[key]
            bestObjs = self.historyBestObjs[key]
            iter = self.historyFEs[key]
            
            if alghType == 0:
                item = {"FEs" : key, "Best Decisions" : bestDecs, "Best Objectives" : bestObjs}
            else:
                metrics = self.historyBestMetrics[key]
                item = {"FEs" : key, "Best Decisions" : bestDecs, "Best Objectives" : bestObjs, "HV": metrics}
            
            if self.historyBestCons[key] is not None:
                item['Best Constrains'] = self.historyBestCons[key]
             
            historyBest[f"iter "+str(iter).zfill(digit)]=item
        
        globalBest={}
        globalBest["Best Decisions"] = self.bestDecs
        globalBest["Best Objectives"] = self.bestTrueObjs
        if self.bestCons is not None:
            globalBest["Best Constrains"] = self.bestCons
        globalBest["FEs"] = self.appearFEs
        globalBest["Iter"] = self.appearIters
        
        result = {
            "History_Population" : historyPopulation,
            "History_Best" : historyBest,
            "Global_Best" : globalBest,
            "Max_Iter" : self.algorithm.iters,
            "Max_FEs" : self.algorithm.FEs }
        
        return result
        
    def reset(self):
        
        self.bestDecs = None
        self.bestObjs = None
        self.bestCons = None
        self.bestFeasible = False
        self.bestMetric = None
        self.appearFEs = None
        self.appearIters = None
        self.historyBestDecs.clear()
        self.historyBestObjs.clear()
        self.historyBestCons.clear()
        self.historyDecs.clear()
        self.historyObjs.clear()
        self.historyCons.clear()
        self.historyFEs.clear()
        self.historyBestMetrics.clear()