import numpy as np 

from .population import Population
class Result():
    
    def __init__(self, algorithm):
        
        #Current best solution
        self.bestDecs = None
        self.bestObjs = None
        self.bestCons = None
        self.bestFeasible = False
        
        #Current best solution appearance
        self.appearFEs = None
        self.appearIters = None
        
        #History Records
        self.historyBestDecs = {}
        self.historyBestObjs = {}
        self.historyBestCons = {}
        self.historyDecs = {}
        self.historyObjs = {}
        self.historyCons = {}
        self.historyFEs = {}
        
        self.algorithm = algorithm
        
    def update(self, pop: Population, problem, FEs, iter, algType):
        
        decs = np.copy(pop.decs)
        
        optType = problem.optType
        
        if problem.encoding == 'mix':
            decs = problem._transform_discrete_var(decs)
        
        if algType == 'EA':
            self._update_EA(pop, FEs, iter, optType)
        else:
            self._update_MOEA(pop, FEs, iter, optType)
        
        self._update_history(pop, FEs, iter, optType)
      
    def _update_EA(self, pop, FEs, iter, optType):
        
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
            self.bestTrueObjs = localBestObjs if optType == 'min' else -1*localBestObjs
            self.bestCons = localBestCons
            self.bestFeasible = localBestFeasible
            self.appearFEs = FEs
            self.appearIters = iter
    
    def _update_MOEA(self, pop, FEs, iter, optType):
        
        bestPop = pop.getBest()
        localBestDecs = bestPop.decs[0]
        localBestObjs = bestPop.objs[0]
        localBestCons = bestPop.cons[0] if bestPop.cons is not None else None
        localBestFeasible = True if localBestCons is None else np.all(np.maximum(0, localBestCons) <= 0)
        
        self.bestDecs = localBestDecs
        self.bestObjs = localBestObjs
        self.bestTrueObjs = localBestObjs if optType == 'min' else -1*localBestObjs
        self.bestCons = localBestCons
        self.bestFeasible = localBestFeasible
        self.appearFEs = FEs
        self.appearIters = iter
        
    def _update_history(self, pop, FEs, iters, optType):
        
        self.historyDecs[FEs] = pop.decs
        self.historyObjs[FEs] = pop.objs if optType == 'min' else -1*pop.objs
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
                if isinstance(metrics[0], tuple):
                    item = {"FEs" : key, "Best Decisions" : bestDecs, "Best Objectives" : bestObjs, "HV": metrics[0][0], "IGD": metrics[0][1]}
                else:
                    item = {"FEs" : key, "Best Decisions" : bestDecs, "Best Objectives" : bestObjs, "HV": metrics}
            
            if self.historyBestCons[key] is not None:
                item['Best Constrains'] = self.historyBestCons[key]
             
            historyBest[f"iter "+str(iter).zfill(digit)]=item
        
        globalBest={}
        globalBest["Best Decisions"] = self.bestDecs
        globalBest["Best Objectives"] = self.bestObjs
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