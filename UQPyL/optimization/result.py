import numpy as np 

from .population import Population
from .metric import HV, IGD
class Result():
    
    def __init__(self, algorithm):
        
        self.bestDec = None
        self.bestObj = None
        self.appearFEs = None
        self.appearIters = None
        self.historyBestDecs = {}
        self.historyBestObjs = {}
        self.historyBestCons = {}
        self.historyDecs = {}
        self.historyObjs = {}
        self.historyCons = {}
        self.historyFEs = {}
        self.historyBestMetrics = {}
        
        self.algorithm = algorithm
        
    def update(self, pop: Population, problem, FEs, iter, algType):
        
        decs = np.copy(pop.decs)
        
        if problem.encoding=='mix':
            decs = problem._transform_discrete_var(pop.decs)
            
        objs = np.copy(pop.objs)
        
        cons = np.copy(pop.cons) if pop.cons is not None else None
        
        if algType =='EA':
            
            if self.bestObj == None or np.min(objs)<self.bestObj:
                iMin = np.argmin(objs)
                self.bestDec = decs[iMin, :]
                self.bestObj = objs[iMin, :]
                self.bestCon = cons[iMin, :] if cons is not None else None
                self.appearFEs = FEs
                self.appearIters = iter
                
            self.historyFEs[FEs] = iter
            self.historyDecs[FEs] = decs
            self.historyObjs[FEs] = objs
            self.historyCons[FEs] = cons
            self.historyBestDecs[FEs] = self.bestDec
            self.historyBestObjs[FEs] = self.bestObj
            self.historyBestCons[FEs] = self.bestCon
            self.historyCons[FEs] = self.bestCon
            
        else:
            
            bests = pop.getBest()
            
            bestDecs = np.copy(bests.decs)
            
            bestCons = np.copy(bests.cons) if bests.cons is not None else None
            
            if problem.encoding == 'mix':
                decs = problem._transform_discrete_var(np.copy(pop.decs))
            
            bestObjs = np.copy(bests.objs)
            
            optimum = self.algorithm.problem.getOptimum()
            
            igdValue = None; hvValue = None
            
            if optimum is not None:
                optimum = optimum[~np.isnan(optimum).any(axis=1)]
                igdValue = IGD(bests, optimum)
            
            hvValue = HV(bests)
            self.historyDecs[FEs] = decs
            self.historyObjs[FEs] = objs
            self.historyBestDecs[FEs] = bestDecs
            self.historyBestMetrics[FEs] = [[hvValue, igdValue] if igdValue is not None else [hvValue]]
            self.historyBestObjs[FEs] = bestObjs
            self.historyBestCons[FEs] = bestCons if bestCons is not None else None
            self.historyFEs[FEs] = iter
            self.bestDec = bestDecs
            self.bestObj = bestObjs
            self.bestCon = bestCons if bestCons is not None else None
            self.bestMetric = [hvValue, igdValue] if igdValue is not None else [hvValue]
            self.appearFEs = FEs
            self.appearIters = iter
        
    def generateHDF5(self):
        
        althType = 1 if self.algorithm.problem.nOutput>1 else 0
        
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
            
            if althType == 0:
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
        globalBest["Best Decisions"] = self.bestDec
        globalBest["Best Objectives"] = self.bestObj
        if self.bestCon is not None:
            globalBest["Best Constrains"] = self.bestCon
        globalBest["FEs"] = self.appearFEs
        globalBest["Iter"] = self.appearIters
        
        if althType == 1:
            if isinstance(self.bestMetric, tuple):
                globalBest["HV"] = self.bestMetric[0]
                globalBest["IGD"] = self.bestMetric[1]
            else:
                globalBest["HV"] = self.bestMetric
        
        result = { "History_Population" : historyPopulation,
                 "History_Best" : historyBest,
                 "Global_Best" : globalBest,
                 "Max_Iter" : self.algorithm.iters,
                 "Max_FEs" : self.algorithm.FEs }
        
        return result
        
    def reset(self):
        self.bestDec = None; self.bestObj = None
        self.appearFEs = None; self.appearIters = None
        self.historyBestDecs = {}; self.historyBestObjs = {}
        self.historyDecs = {}; self.historyObjs = {}
        self.historyFEs = {}; self.historyMetrics = {}