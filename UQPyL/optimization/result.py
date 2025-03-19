import numpy as np 

from .population import Population
from .metric import HV
class Result():

    def __init__(self, algorithm):

        #Current best solution - Algorithm
        self.bestDecs = None
        self.bestObjs = None
        self.bestCons = None
        self.bestMetric = None
        self.bestFeasible = False
        
        #Current best solution - reality
        self.bestTrueDecs = None
        self.bestTrueObjs = None
        
        #Current best solution appearance
        self.appearFEs = None
        self.appearIters = None
        
        #History Records - Algorithm
        self.historyBestDecs = {}
        self.historyBestObjs = {}
        self.historyBestCons = {}
        self.historyBestMetrics = {}
        
        #History Records - reality
        self.historyBestTrueDecs = {}
        self.historyBestTrueObjs = {}
        
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
            self._update_EA(pop, FEs, iter, problem)
        else:
            self._update_MOEA(pop, FEs, iter, problem)
        
        self._update_history(pop, FEs, iter, problem)
      
    def _update_EA(self, pop, FEs, iter, problem):
        
        opt = problem.opt
        
        #Obtain local optima solutions
        bestPop = pop.getBest(k=1)
        localBestDecs = bestPop.decs[0, :][np.newaxis, :]
        localBestObjs = bestPop.objs[0, :][np.newaxis, :]
        localBestCons = bestPop.cons[0, :][np.newaxis, :] if bestPop.cons is not None else None
        localBestFeasible = True if localBestCons is None else np.all(np.maximum(0, localBestCons) <= 0)
        
        # update global optima solutions
        if self.bestObjs is None or (
            (localBestFeasible and not self.bestFeasible) or
            (localBestFeasible == self.bestFeasible and localBestObjs < self.bestObjs)
        ):
            # for running algorithm
            self.bestDecs = localBestDecs
            self.bestObjs = localBestObjs
            
            # for displaying results
            self.bestTrueObjs = localBestObjs * opt
            self.bestTrueDecs = problem._transform_to_I_D(localBestDecs, IFlag = True, DFlag = True)
            
            # for running algorithm
            self.bestCons = localBestCons
            self.bestFeasible = localBestFeasible
            
            self.appearFEs = FEs
            self.appearIters = iter
        
            # tolerate
            if self.bestObjs - localBestObjs > self.algorithm.tolerate: 
                self.algorithm.tolerateTimes = 0
            else:
                self.algorithm.tolerateTimes += 1
        else:
            self.algorithm.tolerateTimes += 1
    
    def _update_MOEA(self, pop, FEs, iter, problem):
        
        opt = problem.opt
        
        bestPop = pop.getBest()
        localBestDecs = bestPop.decs
        localBestObjs = bestPop.objs
        localBestCons = bestPop.cons if bestPop.cons is not None else None
        localBestFeasible = True if localBestCons is None else np.all(np.maximum(0, localBestCons) <= 0)
        
        self.bestDecs = localBestDecs
        self.bestObjs = localBestObjs
        
        self.bestTrueObjs = localBestObjs * opt
        self.bestTrueDecs = problem._transform_to_I_D(localBestDecs, IFlag = False, DFlag = False)
        
        self.bestCons = localBestCons
        self.bestFeasible = localBestFeasible
        
        localHV = HV(pop, refPoint = np.max(pop.objs, axis=0) * 1.1)
        self.bestMetric = localHV
        self.historyBestMetrics[FEs] = self.bestMetric
        
        self.appearFEs = FEs
        self.appearIters = iter
        
        if localHV - self.bestMetric > self.algorithm.tolerate:
            self.algorithm.tolerateTimes = 0
        else:
            self.algorithm.tolerateTimes += 1
        
    def _update_history(self, pop, FEs, iters, problem):
        
        opt = problem.opt
        
        self.historyDecs[FEs] = pop.decs
        self.historyObjs[FEs] = pop.objs * opt
        self.historyCons[FEs] = pop.cons
        self.historyFEs[FEs] = iters
        
        self.historyBestDecs[FEs] = self.bestDecs
        self.historyBestObjs[FEs] = self.bestObjs
        self.historyBestCons[FEs] = self.bestCons
        
        self.historyBestTrueDecs[FEs] = self.bestTrueDecs
        self.historyBestTrueObjs[FEs] = self.bestTrueObjs
    
    def generateHDF5(self):
        
        alghType = 1 if self.algorithm.problem.nOutput > 1 else 0
        
        historyPopulation = {}
        historyPopulation_True = {}
        
        digit = len(str(abs(self.algorithm.iters)))
        
        for key in self.historyDecs.keys():
            
            decs = self.historyDecs[key]
            objs = self.historyObjs[key]
            iter = self.historyFEs[key]
            
            decs_True = self.historyBestTrueDecs[key]
            objs_True = self.historyBestTrueObjs[key]
            
            item = {"FEs" : key , "Decisions" : decs, "Objectives" : objs}

            item_True = {"FEs" : key , "Decisions" : decs_True, "Objectives" : objs_True}
            
            if self.historyBestCons[key] is not None:
                item['Constrains'] = self.historyBestCons[key]
                item_True['Constrains'] = self.historyBestCons[key]
            
            historyPopulation[f"iter "+str(iter).zfill(digit)]=item
            historyPopulation_True[f"iter "+str(iter).zfill(digit)]=item_True
        
        historyBest = {}
        historyBest_True = {}       
        for key in self.historyBestDecs.keys():
            
            bestDecs = self.historyBestDecs[key]
            bestObjs = self.historyBestObjs[key]
            bestDecs_True = self.historyBestTrueDecs[key]
            bestObjs_True = self.historyBestTrueObjs[key]
            
            iter = self.historyFEs[key]
            
            if alghType == 0:
                item = {"FEs" : key, "Best Decisions" : bestDecs, "Best Objectives" : bestObjs}
                item_True = {"FEs" : key, "Best Decisions" : bestDecs_True, "Best Objectives" : bestObjs_True}
            else:
                metrics = self.historyBestMetrics[key]
                item = {"FEs" : key, "Best Decisions" : bestDecs, "Best Objectives" : bestObjs, "HV": metrics}
                item_True = {"FEs" : key, "Best Decisions" : bestDecs_True, "Best Objectives" : bestObjs_True, "HV": metrics}
            
            if self.historyBestCons[key] is not None:
                item['Best Constrains'] = self.historyBestCons[key]
                item_True['Best Constrains'] = self.historyBestCons[key]
             
            historyBest[f"iter "+str(iter).zfill(digit)]=item
            historyBest_True[f"iter "+str(iter).zfill(digit)]=item_True
            
        #global best record
        globalBest={}
        globalBest["Best Decisions"] = self.bestDecs
        globalBest["Best Objectives"] = self.bestTrueObjs
        
        globalBest["Best True Decisions"] = self.bestTrueDecs
        globalBest["Best True Objectives"] = self.bestTrueObjs
        
        if self.bestCons is not None:
            globalBest["Best Constrains"] = self.bestCons
            
        globalBest["FEs"] = self.appearFEs
        globalBest["Iter"] = self.appearIters
        
        result = {
            "History_Population_Algorithm" : historyPopulation,
            "History_Best_Algorithm" : historyBest,
            "History_Population_Reality" : historyPopulation_True,
            "History_Best_Reality" : historyBest_True,
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
        self.historyBestTrueDecs.clear()
        self.historyBestTrueObjs.clear()
        self.historyDecs.clear()
        self.historyObjs.clear()
        self.historyCons.clear()
        self.historyFEs.clear()
        self.historyBestMetrics.clear()