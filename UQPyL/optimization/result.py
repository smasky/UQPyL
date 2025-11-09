import numpy as np 
from datetime import datetime
import xarray as xr

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
        self.bestDecs_True = None
        self.bestObjs_True = None
        
        #Current best solution appearance
        self.appearFEs = None
        self.appearIters = None
        
        #History Records - Algorithm
        self.historyBestDecs = []
        self.historyBestObjs = []
        self.historyBestCons = []
        self.historyBestMetrics = []
        
        #History Records - reality
        self.historyDecs_True = []
        self.historyObjs_True = []
        self.historyBestDecs_True = []
        self.historyBestObjs_True = []
        
        self.historyDecs = []
        self.historyObjs = []
        self.historyCons = []
        self.iterToFEs = [] # iter -> FEs
        
        self.algorithm = algorithm
        self.runtime = 0
        
    def update(self, pop: Population, problem, FEs, iter, algType):

        # if problem.encoding == 'mix':
        #     decs = np.copy(pop.decs)
        #     decs = problem._transform_discrete_var(decs)
        
        if algType == 'EA':
            self._update_EA(pop, FEs, iter, problem)
        else:
            self._update_MOEA(pop, FEs, iter, problem)
        
        self._update_history(pop, FEs, iter, problem)
      
    def _update_EA(self, pop, FEs, iter, problem):
          
        #Obtain local optima solutions
        bestPop = pop.getBest(k = 1)
        localBestDecs = bestPop.decs[0:1, :]
        localBestObjs = bestPop.objs[0:1, :]
        localBestCons = bestPop.cons[0:1, :] if bestPop.cons is not None else None
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
            self.bestObjs_True = localBestObjs * problem.opt  # TODO: min
            self.bestDecs_True = problem._transform_to_I_D(localBestDecs, IFlag = True, DFlag = True)
            
            # for running algorithm
            self.bestCons = localBestCons
            self.bestFeasible = localBestFeasible
            
            self.appearFEs = FEs
            self.appearIters = iter
    
    def _update_MOEA(self, pop, FEs, iter, problem):
             
        bestPop = pop.getBest()
        localBestDecs = bestPop.decs
        localBestObjs = bestPop.objs
        localBestCons = bestPop.cons if bestPop.cons is not None else None
        localBestFeasible = True if localBestCons is None else np.all(np.maximum(0, localBestCons) <= 0)
        
        self.bestDecs = localBestDecs
        self.bestObjs = localBestObjs
        
        self.bestObjs_True = localBestObjs * problem.opt # TODO: min
        self.bestDecs_True = problem._transform_to_I_D(localBestDecs, IFlag = True, DFlag = True)
        
        self.bestCons = localBestCons
        self.bestFeasible = localBestFeasible
        
        localHV = HV(localBestObjs, refPoint = np.max(localBestObjs, axis=0) * 1.1)
        
        self.bestMetric = localHV
        self.historyBestMetrics.append(self.bestMetric)
        
        self.appearFEs = FEs
        self.appearIters = iter
        
    def _update_history(self, pop, FEs, iters, problem):
        
        self.historyDecs.append(pop.decs)
        self.historyObjs.append(pop.objs)
        self.historyCons.append(pop.cons) if pop.cons is not None else None
        
        self.historyDecs_True.append(problem._transform_to_I_D(pop.decs, IFlag = True, DFlag = True))
        self.historyObjs_True.append(pop.objs * problem.opt)
        
        self.iterToFEs.append([iters, FEs])
        
        self.historyBestDecs.append(self.bestDecs_True)
        self.historyBestObjs.append(self.bestObjs_True)
        self.historyBestCons.append(self.bestCons) if self.bestCons is not None else None
    
    def generateNetCDF(self):
        
        algType = 1 if self.algorithm.problem.nOutput > 1 else 0
        
        nInput = self.algorithm.problem.nInput
        nOutput = self.algorithm.problem.nOutput
        nCons = self.algorithm.problem.nCons
        iters = self.algorithm.iters
        
        # History
        
        historyDecs_ = np.vstack(self.historyDecs)
        historyObjs_ = np.vstack(self.historyObjs)
        
        historyDecs = np.vstack(self.historyDecs_True)
        historyObjs = np.vstack(self.historyObjs_True)
        historyCons = np.vstack(self.historyCons) if len(self.historyCons) > 0 else None
        
        iterToFEs = np.array(self.iterToFEs)
        
        iters = iterToFEs[:, 0].astype(int)
        FEs = iterToFEs[:, 1].astype(int)
        
        counts = np.array([d.shape[0] for d in self.historyDecs])
        iterStart = np.concatenate([[0], np.cumsum(counts)[:-1]])
        
        history = xr.Dataset(
            
            data_vars={
                "decs": (("idx", "nI"), historyDecs, {"description": "Decisions transformed to reality"}),
                "objs": (("idx", "nO"), historyObjs, {"description": "Objectives transformed to reality"}),
                **({"cons": (("idx", "nC"), historyCons)} if historyCons is not None else {}),
                "decs_": (("idx", "nI"), historyDecs_, {"description": "Decisions within algorithm"}),
                "objs_": (("idx", "nO"), historyObjs_, {"description": "Objectives within algorithm"}),
            },
            
            coords={
                    "idx" : ("idx", np.arange(historyDecs.shape[0]), {"description": "Global index across all iterations"}),
                    "nI": ("nI", np.arange(nInput),  {"description": "Index of decision variables (input dimensions)"}), 
                    "nO": ("nO", np.arange(nOutput), {"description": "Index of objective variables (output dimensions)"}), 
                    "nC": ("nC", np.arange(nCons), {"description": "Index of constraint variables (constraint dimensions)"}),
                    "iter": ("iter", iters, {"description": "Iteration index"}),
                    "fe": ("iter", FEs, {"description": "Function evaluation count corresponding to each iteration"}),
                    "start": ("iter", iterStart, {"description": "Starting index in 'idx' for this iteration's data slice"}), 
                    "length": ("iter", counts, {"description": "Number of samples in this iteration's data slice"}),
                },
            
            attrs={"algorithm": self.algorithm.name, "problem": self.algorithm.problem.name},
        )
        
        # Result
        
        historyBestDecs = np.vstack(self.historyBestDecs)
        historyBestObjs = np.vstack(self.historyBestObjs)
        historyBestCons = np.vstack(self.historyBestCons) if len(self.historyBestCons) > 0 else None
        
        counts = np.array([d.shape[0] for d in self.historyBestDecs])
        iterStart = np.concatenate([[0], np.cumsum(counts)[:-1]])
        
        idx1 = self.bestDecs_True.shape[0]
        idx2 = historyBestDecs.shape[0]
        
        if nOutput > 1:  
            bestMetric = np.array(self.historyBestMetrics)
               
        result = xr.Dataset(
            data_vars = {
                
                "bestDecs": (("idx1", "nI"), self.bestDecs_True, {"description": "Best decisions in reality"}),
                "bestObjs": (("idx1", "nO"), self.bestObjs_True, {"description": "Best objectives in reality"}),
                **({"bestCons": (("idx1", "nC"), self.bestCons)} if self.bestCons is not None else {}),

                "bestDecs_Iter": (("idx2", "nI"), historyBestDecs),
                "bestObjs_Iter": (("idx2", "nO"), historyBestObjs),
                **({"bestCons_Iter": (("idx2", "nC"), historyBestCons)} if historyBestCons is not None else {}),
            
                **({"bestMetric": (("iter"), bestMetric)} if nOutput > 1 else {}),
            },
            
            coords = {
                "nI": ("nI", np.arange(nInput), {"description": "Index of decision variables (input dimensions)"}),
                "nO": ("nO", np.arange(nOutput), {"description": "Index of objective variables (output dimensions)"}),
                "iter": ("iter", iters, {"description": "Iteration index"}),
                "fe": ("iter", FEs, {"description": "Function evaluation count corresponding to each iteration"}),
                "start": ("iter", iterStart, {"description": "Starting index in 'idx' for this iteration's data slice"}), 
                "length": ("iter", counts, {"description": "Number of samples in this iteration's data slice"}),
                **({"nC": ("nC", np.arange(nCons), {"description": "Index of constraint variables (constraint dimensions)"})} if nCons > 0 else {}),
                "idx1": ("idx1", np.arange(idx1), {"description": "Global index across all iterations"}),          
                "idx2": ("idx2", np.arange(idx2), {"description": "Iteration index"}),
            },
        )
        
        infos = xr.Dataset(
            attrs = {
                "algorithm": self.algorithm.name,
                "problem": self.algorithm.problem.name,
                "maxIter": self.algorithm.iters,
                "maxFEs": self.algorithm.FEs,
                "runtime": f"{self.runtime:.2f}",
                "nInput": nInput,
                "nOutput": nOutput,
                **({"nCons": nCons} if nCons > 0 else {}),
                **self.algorithm.setting.dicts,
                "created": datetime.now().isoformat(timespec='seconds'),
            }
        )
        
        history.attrs.update(infos.attrs)
        result.attrs.update(infos.attrs)
        
        res = {"history": history, "result": result}
        
        return res
        
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
        self.historyBestDecs_True.clear()
        self.historyBestObjs_True.clear()
        self.historyDecs.clear()
        self.historyObjs.clear()
        self.historyCons.clear()
        self.historyDecs_True.clear()
        self.historyObjs_True.clear()
        self.iterToFEs.clear()
        self.historyBestMetrics.clear()