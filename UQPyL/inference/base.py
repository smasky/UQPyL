import abc
import numpy as np
from pandas.compat import F
import xarray as xr
from datetime import datetime

from .chain import Chain
from ..problem import ProblemABC
from ..doe import LHS
from ..util import Verbose

class InferenceABC(metaclass = abc.ABCMeta):
    
    def __init__(self, maxIters: int = 1000,
                 verboseFlag: bool = True, verboseFreq: int = 10, 
                 logFlag: bool = False, saveFlag: bool = False):
        
        self.verboseFlag = verboseFlag
        self.verboseFreq = verboseFreq
        self.logFlag = logFlag
        self.saveFlag = saveFlag
        
        self.maxIters = maxIters
        
        self.setting = Setting()
        
    def run(self):
        pass
    
    def setup(self, problem: ProblemABC, seed: int = None):
        
        # check problem type
        if problem.nOutput > 1:
            raise ValueError("This MH can only handle single-objective problems")
        
        self.reset()
        
        self.setProblem(problem)
        
        # set seed
        if seed is None:
            seed = np.random.randint(1, 1000000)
        self.setParaVal('seed', seed)
        np.random.seed(seed)
    
    def initialSampling(self, problem: ProblemABC, nChains: int, seed: int = None):
        
        sampler = LHS()
        sample_seed = np.random.randint(1, 1000000)
        X0 = sampler.sample(self.problem, nChains, sample_seed)
        Objs0, Cons0 = self.evaluate(X0)
        
        return X0, Objs0, Cons0
    
    def _check_bound_(self, X, ub, lb):
        
        span = ub - lb
        y = (X - lb) % (2 * span)
        y = np.where(y > span, 2 * span - y, y)
        X_reflect = lb + y
        
        return X_reflect
    
    def _check_gamma_(self, gamma):
        
        nChains = self.getParaVal('nChains')
        nInput = self.problem.nInput
        
        if isinstance(gamma, float):
            
            gamma = np.full((nChains, nInput), gamma)
            
        elif isinstance(gamma, np.ndarray):
            
            gamma = np.atleast_2d(gamma)
            
            n, _ = gamma.shape
            
            if n == 1:
                gamma = np.tile(gamma, (nChains, 1))
            elif n == nChains:
                gamma = gamma
            else:
                raise ValueError("The shape of gamma must be (nChains, nInput) or (1, nInput)")
        else:
            raise ValueError("gamma must be a float or a numpy array")
        
        return gamma.ravel()
    
    def setProblem(self, problem: ProblemABC):
        
        self.problem = problem
    
    def initChains(self, nChains: int, X: np.ndarray, Objs: np.ndarray, Cons: np.ndarray = None):
        
        nI = self.problem.nInput; nO = self.problem.nOutput; nC = self.problem.nCons
        
        iters = self.maxIters
        
        chains = []
        
        for _ in range(nChains):
            
            chains.append(Chain(nI, nO, nC, iters))
        
        for i in range(nChains):
            chains[i].add(X[i], Objs[i], Cons[i] if nC > 0 else None)
        
        return chains
    
    def checkTermination(self, chains):
        
        self.iter += 1
        
        if self.verboseFlag and self.iter % self.verboseFreq == 0:
            
            verbRes = self.generateVerb(chains)
            
            Verbose.verboseInference(verbRes, self.problem)
           
        if self.iter >= self.maxIters:
            return False
        
        return True 
    
    def generateVerb(self, chains):
        
        nO = self.problem.nOutput
        nC = self.problem.nCons

        decs = np.vstack([c.decs[:self.iter] for c in chains])
        objs = np.vstack([c.objs[:self.iter] for c in chains])
        objs_min = objs * self.problem.opt
        
        if nC > 0:
            cons = np.vstack([c.cons[:self.iter] for c in chains.values()])
            feasibleMask = (cons <= 0).all(axis = 1)
        else:
            feasibleMask = np.ones(decs.shape[0], dtype=bool)
        
        feasibleDecs = decs[feasibleMask]
        feasibleObjs = objs_min[feasibleMask]
        
        if nO == 1:
        
            # mean = np.mean(feasibleDecs, axis = 0)
            # std = np.std(feasibleDecs, axis = 0)
            
            bestDec = feasibleDecs[np.argmin(feasibleObjs)]
            bestObj = np.min(feasibleObjs) * self.problem.opt
            
            verbRes = {"iter" : self.iter, "bestDecs" : bestDec, "bestObjs" : bestObj}
        
        # else: 
            
        #     numPareto = self.paretoSet["BestDecs"].shape[0]
            
        #     verbRes = {"iter" : self.iter, "numPareto" : numPareto}
            
        return verbRes
            
    def reset(self):
        
        self.iter = 0
    
    def evaluate(self, decs: np.ndarray):
        
        res = self.problem.evaluate(decs)
        
        return res['objs']*self.problem.opt, res['cons']

    def genNetCDF(self, chains, problem):
        
        res = {};
        
        nChains = len(chains); draw = chains[0].count
        
        nInput = problem.nInput; nOutput = problem.nOutput; nCons = problem.nCons
        
        decs = np.stack([c.decs for c in chains])
        objs = np.stack([c.objs for c in chains])
        objs_min = objs * problem.opt
        
        if nCons > 0:
            cons = np.stack([c.cons for c in chains])
            feasibleMask = (cons <= 0).all("consDim")
        else:
            feasibleMask = np.ones((nChains, decs.shape[1]), dtype=bool)
        
        posterior_ds = xr.Dataset(
            data_vars = {
                "decs" : (("chain", "draw", "decsDim"), decs, {"long_name": "decision variables / parameters",  "description": "posterior samples after burn-in in each chain"}),
                "objs" : (("chain", "draw", "objsDim"), objs, {"long_name": "objective values",  "description": "objective values for each sample"}),
                **({"cons" : (("chain", "draw", "consDim"), cons, {"long_name": "constraint values",  "description": "constraint values for each sample"})} if nCons > 0 else {}),
                "feasibleMask" : (("chain", "draw"), feasibleMask, {"long_name": "feasible mask",  "description": "feasible mask for each chain and each draw"}),
                "proUB" : (("decsDim"), problem.ub.ravel(), {"long_name": "upper bound of decision variables / parameters",  "description": "upper bound of decision variables / parameters"}),
                "proLB" : (("decsDim"), problem.lb.ravel(), {"long_name": "lower bound of decision variables / parameters",  "description": "lower bound of decision variables / parameters"}),
            },
            coords = {
                "chain": np.arange(nChains),
                "draw": np.arange(draw),
                "decsDim": np.arange(nInput),
                "objsDim": np.arange(nOutput),
                **({"consDim": np.arange(nCons)} if nCons > 0 else {}),
            },
            attrs = {
                "description": "Posterior samples",
                "problem": f"{problem.name}_{problem.nInput}D_{problem.nOutput}O_{problem.nCons}C",
                "method": self.name,
                "created": datetime.now().isoformat(timespec = 'seconds'),
                **self.setting.dicts,
            }
        )
        
        res["posterior"] = posterior_ds
        
        # statistics and optimization
        
        meanEvery = []
        stdEvery = []
        
        bestDecs = []
        bestObjs = []
        
        for i in range(nChains):
            
            decs_i = decs[i]
            objs_i = objs_min[i]
            feasible_i = feasibleMask[i]
            
            feasibleDecs_i = decs_i[feasible_i]
            feasibleObjs_i = objs_i[feasible_i]
            
            meanEvery.append(np.mean(feasibleDecs_i, axis = 0))
            stdEvery.append(np.std(feasibleDecs_i, axis = 0))
            
            if problem.nOutput == 1:
                idx_min = np.argmin(feasibleObjs_i)
                bestDecs.append(feasibleDecs_i[idx_min])
                bestObjs.append(feasibleObjs_i[idx_min] * problem.opt)
        
        # global infos
        
        countFeasible = feasibleMask.sum(axis = 1)
        weights = countFeasible[:, None] / countFeasible.sum()
        meanAll = np.sum(meanEvery * weights, axis = 0)
        stdAll = np.sum(stdEvery * weights, axis = 0)
        
        if problem.nOutput == 1:
            
            t = bestObjs * problem.opt
            idx_min = np.argmin(t)
            globalBestDec = bestDecs[idx_min]
            globalBestObj = t[idx_min] * problem.opt
            
        # else:
            
        #     from ..optimization.util import NDSort

        #     decs = decs.reshape(-1, decs.shape[2])
        #     objs_min = objs_min.reshape(-1, objs_min.shape[2])
        #     cons = cons.reshape(-1, cons.shape[2]) if nCons > 0 else None

        #     if nCons > 0:
        #         t = cons <= 0
        #         feasible = t.all(axis = 1)
        #     else:
        #         feasible = np.ones((cons.shape[0]), dtype = bool)
            
        #     feasibleDecs = decs[feasible]
        #     feasibleObjs = objs_min[feasible]
            
        #     frontNo, _ = NDSort(feasibleObjs)
        #     paretoDecs = feasibleDecs[frontNo == 1]
        #     paretoObjs = feasibleObjs[frontNo == 1] * problem.opt
            
        stats_ds = xr.Dataset(
            data_vars = {
                "meanEvery" : (("chain", "decsDim"), meanEvery, {"long_name": "mean of feasible decision variables for each chain",  "description": "mean of feasible decision variables for each chain"}),
                "stdEvery" : (("chain", "decsDim"), stdEvery, {"long_name": "standard deviation of feasible decision variables for each chain",  "description": "standard deviation of feasible decision variables for each chain"}),
                "meanAll" : (("decsDim"), meanAll, {"long_name": "mean of feasible decision variables for all chains",  "description": "mean of feasible decision variables for all chains"}),
                "stdAll" : (("decsDim"), stdAll, {"long_name": "standard deviation of feasible decision variables for all chains",  "description": "standard deviation of feasible decision variables for all chains"}),
            },
            coords = {
                "chain": np.arange(nChains),
                "decsDim": np.arange(nInput),
                "objsDim": np.arange(nOutput),
                **({"consDim": np.arange(nCons)} if nCons > 0 else {}),
            },
            attrs = {
                "description": "Statistics results",
                "problem": f"{problem.name}_{problem.nInput}D_{problem.nOutput}O_{problem.nCons}C",
                "method": self.name,
                "created": datetime.now().isoformat(timespec='seconds'),
                **self.setting.dicts,
            }
        )
        
        res["stats"] = stats_ds
        
        if problem.nOutput == 1:
            
            optimization_ds = xr.Dataset(
                data_vars = {
                    "localBestDecs" : (("chain", "decsDim"), bestDecs, {"long_name": "best decision variables / parameters for each chain",  "description": "best decision variables / parameters for each chain"}),
                    "localBestObjs" : (("chain", "objsDim"), bestObjs, {"long_name": "best objective values for each chain",  "description": "best objective values for each chain"}),
                    "globalBestDec" : (("decsDim"), globalBestDec, {"long_name": "global best decision variables / parameters",  "description": "global best decision variables / parameters"}),
                    "globalBestObj" : (("objsDim"), globalBestObj, {"long_name": "global best objective values",  "description": "global best objective values"}),
                },
                
                coords = {
                    "chain": np.arange(nChains),
                    "decsDim": np.arange(nInput),
                    "objsDim": np.arange(nOutput),
                },
                attrs = {
                    "description": "Optimization results",
                    "problem": f"{problem.name}_{problem.nInput}D_{problem.nOutput}O_{problem.nCons}C",
                    "method": self.name,
                    **self.setting.dicts
                }
            )
            
        # else:
        #     optimization_ds = xr.Dataset(
        #         data_vars = {
        #             "paretoDecs" : (("idx", "decsDim"), paretoDecs, {"long_name": "Pareto decision variables / parameters",  "description": "Pareto decision variables / parameters for each Pareto front"}),
        #             "paretoObjs" : (("idx", "objsDim"), paretoObjs, {"long_name": "Pareto objective values",  "description": "Pareto objective values for each Pareto front"}),
        #         },
                
        #         coords = {
        #             "idx": np.arange(bestObjs.shape[0]),
        #             "decsDim": np.arange(nInput),
        #             "objsDim": np.arange(nOutput),
        #         },
        #         attrs = {
        #             "description": "Optimization results",
        #             "problem": f"{problem.name}_{problem.nInput}D_{problem.nOutput}O_{problem.nCons}C",
        #             "problem_ub": problem.ub,
        #             "problem_lb": problem.lb,
        #             "method": self.name,
        #             **self.setting.dicts
        #         }
        #     )
        
        res["optimization"] = optimization_ds
            
        return res

    def setParaVal(self, key, value):
        
        self.setting.setPara(key, value)
    
    def getParaVal(self, *args):
        
        return self.setting.getVal(*args)
    
class Setting():
    """
    Save the parameter setting of the inference
    """
    
    def __init__(self):
        self.keys = []
        self.values = []
        self.dicts = {}
    
    def setPara(self, key, value):
        
        self.dicts[key] = value
        self.keys.append(key)
        self.values.append(value)
    
    def getVal(self, *args):
        
        values = []
        for arg in args:
            values.append(self.dicts[arg])
        
        if len(args) > 1:
            return tuple(values)
        else:
            return values[0]