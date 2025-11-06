import numpy as np
import xarray as xr
from typing import Literal, Union

from .base import InferenceABC
from ..problem import ProblemABC
from ..doe import LHS
from ..util.verbose import Verbose


class DEMC(InferenceABC):
    
    # Differential Evolution Markov Chain
    # ---------------------------------- #
    
    name = "Differential Evolution Markov Chain"
    
    def __init__(self,  nChain: int = 1, warmUp: int = 1000, 
                        maxIterTimes: int = 1000, 
                        verboseFlag: bool = True, verboseFreq: int = 10,
                        logFlag: bool = False, saveFlag: bool = True):
        
        super().__init__(maxIterTimes, verboseFlag, verboseFreq, logFlag, saveFlag)
                
        self.setParaVal('nChain', nChain)
        self.setParaVal('warmUp', warmUp)
    
    @Verbose.inference
    def run(self, problem: ProblemABC, gamma: Union[float, np.ndarray] = None, seed: int = None):
        
        self.setup(problem, seed)
        
        nChain = self.getParaVal('nChain'); warmUp = self.getParaVal('warmUp')
        
        X_init, Objs_init, Cons_init = self.initialSampling(problem, nChain)
        
        chains = self.initChains(nChain, X_init, Objs_init, Cons_init)
        
        ac_rate = np.zeros(nChain)
        
        X_cur = X_init; Objs_cur = Objs_init; Cons_cur = Cons_init
        
        if gamma is not None:
            gamma = self._check_gamma_(gamma)
        
        # warm up
        for _ in range(warmUp):
            
            X_star = self.f_prop(X_cur, problem.ub, problem.lb, gamma)
            
            Objs_star, Cons_star = self.evaluate(X_star)
            
            for i in range(nChain):
                
                if np.log(np.random.rand()) < (self.log_prob(Objs_star[i]) - self.log_prob(Objs_cur[i])) \
                    and (problem.nCons == 0 or (problem.nCons > 0 and all(Cons_star[i] <= 0))):
                    
                    X_cur[i] = X_star[i]; Objs_cur[i] = Objs_star[i]
                    
                    if problem.nCons > 0:
                        Cons_cur[i] = Cons_star[i]
                    
        # main loop
        ac_rate = np.zeros(nChain)
        while self.checkTermination(chains):
            
            X_star = self.f_prop(X_cur, problem.ub, problem.lb, gamma)
            
            Objs_star, Cons_star = self.evaluate(X_star)
            
            for i, chain in enumerate(chains):
                
                if np.log(np.random.rand()) < (self.log_prob(Objs_star[i]) - self.log_prob(Objs_cur[i])) \
                    and (problem.nCons == 0 or (problem.nCons > 0 and all(Cons_star[i] <= 0))):
                    
                    ac_rate[i] += 1 / self.maxIter
                    
                    X_cur[i] = X_star[i]; Objs_cur[i] = Objs_star[i]
                    
                    if problem.nCons > 0:
                        Cons_cur[i] = Cons_star[i]
                    
                chain.add(X_cur[i], Objs_cur[i], Cons_cur[i] if problem.nCons > 0 else None)
                    
        # generate basic result
        res = self.genNetCDF(chains, problem)
        
        # add extra results to basic result
        mean_ac_rate = np.mean(ac_rate)
        
        stats_ds = xr.Dataset(
            data_vars = {
                "acceptanceRate" : (("chain",), ac_rate, {"long_name": "acceptance rate for each chain",  "description": "acceptance rate for each chain"}),
                "acceptanceRate_mean" : ((), mean_ac_rate, {"long_name": "mean of acceptance rate for all chains",  "description": "mean of acceptance rate for all chains"})
            },
            coords = {
                "chain": np.arange(nChain),
            },
        )
        
        objs = res['posterior']['objs'].data
        lg = self.log_prob(objs)
        posterior_ds = xr.Dataset(
            data_vars = {
                "lg" : (("chain", "draw", "objsDim"), lg, {"long_name": "log probability",  "description": "log probability for each sample"}),
            },
            coords = {
                "chain": np.arange(nChain),
                "draw": np.arange(self.iter),
                "objsDim": np.arange(self.problem.nOutput),
            },
        )
        
        res['stats'] = xr.merge([res['stats'], stats_ds])
        
        res['posterior'] = xr.merge([res['posterior'], posterior_ds])
        
        return res
    
    def _check_alpha(self, alpha):
        
        nChain = self.getParaVal('nChain')
        nInput = self.problem.nInput
        
        if isinstance(alpha, float):
            alpha = np.full((nChain, nInput), alpha)
            
        elif isinstance(alpha, np.ndarray):
            alpha = np.atleast_2d(alpha)
            n, _ = alpha.shape
            if n == 1:
                alpha = np.tile(alpha, (nChain, 1))
            elif n == nChain:
                alpha = alpha
            else:
                raise ValueError("The shape of alpha must be (nChain, nInput) or (1, nInput)")
        else:
            raise ValueError("alpha must be a float or a numpy array")
        
        return alpha
    
    
    def check_bound(self, X, ub, lb):
        
        span = ub - lb
        y = (X - lb) % (2 * span)
        y = np.where(y > span, 2 * span - y, y)
        X_reflect = lb + y
        
        return X_reflect
    
    def f_prop(self, X_cur, ub, lb, gamma = None):
        
        X_star = np.zeros_like(X_cur)
        
        nChain = X_cur.shape[0]
        
        if gamma is None:
            gamma = np.full(nChain, 2.38 / np.sqrt(2 * self.problem.nInput))
        
        
        for i in range(nChain):
            
            idx = [j for j in range(nChain) if j != i]
            j, k = np.random.choice(idx, 2, replace=False)
            
            X_star[i] = X_cur[i] + gamma[i] * (X_cur[j] - X_cur[k]) + 1e-6 * gamma[i]
        
        return self.check_bound(X_star, ub, lb)
        
    def setProblem(self, problem: ProblemABC):
        
        self.problem = problem
        
    
    def log_prob(self, y):
    
        return -y