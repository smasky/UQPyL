import numpy as np
import xarray as xr
from typing import Literal, Union

from .base import InferenceABC
from ..problem import ProblemABC
from ..doe import LHS
from ..util.verbose import Verbose

class AMH(InferenceABC):
    
    # Adaptive Metropolis-Hastings
    # ---------------------------- #
    
    name = "AMH"
    
    def __init__(self, nChains: int = 1, warmUp: int = 1000, maxIterTimes: int = 1000, 
                       propDist: Literal['gauss', 'uniform'] = 'gauss',
                       verboseFlag: bool = True, verboseFreq: int = 10,
                       logFlag: bool = False, saveFlag: bool = True):
        super().__init__(maxIterTimes, verboseFlag, verboseFreq, logFlag, saveFlag)
                
        self.setParaVal('nChains', nChains)
        self.setParaVal('warmUp', warmUp)
        self.setParaVal('propDist', propDist)
        
        if propDist not in ['gauss', 'uniform']:
            raise ValueError("propDist must be 'gauss' or 'uniform'")
    
    @Verbose.inference
    def run(self, problem: ProblemABC, gamma: Union[float, np.ndarray] = 0.1, seed: int = None):
        
        if problem.nOutput > 1:
            raise ValueError("This AMH can only handle single-objective problems")
        
        self.setup(problem, seed)
        
        nChains = self.getParaVal('nChains')
        warmUp = self.getParaVal('warmUp')
        propDist = self.getParaVal('propDist')
        self.setParaVal('gamma', gamma)
        
        gamma = self._check_gamma_(gamma)
        sd = 2.38**2 / problem.nInput
        
        X_init, Objs_init, Cons_init = self.initialSampling(problem, nChains)
        
        chains = self.initChains(nChains, X_init, Objs_init, Cons_init)
        
        # calculate proposal covariance matrix
        gamma = self._check_gamma_(gamma)
        propCovs = []
        
        for i in range(nChains):
            cov = (gamma[i] * (problem.ub - problem.lb))**2
            propCovs.append(np.diag(cov.ravel()))
        
        ac_rate = np.zeros(nChains)
        
        X_cur = X_init; Objs_cur = Objs_init; Cons_cur = Cons_init
        
        for _ in range(warmUp):
            
            X_star = self.f_prop(X_cur, propDist, propCovs, problem.ub, problem.lb)
            
            Objs_star, Cons_star = self.evaluate(X_star)
            
            for i in range(nChains):
                
                if np.log(np.random.rand()) < (self.log_prob(Objs_star[i]) - self.log_prob(Objs_cur[i])) \
                    and (problem.nCons == 0 or (problem.nCons > 0 and all(Cons_star[i] <= 0))):
                    
                    X_cur[i] = X_star[i]; Objs_cur[i] = Objs_star[i]
                    
                    if problem.nCons > 0:
                        Cons_cur[i] = Cons_star[i]
                
        while self.checkTermination(chains):
            
            X_star = self.f_prop(X_cur, propDist, propCovs, problem.ub, problem.lb)
            
            Objs_star, Cons_star = self.evaluate(X_star)
            
            for i, chain in enumerate(chains):
                
                if np.log(np.random.rand()) < (self.log_prob(Objs_star[i]) - self.log_prob(Objs_cur[i])) \
                    and (problem.nCons == 0 or (problem.nCons > 0 and all(Cons_star[i] <= 0))):
                    
                    ac_rate[i] += 1 / self.maxIters
                    
                    X_cur[i] = X_star[i]; Objs_cur[i] = Objs_star[i]
                    
                    if problem.nCons > 0:
                        Cons_cur[i] = Cons_star[i]
                
                chain.add(X_cur[i], Objs_cur[i], Cons_cur[i] if problem.nCons > 0 else None)
                
            # update proposal covariance matrix
            propCovs = self.updateCovs(chains, sd)
        
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
                "chain": np.arange(nChains),
            },
        )
        
        objs = res['posterior']['objs'].data
        lg = self.log_prob(objs)
        posterior_ds = xr.Dataset(
            data_vars = {
                "lg" : (("chain", "draw", "objsDim"), lg, {"long_name": "log probability",  "description": "log probability for each sample"}),
            },
            coords = {
                "chain": np.arange(nChains),
                "draw": np.arange(self.iter),
                "objsDim": np.arange(self.problem.nOutput),
            },
        )
        
        res['stats'] = xr.merge([res['stats'], stats_ds])
        
        res['posterior'] = xr.merge([res['posterior'], posterior_ds])
        
        return res
    
    def updateCovs(self, chains, sd):
        
        propCovs = []
        
        for chain in chains:
            propCovs.append((np.cov(chain.decs[:chain.count].T) * sd + 1e-6 * np.eye(chain.decs.shape[1]) * sd ))
        
        return propCovs
    
    def _check_alpha(self, gamma):
        
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
        
        return gamma
    
    
    def check_bound(self, X, ub, lb):
        
        span = ub - lb
        y = (X - lb) % (2 * span)
        y = np.where(y > span, 2 * span - y, y)
        X_reflect = lb + y
        
        return X_reflect
    
    def f_prop(self, X_cur, propDist, propCovs, ub, lb):
        
        X_star = np.zeros(X_cur.shape)
        
        for i in range(X_cur.shape[0]):
            if propDist == 'gauss':
                X_star[i] = np.random.multivariate_normal(X_cur[i].ravel(), propCovs[i])
            elif propDist == 'uniform':
                X_star[i] = np.random.uniform(X_cur[i] - propCovs[i].diagonal(), X_cur[i] + propCovs[i].diagonal())
            else:
                raise ValueError("propDist must be 'gauss' or 'uniform'")
        
        return self.check_bound(X_star, ub, lb)
        
    def setProblem(self, problem: ProblemABC):
        
        self.problem = problem
        
    
    def log_prob(self, y):
    
        return -y
        