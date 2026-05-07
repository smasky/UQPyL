import numpy as np
from typing import Literal, Union

from ..base import InferenceABC
from ...problem import ProblemABC

class AMH(InferenceABC):
    """
    Adaptive Metropolis-Hastings Inference
    Metropolis-Hastings sampling with adaptive proposal covariance updates.

    Examples:
        >>> amh = AMH(nChains=4, warmUp=500, maxIterTimes=2000)
        >>> res = amh.run(problem, gamma=0.1, seed=1234)
        >>> print(res.acceptanceRate)

    References:
        [1] H. Haario, E. Saksman, and J. Tamminen, An adaptive Metropolis algorithm,
            Bernoulli, vol. 7, no. 2, pp. 223-242, 2001.
    """
    
    name = "AMH"
    
    def __init__(self, nChains: int = 1, warmUp: int = 1000, maxIterTimes: int = 1000, 
                       propDist: Literal['gauss', 'uniform'] = 'gauss',
                       verboseFlag: bool = True, verboseFreq: int = 10,
                       logFlag: bool = False, saveFlag: bool = True,
                       saveFreq: int = 100, logProbFunc=None,
                       maxInitAttempts: int = 1000):
        """
        Initialize the adaptive Metropolis-Hastings inference method.

        Args:
            nChains: Number of sampling chains.
            warmUp: Number of warm-up iterations before formal sampling.
            maxIterTimes: Number of formal sampling draws.
            propDist: Proposal distribution type.
            verboseFlag: Whether to print compact runtime summaries.
            verboseFreq: Iteration interval for terminal and log summaries.
            logFlag: Whether to write a log file.
            saveFlag: Whether to persist snapshots and final result to sqlite.
            saveFreq: Iteration interval for sqlite snapshots.
            logProbFunc: Optional custom log-probability function.
            maxInitAttempts: Maximum LHS batches used to find feasible initial chains.
        """
        super().__init__(
            maxIterTimes, verboseFlag, verboseFreq, logFlag, saveFlag,
            saveFreq, logProbFunc, maxInitAttempts,
        )
                
        self.set('nChains', nChains)
        self.set('warmUp', warmUp)
        self.set('propDist', propDist)
        
        if propDist not in ['gauss', 'uniform']:
            raise ValueError("propDist must be 'gauss' or 'uniform'")
    
    def run(self, problem: ProblemABC, gamma: Union[float, np.ndarray] = 0.1, seed: int = None):
        self.setup(problem, seed)
        
        nChains = self.get('nChains')
        warmUp = self.get('warmUp')
        propDist = self.get('propDist')
        self.set('gamma', gamma)
        
        gamma = self._check_gamma_(gamma)
        sd = 2.38**2 / problem.nInput
        
        X_init, Objs_init, Cons_init = self.initialSampling(problem, nChains)
        
        chains = self.initChains(nChains, X_init, Objs_init, Cons_init)
        
        gamma = self._check_gamma_(gamma)
        propCovs = []
        
        for i in range(nChains):
            cov = (gamma[i] * (problem.ub - problem.lb))**2
            propCovs.append(np.diag(cov.ravel()))
        
        X_cur = X_init; Objs_cur = Objs_init; Cons_cur = Cons_init
        
        for _ in range(warmUp):
            
            X_star = self.f_prop(X_cur, propDist, propCovs, problem.ub, problem.lb)
            
            Objs_star, Cons_star = self.evaluate(X_star)
            
            for i in range(nChains):
                
                if self.accept(
                    Objs_star[i], Objs_cur[i],
                    Cons_star[i] if problem.nCons > 0 else None,
                    decStar=X_star[i], decCur=X_cur[i],
                    consCur=Cons_cur[i] if problem.nCons > 0 else None,
                ):
                    
                    X_cur[i] = X_star[i]; Objs_cur[i] = Objs_star[i]
                    
                    if problem.nCons > 0:
                        Cons_cur[i] = Cons_star[i]
                
        chains = self.initChains(nChains, X_cur, Objs_cur, Cons_cur)
        self.update(chains)
        while self.checkTermination(chains):
            
            X_star = self.f_prop(X_cur, propDist, propCovs, problem.ub, problem.lb)
            
            Objs_star, Cons_star = self.evaluate(X_star)
            
            for i, chain in enumerate(chains):
                
                accepted = self.accept(
                    Objs_star[i], Objs_cur[i],
                    Cons_star[i] if problem.nCons > 0 else None,
                    decStar=X_star[i], decCur=X_cur[i],
                    consCur=Cons_cur[i] if problem.nCons > 0 else None,
                )
                if accepted:
                    X_cur[i] = X_star[i]; Objs_cur[i] = Objs_star[i]
                    
                    if problem.nCons > 0:
                        Cons_cur[i] = Cons_star[i]
                
                chain.add(
                    X_cur[i],
                    Objs_cur[i],
                    Cons_cur[i] if problem.nCons > 0 else None,
                    logProb=self.log_prob(Objs_cur[i], decs=X_cur[i], cons=Cons_cur[i] if problem.nCons > 0 else None),
                    accepted=accepted,
                )
                
            propCovs = self.updateCovs(chains, sd, propCovs)
            self.update(chains)

        return self.finalize()
    
    def updateCovs(self, chains, sd, currentCovs=None):
        
        propCovs = []
        
        for i, chain in enumerate(chains):
            if chain.count < 3:
                if currentCovs is not None:
                    propCovs.append(currentCovs[i].copy())
                else:
                    propCovs.append(np.eye(chain.decs.shape[1]) * sd)
                continue

            cov = np.cov(chain.decs[:chain.count].T) * sd
            floor = 1e-3 * np.eye(chain.decs.shape[1])
            propCovs.append(cov + floor)
        
        return propCovs
    
    def _check_alpha(self, gamma):
        
        nChains = self.get('nChains')
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
                X_star[i] = self.rng.multivariate_normal(X_cur[i].ravel(), propCovs[i])
            elif propDist == 'uniform':
                X_star[i] = self.rng.uniform(X_cur[i] - propCovs[i].diagonal(), X_cur[i] + propCovs[i].diagonal())
            else:
                raise ValueError("propDist must be 'gauss' or 'uniform'")
        
        return self.check_bound(X_star, ub, lb)
        
    def setProblem(self, problem: ProblemABC):
        
        self.problem = problem
        
    
