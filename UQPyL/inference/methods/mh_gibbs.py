import numpy as np
from typing import Literal, Union

from ..base import InferenceABC
from ...problem import ProblemABC


class MH_Gibbs(InferenceABC):
    """
    Metropolis-Hastings Gibbs Inference
    Coordinate-wise Metropolis-Hastings sampling with Gibbs-style updates.

    Examples:
        >>> mh_gibbs = MH_Gibbs(nChains=4, warmUp=500, maxIters=2000)
        >>> res = mh_gibbs.run(problem, gamma=0.1, seed=1234)
        >>> print(res.bestDecs)

    References:
        [1] S. Geman and D. Geman, Stochastic relaxation, Gibbs distributions, and the
            Bayesian restoration of images, IEEE Transactions on Pattern Analysis and
            Machine Intelligence, vol. PAMI-6, no. 6, pp. 721-741, 1984.
        [2] W. K. Hastings, Monte Carlo sampling methods using Markov chains and their applications,
            Biometrika, vol. 57, no. 1, pp. 97-109, 1970.
    """
    
    name = "MH-Gibbs"
    
    def __init__(self, nChains: int = 1, warmUp: int = 1000, maxIters: int = 1000, 
                       propDist: Literal['gauss', 'uniform'] = 'gauss',
                       verboseFlag: bool = True, verboseFreq: int = 10,
                       logFlag: bool = False, saveFlag: bool = True,
                       saveFreq: int = 100, logProbFunc=None,
                       maxInitAttempts: int = 1000):
        """
        Initialize the MH-within-Gibbs inference method.

        Args:
            nChains: Number of sampling chains.
            warmUp: Number of warm-up iterations before formal sampling.
            maxIters: Number of formal sampling draws.
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
            maxIters, verboseFlag, verboseFreq, logFlag, saveFlag,
            saveFreq, logProbFunc, maxInitAttempts,
        )
                
        self.set('nChains', nChains)
        self.set('warmUp', warmUp)
        self.set('propDist', propDist)
        
        if propDist not in ['gauss', 'uniform']:
            raise ValueError("propDist must be 'gauss' or 'uniform'")
    
    def run(self, problem: ProblemABC, gamma: Union[float, np.ndarray, list] = 0.1, seed: int = None):
        
        self.setup(problem, seed)
        
        nChains = self.get('nChains'); warmUp = self.get('warmUp')
        propDist = self.get('propDist')
        
        X_init, Objs_init, Cons_init = self.initialSampling(problem, nChains, seed)
        
        chains = self.initChains(nChains, X_init, Objs_init, Cons_init)        
        
        gamma = self._check_gamma_(gamma)
        propCovs = []
        
        for i in range(nChains):
            cov = (gamma[i] * (problem.ub - problem.lb))**2
            propCovs.append(np.diag(cov.ravel()))
        
        X_cur = X_init; Objs_cur = Objs_init; Cons_cur = Cons_init
        
        wI = 0
        for _ in range(warmUp):
            
            dim = wI % problem.nInput
            
            X_star = self.f_prop(dim, X_cur, propDist, propCovs, problem.ub, problem.lb)
            
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
            
            wI += 1
        
        chains = self.initChains(nChains, X_cur, Objs_cur, Cons_cur)
        self.update(chains)
        while self.checkTermination(chains):
            
            dim = self.iter % problem.nInput
            
            X_star = self.f_prop(dim, X_cur, propDist, propCovs, problem.ub, problem.lb)
            
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

            self.update(chains)

        return self.finalize()
        
    def f_prop(self, dim, X_cur, propDist, propCovs, ub, lb):
        
        X_star = X_cur.copy()
        
        for i in range(X_cur.shape[0]):
            
            if propDist == 'gauss':
                X_star[i, dim] = self.rng.normal(X_cur[i][dim], propCovs[i].diagonal()[dim])
                
            elif propDist == 'uniform':
                X_star[i, dim] = self.rng.uniform(X_cur[i][dim] - propCovs[i].diagonal()[dim], X_cur[i][dim] + propCovs[i].diagonal()[dim])
            
            else:
                raise ValueError("propDist must be 'gauss' or 'uniform'")
        
        return self._check_bound_(X_star, ub, lb)
        
    def setProblem(self, problem: ProblemABC):
        
        self.problem = problem
    
    def _check_gamma_(self, gamma):
        
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
    
