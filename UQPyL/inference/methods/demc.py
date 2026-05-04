import numpy as np
from typing import Literal, Union

from ..base import InferenceABC
from ...problem import ProblemABC


class DEMC(InferenceABC):
    """
    Differential Evolution Markov Chain Inference
    Multi-chain differential evolution sampling for posterior exploration.

    Examples:
        >>> demc = DEMC(nChains=6, warmUp=500, maxIterTimes=2000)
        >>> res = demc.run(problem, seed=1234)
        >>> print(res.bestObjs)
    """
    
    name = "DEMC"
    
    def __init__(self,  nChains: int = 1, warmUp: int = 1000, 
                        maxIterTimes: int = 1000, 
                        verboseFlag: bool = True, verboseFreq: int = 10,
                        logFlag: bool = False, saveFlag: bool = True,
                        saveFreq: int = 100, logProbFunc=None,
                        maxInitAttempts: int = 1000):
        """
        Initialize the DEMC inference method.

        Args:
            nChains: Number of sampling chains.
            warmUp: Number of warm-up iterations before formal sampling.
            maxIterTimes: Number of formal sampling draws.
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
                
        self.setParaVal('nChains', nChains)
        self.setParaVal('warmUp', warmUp)
    
    def run(self, problem: ProblemABC, gamma: Union[float, np.ndarray] = None, seed: int = None):
        
        self.setup(problem, seed)
        
        nChains = self.getParaVal('nChains'); warmUp = self.getParaVal('warmUp')
        
        X_init, Objs_init, Cons_init = self.initialSampling(problem, nChains)
        
        chains = self.initChains(nChains, X_init, Objs_init, Cons_init)
        
        X_cur = X_init; Objs_cur = Objs_init; Cons_cur = Cons_init
        
        if gamma is not None:
            gamma = self._check_gamma_(gamma)
        
        for _ in range(warmUp):
            
            X_star = self.f_prop(X_cur, problem.ub, problem.lb, gamma)
            
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
            
            X_star = self.f_prop(X_cur, problem.ub, problem.lb, gamma)
            
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
    
    def _check_alpha(self, alpha):
        
        nChains = self.getParaVal('nChains')
        nInput = self.problem.nInput
        
        if isinstance(alpha, float):
            alpha = np.full((nChains, nInput), alpha)
            
        elif isinstance(alpha, np.ndarray):
            alpha = np.atleast_2d(alpha)
            n, _ = alpha.shape
            if n == 1:
                alpha = np.tile(alpha, (nChains, 1))
            elif n == nChains:
                alpha = alpha
            else:
                raise ValueError("The shape of alpha must be (nChains, nInput) or (1, nInput)")
        else:
            raise ValueError("alpha must be a float or a numpy array")
        
        return alpha

    def validateProblem(self):
        super().validateProblem()
        if self.getParaVal('nChains') < 3:
            raise ValueError("DEMC requires nChains >= 3.")
    
    
    def check_bound(self, X, ub, lb):
        
        span = ub - lb
        y = (X - lb) % (2 * span)
        y = np.where(y > span, 2 * span - y, y)
        X_reflect = lb + y
        
        return X_reflect
    
    def f_prop(self, X_cur, ub, lb, gamma = None):
        
        X_star = np.zeros_like(X_cur)
        
        nChains = X_cur.shape[0]
        
        if gamma is None:
            gamma = np.full(nChains, 2.38 / np.sqrt(2 * self.problem.nInput))
        
        
        for i in range(nChains):
            
            idx = [j for j in range(nChains) if j != i]
            j, k = np.random.choice(idx, 2, replace=False)
            
            X_star[i] = X_cur[i] + gamma[i] * (X_cur[j] - X_cur[k]) + 1e-6 * gamma[i]
        
        return self.check_bound(X_star, ub, lb)
        
    def setProblem(self, problem: ProblemABC):
        
        self.problem = problem
        
    
