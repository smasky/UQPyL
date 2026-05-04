import numpy as np
from typing import Literal, Union

from ..base import InferenceABC
from ...problem import ProblemABC

class DREAM_ZS(InferenceABC):
    """
    DREAM-ZS Inference
    Differential evolution MCMC with snooker updates and adaptive crossover weights.

    Examples:
        >>> dream = DREAM_ZS(nChains=8, warmUp=500, maxIters=2000)
        >>> res = dream.run(problem, seed=1234)
        >>> print(res.acceptanceRate)
    """
    
    name = "DREAM-ZS"
    
    def __init__(self, nChains: int = 10, warmUp: int = 1000, 
                       ps: float = 0.1, k: int = 1, jitter: float = 0.1,
                       adpInterval: int = 50, archSize: int = 10,
                       acTarget: float = 0.25, nCR: int = 5,
                       maxIters: int = 1000,
                       verboseFlag: bool = True, verboseFreq: int = 10,
                       logFlag: bool = False, saveFlag: bool = True,
                       saveFreq: int = 100, logProbFunc=None,
                       maxInitAttempts: int = 1000):
        """
        Initialize the DREAM-ZS inference method.

        Args:
            nChains: Number of sampling chains.
            warmUp: Number of warm-up iterations before formal sampling.
            ps: Probability of snooker update.
            k: Number of differential evolution pairs.
            jitter: Multiplicative proposal jitter scale.
            adpInterval: Interval for adaptive crossover updates.
            archSize: Archive size multiplier relative to chain count.
            acTarget: Target acceptance rate for gamma scaling.
            nCR: Number of crossover rate candidates.
            maxIters: Number of formal sampling draws.
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
        
        self.setParaVal('nChains', nChains)
        self.setParaVal('warmUp', warmUp)
        self.setParaVal('ps', ps)
        self.setParaVal('k', k)
        self.setParaVal('nCR', nCR)
        self.setParaVal('jitter', jitter)
        self.setParaVal('archSize', archSize)
        self.setParaVal('adpInterval', adpInterval)
        self.setParaVal('acTarget', acTarget)
    
    def run(self, problem: ProblemABC, gamma: Union[float, np.ndarray] = None, seed: int = None):
        
        self.setup(problem, seed)
        
        nChains = self.getParaVal('nChains')
        warmUp = self.getParaVal('warmUp')
        ps = self.getParaVal('ps')
        jitter = self.getParaVal('jitter')
        k = self.getParaVal('k')
        archSize = self.getParaVal('archSize')
        adpInterval = self.getParaVal('adpInterval')
        nCR = self.getParaVal('nCR')
        acTarget = self.getParaVal('acTarget')
        
        crSet = np.linspace(0.1, 0.9, nCR)
        pCR = np.ones(nCR) / nCR
        cr_gain = np.zeros(nCR)
        cr_tries = np.zeros(nCR)
        
        if gamma is None:
            gamma = 2.38 / np.sqrt(2 * problem.nInput)
        gamma = self._check_gamma_(gamma)
        gamma_scale = 1.0
        
        X_init, Objs_init, Cons_init = self.initialSampling(problem, nChains)

        chains = self.initChains(nChains, X_init, Objs_init, Cons_init)
        archSize = int(nChains * archSize)
        archive = [x for x in X_init]
        
        X_cur = X_init; Objs_cur = Objs_init; Cons_cur = Cons_init
        
        for _ in range(warmUp):
            
            X_star, Q_ratio, crIdxs = self.f_prop_ratio(
                X_cur, archive, ps, k, jitter, gamma, gamma_scale, crSet, pCR,
                cr_tries, problem.ub, problem.lb,
            )
            Objs_star, Cons_star = self.evaluate(X_star)
            
            for i, chain in enumerate(chains):
                
                if self.accept(
                    Objs_star[i], Objs_cur[i],
                    Cons_star[i] if problem.nCons > 0 else None,
                    Q_ratio[i],
                    decStar=X_star[i], decCur=X_cur[i],
                    consCur=Cons_cur[i] if problem.nCons > 0 else None,
                ):
                        
                    X_cur[i] = X_star[i]; Objs_cur[i] = Objs_star[i]
                    if problem.nCons > 0:
                        Cons_cur[i] = Cons_star[i]
                    
                    archive.append(X_cur[i])
                    
                if len(archive) > archSize:
                    archive = archive[-archSize:]
        
        chains = self.initChains(nChains, X_cur, Objs_cur, Cons_cur)
        self.update(chains)
        ac_local = np.zeros(nChains)
        while self.checkTermination(chains):
            
            X_star, Q_ratio, crIdxs = self.f_prop_ratio(
                X_cur, archive, ps, k, jitter, gamma, gamma_scale, crSet, pCR,
                cr_tries, problem.ub, problem.lb,
            )
            Objs_star, Cons_star = self.evaluate(X_star)
            
            for i, chain in enumerate(chains):
                
                accepted = self.accept(
                    Objs_star[i], Objs_cur[i],
                    Cons_star[i] if problem.nCons > 0 else None,
                    Q_ratio[i],
                    decStar=X_star[i], decCur=X_cur[i],
                    consCur=Cons_cur[i] if problem.nCons > 0 else None,
                )
                if accepted:
                    
                    if crIdxs[i] > -1:
                        jumpDist = float(np.sum(X_star[i] - X_cur[i])**2)
                        cr_gain[crIdxs[i]] += jumpDist
                        
                    X_cur[i] = X_star[i]; Objs_cur[i] = Objs_star[i]
                    if problem.nCons > 0:
                        Cons_cur[i] = Cons_star[i]
                    
                    denom = warmUp if warmUp and warmUp > 0 else max(self.maxIters, 1)
                    ac_local[i] += 1 / denom
                    
                    archive.append(X_cur[i])
                    
                chain.add(
                    X_cur[i],
                    Objs_cur[i],
                    Cons_cur[i] if problem.nCons > 0 else None,
                    logProb=self.log_prob(Objs_cur[i], decs=X_cur[i], cons=Cons_cur[i] if problem.nCons > 0 else None),
                    accepted=accepted,
                )
            
            if len(archive) > archSize:
                archive = archive[-archSize:]
            
            if self.iter % adpInterval == 0:
                pCR, cr_gain, cr_tries, gamma_scale = self.adaption(pCR, cr_gain, cr_tries, ac_local, gamma_scale, acTarget)

                ac_local = ac_local * 0.0

            self.update(chains)

        return self.finalize()
    
    def adaption(self, pCR, cr_gain, cr_tries, ac_local, gamma_scale, acTarget = None):
        
        if np.any(cr_tries > 0):
            avg = cr_gain / np.maximum(cr_tries, 1e-6)
            
            if np.all(avg == 0):
                avg = np.ones(cr_gain.shape[0])
            
            w = avg / np.sum(avg)
            
            w = 0.8 * w + 0.2 * (np.ones_like(w)) / w.shape[0]

            pCR = w / np.sum(w)
        
        cr_gain = cr_gain * 0.0
        cr_tries = cr_tries * 0.0
        
        if acTarget is not None:
            
            ac_mean = float(np.mean(ac_local))
            
            gamma_scale *= np.exp(0.1*(ac_mean - acTarget))
            
            gamma_scale = np.clip(gamma_scale, 0.3, 3.0)
        
        return pCR, cr_gain, cr_tries, gamma_scale

    def validateProblem(self):
        super().validateProblem()
        if self.getParaVal('nChains') < 3:
            raise ValueError("DREAM_ZS requires nChains >= 3.")

    def f_prop_ratio(self, X_cur, archive, ps, k, jitter, gamma, gamma_scale, crSet, pCR, cr_tries, ub, lb):
        
        nChains = X_cur.shape[0]
        
        X_star = np.zeros_like(X_cur)
  
        Q_ratio = np.ones(nChains)
        
        crIdxs = np.full(nChains, -1)
        
        gamma = gamma * gamma_scale * (1.0 + jitter * np.random.normal(0, 1))
        
        for i in range(nChains):
            
            if np.random.rand() < ps:
                X_star[i], q_ratio = self.snooker_update(i, X_cur, archive, gamma[i])
            else:
                cr_idx = np.random.choice(crSet.shape[0], p = pCR)
                cr = crSet[cr_idx]
                crIdxs[i] = cr_idx
                cr_tries[cr_idx] += 1
                X_star[i], q_ratio = self.de_prop(i, X_cur, archive, k, cr, gamma[i])
            
            Q_ratio[i] = q_ratio
            
        return self.check_bound(X_star, ub, lb), Q_ratio, crIdxs
    
    def snooker_update(self, i, X_cur, archive, gamma):
        
        x_i = X_cur[i]
        
        z = archive[np.random.randint(0, len(archive))]
                
        idx = [k for k in range(X_cur.shape[0]) if k != i]
        
        r, s = np.random.choice(idx, size = 2, replace = False)
        
        v = X_cur[r] - X_cur[s]
        v_norm2 = np.dot(v, v)
        
        if v_norm2 == 0:
            v = np.random.normal(X_cur.shape[1]) * 1e-6
            v_norm2 = np.dot(v, v)
        
        proj = np.dot(v, z - x_i) / v_norm2
        
        x_prop = x_i + gamma * proj * v + 1e-6 * np.random.rand()
        
        q_ratio = (np.dot(x_prop - X_cur[r], x_prop - X_cur[s]) / np.dot(x_i - X_cur[r], x_i - X_cur[s]))
        q_ratio = abs(q_ratio) ** (X_cur.shape[1] / 2)
                
        return x_prop, q_ratio
    
    def de_prop(self, i, X_cur, archive, k, cr, gamma):
        
        n, d = X_cur.shape
        
        x_i = X_cur[i]
        
        pool = np.vstack([X_cur, archive])
        
        pool_idx_X = np.arange(n)
        pool_idx_X = pool_idx_X[pool_idx_X != i]
        pool_idx_A = n + np.arange(len(archive))
        pool_idx = np.concatenate([pool_idx_X, pool_idx_A])
        
        need = 2 * k
        choose = np.random.choice(pool_idx, size = need, replace = False)
        
        delta = np.zeros(d)
        for i in range(k):
            a_idx = choose[2 * i]
            b_idx = choose[2 * i + 1]
            delta += pool[a_idx] - pool[b_idx]

        mask = np.random.rand(d) < cr
        if not mask.any():
            mask[np.random.randint(0, d)] = True
        
        x_prop = x_i.copy()
        x_prop[mask] = x_prop[mask] + gamma[mask] * delta[mask] + 1e-6
        
        return x_prop, 1.0
        
        
    def check_bound(self, X, ub, lb):
        
        span = ub - lb
        y = (X - lb) % (2 * span)
        y = np.where(y > span, 2 * span - y, y)
        X_reflect = lb + y
        
        return X_reflect

    def setProblem(self, problem: ProblemABC):
        
        self.problem = problem
        
    
