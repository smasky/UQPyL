from re import I
import numpy as np
import xarray as xr
from typing import Literal, Union

from .base import InferenceABC
from ..problem import ProblemABC
from ..doe import LHS
from ..util.verbose import Verbose

class DREAM_ZS(InferenceABC):
    
    # Differential Evolution Adaptive Metropolis Z-score
    # ---------------------------------------- #
    
    name = "DREAM-ZS"
    
    def __init__(self, nChains: int = 10, warmUp: int = 1000, 
                       ps: float = 0.1, k: int = 1, jitter: float = 0.1,
                       adpInterval: int = 50, archSize: int = 10,
                       acTarget: float = 0.25, nCR: int = 5,
                       maxIters: int = 1000,
                       verboseFlag: bool = True, verboseFreq: int = 10,
                       logFlag: bool = False, saveFlag: bool = True):
        
        super().__init__(maxIters, verboseFlag, verboseFreq, logFlag, saveFlag)
        
        self.setParaVal('nChains', nChains)
        self.setParaVal('warmUp', warmUp)
        self.setParaVal('ps', ps)
        self.setParaVal('k', k)
        self.setParaVal('nCR', nCR)
        self.setParaVal('jitter', jitter)
        self.setParaVal('archSize', archSize)
        self.setParaVal('adpInterval', adpInterval)
        self.setParaVal('acTarget', acTarget)
    
    @Verbose.inference
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

        # init chains
        chains = self.initChains(nChains, X_init, Objs_init, Cons_init)
        archSize = int(nChains * archSize)
        archive = [x for x in X_init]
        
        # warm up
        X_cur = X_init; Objs_cur = Objs_init; Cons_cur = Cons_init
        
        for _ in range(warmUp):
            
            X_star, Q_ratio, crIdxs = self.f_prop_ratio(X_cur, archive, ps, k, jitter, gamma, gamma_scale, crSet, pCR, problem.ub, problem.lb)
            Objs_star, Cons_star = self.evaluate(X_star)
            
            for i, chain in enumerate(chains):
                
                if np.log(np.random.rand()) < (self.log_prob(Objs_star[i]) - self.log_prob(Objs_cur[i]) + np.log(Q_ratio[i])) \
                    and (problem.nCons == 0 or (problem.nCons > 0 and all(Cons_star[i] <= 0))):
                        
                    X_cur[i] = X_star[i]; Objs_cur[i] = Objs_star[i]
                    if problem.nCons > 0:
                        Cons_cur[i] = Cons_star[i]
                    
                    archive.append(X_cur[i])
                    
                if len(archive) > archSize:
                    archive = archive[-archSize:]
        
        # main loop
        ac_rate = np.zeros(nChains)
        ac_local = np.zeros(nChains)
        while self.checkTermination(chains):
            
            X_star, Q_ratio, crIdxs = self.f_prop_ratio(X_cur, archive, ps, k, jitter, gamma, gamma_scale, crSet, pCR, problem.ub, problem.lb)
            Objs_star, Cons_star = self.evaluate(X_star)
            
            for i, chain in enumerate(chains):
                
                if np.log(np.random.rand()) < (self.log_prob(Objs_star[i]) - self.log_prob(Objs_cur[i]) + np.log(Q_ratio[i])) \
                    and (problem.nCons == 0 or (problem.nCons > 0 and all(Cons_star[i] <= 0))):
                    
                    if crIdxs[i] > -1:
                        jumpDist = float(np.sum(X_star[i] - X_cur[i])**2)
                        cr_gain[crIdxs[i]] += jumpDist
                        
                    X_cur[i] = X_star[i]; Objs_cur[i] = Objs_star[i]
                    if problem.nCons > 0:
                        Cons_cur[i] = Cons_star[i]
                    
                    ac_local[i] += 1 / warmUp
                    
                    archive.append(X_cur[i])
                    
                chain.add(X_cur[i], Objs_cur[i], Cons_cur[i] if problem.nCons > 0 else None)
            
            if len(archive) > archSize:
                archive = archive[-archSize:]
            
            if self.iter % adpInterval == 0:
                pCR, cr_gain, cr_tries, gamma_scale = self.adaption(pCR, cr_gain, cr_tries, ac_local, gamma_scale, acTarget)

                ac_local = ac_local * 0.0
            
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
    
    def adaption(self, pCR, cr_gain, cr_tries, ac_local, gamma_scale, acTarget = None):
        
        # update pCR
        if np.any(cr_tries > 0):
            avg = cr_gain / np.maximum(cr_tries, 1e-6)
            
            if np.all(avg == 0):
                avg = np.ones(cr_gain.shape[0])
            
            w = avg / np.sum(avg)
            
            w = 0.8 * w + 0.2 * (np.ones_like(w)) / w.shape[0]

            pCR = w / np.sum(w)
        
        cr_gain = cr_gain * 0.0
        cr_tries = cr_tries * 0.0
        
        # gamma_scale
        if acTarget is not None:
            
            ac_mean = float(np.mean(ac_local))
            
            gamma_scale *= np.exp(0.1*(ac_mean - acTarget))
            
            gamma_scale = np.clip(gamma_scale, 0.3, 3.0)
        
        return pCR, cr_gain, cr_tries, gamma_scale

    def f_prop_ratio(self, X_cur, archive, ps, k, jitter, gamma, gamma_scale, crSet, pCR, ub, lb):
        
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
            delta += pool[a_idx] - pool[b_idx]  # (D,)

        mask = np.random.rand(d) < cr
        if not mask.any():
            mask[np.random.randint(0, d)] = True
        
        x_prop = x_i.copy()
        x_prop[mask] = x_prop[mask] + gamma * delta[mask] + 1e-6
        
        return x_prop, 1.0
        
        
    def check_bound(self, X, ub, lb):
        
        span = ub - lb
        y = (X - lb) % (2 * span)
        y = np.where(y > span, 2 * span - y, y)
        X_reflect = lb + y
        
        return X_reflect

    def setProblem(self, problem: ProblemABC):
        
        self.problem = problem
        
    
    def log_prob(self, y):
    
        return -y