# M&L Shuffled Complex Evolution-UA <Single>

import numpy as np

from typing import Optional

from ..base import AlgorithmABC
from ..population import Population
from ..core.constraint import compareSolutions

class ML_SCE_UA(AlgorithmABC):
    """
    Single-objective M&L shuffled complex evolution algorithm.
    """
    
    name = "ML-SCE-UA"
    alg_type = "EA"
    
    def __init__(self, ngs: int = 3, npg: int = 7, nps: int = 4, nspl: int = 7, 
                 alpha: float = 1.0, beta: float = 0.5, sita: float = 0.2,
                 maxFEs: int = 50000, 
                 maxIters: int = 1000, 
                 maxTolerates: int = 1000, tolerate: float = 1e-6,
                 verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = False, saveFlag: bool = True,
                 saveFreq: int = 100):
        """
        Initialize the algorithm.

        :param ngs: Number of complexes.
        :param npg: Number of points in each complex.
        :param nps: Number of points in each simplex.
        :param nspl: Number of evolution steps in each complex.
        :param alpha: Reflection coefficient.
        :param beta: Contraction coefficient.
        :param sita: Smoothing parameter.
        :param maxFEs: Maximum number of function evaluations.
        :param maxIters: Maximum number of iterations.
        :param maxTolerates: Maximum tolerated non-improving iterations.
        :param tolerate: Improvement tolerance.
        :param verboseFlag: Whether to print terminal output.
        :param verboseFreq: Summary output frequency.
        :param logFlag: Whether to save full text logs.
        :param saveFlag: Whether to save sqlite results.
        :param saveFreq: Snapshot save frequency.
        """
        
        super().__init__(maxFEs = maxFEs, maxIters = maxIters, 
                         maxTolerates = maxTolerates, tolerate = tolerate, 
                         verboseFlag = verboseFlag, verboseFreq = verboseFreq, logFlag = logFlag, saveFlag = saveFlag,
                         saveFreq = saveFreq)
        
        # Set algorithm parameters
        self.setParaVal('ngs', ngs)
        self.setParaVal('npg', npg)
        self.setParaVal('nps', nps)
        self.setParaVal('nspl', nspl)
        self.setParaVal('alpha', alpha)
        self.setParaVal('beta', beta)
        self.setParaVal('sita', sita)
        
    def run(self, problem, seed: Optional[int] = None):
        """
        Run the algorithm on the given problem.

        :param problem: Problem instance.
        :param seed: Random seed.
        :return OptResult: Final optimization result.
        """
        # setup algorithm
        self.setup(problem, seed)
        
        # Retrieve parameter values
        ngs, npg, nps, nspl = self.getParaVal('ngs', 'npg', 'nps', 'nspl')
        alpha, beta, sita = self.getParaVal('alpha', 'beta', 'sita')
    
        # Adjust number of complexes if necessary
        if ngs == 0:
            ngs = problem.nInput 
            if ngs > 15:
                ngs = 15
        
        # Initialize SCE parameters
        npg = 2 * ngs + 1
        nps = ngs + 1
        nspl = npg
        nInit = npg * ngs
    
        # Generate initial population
        pop = self.initPop(nInit)
        self.update(pop)
        
        # Sort the population in order of increasing function values
        idx = pop.argsort()
        pop = pop[idx]
                 
        # Iterative process
        while self.checkTermination(pop):
            for igs in range(ngs):
                # Partition the population into complexes (sub-populations)
                outerIdx = np.linspace(0, npg-1, npg, dtype=np.int64) * ngs + igs
                igsPop = pop[outerIdx]
                
                # Evolve sub-population igs for nspl steps
                for _ in range(nspl):
                    # Select simplex by sampling the complex according to a linear probability distribution
                    p = 2 * (npg + 1 - np.linspace(1, npg, npg)) / ((npg + 1) * npg)
                    innerIdx = np.random.choice(npg, nps, p=p, replace=False)
                    innerIdx = np.sort(innerIdx)
                    sPop = igsPop[innerIdx]
                    bPop = igsPop[0]
                    
                    # Execute CCE for simplex
                    sNew = self._cce(sPop, bPop, alpha, beta, sita)
                    igsPop.replace(innerIdx[-1], sNew)
                    
                # End of inner loop for competitive evolution of simplexes
                pop.replace(outerIdx, igsPop)
                
            # Sort the population again
            idx = pop.argsort()
            pop = pop[idx]
            self.update(pop)
                        
        # Return the final result
        return self.finalize()
                     
    def _cce(self, sPop, bPop, alpha, beta, sita):
        """
        Competitive Complex Evolution (CCE) for a given simplex.

        :param sPop: The current simplex population.
        :param bPop: The best population member.
        :param alpha: Reflection coefficient.
        :param beta: Contraction coefficient.
        :param sita: Smoothing parameter.
        
        :return: The new population after CCE.
        """
        
        N, D = sPop.size()
        
        sPopDecs = sPop.decs
        bPopDecs = bPop.decs
        
        sWorst = sPop[-1:]
        sWorstDecs = sWorst.decs
        sWorstObjs = sWorst.objs
        
        # Calculate the centroid of the simplex
        ce = np.mean(sPopDecs[:N], axis=0).reshape(1, -1)
        
        # Reflection step
        sNewDecs = ((sWorstDecs - ce) * alpha * -1 + ce) * (1 - sita) + bPopDecs * sita
        np.clip(sNewDecs, self.problem.lb, self.problem.ub, out=sNewDecs)
        
        sNew = Population(sNewDecs)
        self.evaluate(sNew)
        
        # Contraction step if reflection fails
        if compareSolutions(sNew.objs, sNew.cons, sWorstObjs, sWorst.cons, self.problem.conWgt) >= 0:
            sNewDecs = (sWorstDecs + (sNewDecs - sWorstDecs) * beta) * (1 - sita) + bPopDecs * sita
            np.clip(sNewDecs, self.problem.lb, self.problem.ub, out=sNewDecs)
            
            sNew = Population(sNewDecs)
            self.evaluate(sNew)
        
        # Random point if both reflection and contraction fail
        if compareSolutions(sNew.objs, sNew.cons, sWorstObjs, sWorst.cons, self.problem.conWgt) >= 0:
            sNewDecs = self.problem.lb + np.random.random(D) * (self.problem.ub - self.problem.lb)
            sNew = Population(sNewDecs)
            self.evaluate(sNew)
        
        # End of CCE
        return sNew
