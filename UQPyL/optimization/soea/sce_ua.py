# Shuffled Complex Evolution-UA
import numpy as np

from typing import Optional

from ..base import AlgorithmABC
from ..population import Population
from ..core.constraint import compareSolutions

class SCE_UA(AlgorithmABC):
    """
    Single-objective shuffled complex evolution algorithm.
    """
    
    name = "SCE-UA"
    alg_type = "EA"
    
    def __init__(self, ngs: int = 3, npg: int = 7, nps: int = 4, nspl: int = 7,
                 alpha: float = 1.0, beta: float = 0.5,
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
        alpha, beta = self.getParaVal('alpha', 'beta')
                    
        # Adjust ngs if necessary
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
        
        # Sort the population by increasing function values
        pop = pop[pop.argsort()]
                
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
                    
                    # Execute CCE for simplex
                    sNew = self._cce(sPop, alpha, beta)
                    igsPop.replace(innerIdx[-1], sNew)
                
                # Replace the complex with the evolved sub-population
                pop.replace(outerIdx, igsPop)
                
            # Sort the population by increasing function values
            idx = pop.argsort()
            pop = pop[idx]
            self.update(pop)
                   
        # Return the final result
        return self.finalize()
                     
    def _cce(self, sPop, alpha, beta):
        '''
        Perform the Competitive Complex Evolution (CCE) on a simplex.

        :param sPop: The simplex population to evolve.
        :param alpha: Reflection coefficient.
        :param beta: Contraction coefficient.
        
        :return: The new evolved population.
        '''
        
        N, D = sPop.size()

        sPopDecs = sPop.decs
        
        sWorstDecs = sPop.decs[-1:]
        sWorstObjs = sPop.objs[-1:]
        sWorstCons = sPop.cons[-1:] if sPop.cons is not None else None
        
        # Calculate the centroid of the best N-1 points
        ce = np.mean(sPopDecs[:N], axis=0).reshape(1, -1)
        
        # Reflect the worst point
        sNewDecs = (sWorstDecs - ce) * alpha * -1 + ce
        np.clip(sNewDecs, self.problem.lb, self.problem.ub, out=sNewDecs)
        
        sNew = Population(sNewDecs)
        self.evaluate(sNew)
        
        if compareSolutions(sNew.objs, sNew.cons, sWorstObjs, sWorstCons, self.problem.conWgt) >= 0:
            # Contract the worst point
            sNewDecs = sWorstDecs + (sNewDecs - sWorstDecs) * beta
            np.clip(sNewDecs, self.problem.lb, self.problem.ub, out=sNewDecs)
            
            sNew = Population(sNewDecs)
            self.evaluate(sNew)
            # If both reflection and contraction fail, generate a random point
            if compareSolutions(sNew.objs, sNew.cons, sWorstObjs, sWorstCons, self.problem.conWgt) >= 0:
                sNewDecs = self.problem.lb + np.random.random(D) * (self.problem.ub - self.problem.lb)
                sNew = Population(sNewDecs)
                self.evaluate(sNew)
                
        # Return the new evolved population
        return sNew
            
        
                
                
        
        
        
        
        
        
        
        
        
        
