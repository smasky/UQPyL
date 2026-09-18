# Reference vector guided evolutionary algorithm (RVEA) <Multi>
import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional

from ..base import AlgorithmABC
from ..core import uniformPoint, gaOperator
from ..population import Population
from ..core.constraint import calcConstraintViolation

class RVEA(AlgorithmABC):
    """
    Multi-objective reference vector guided evolutionary algorithm.

    Examples:
        >>> rvea = RVEA(nPop=100, maxFEs=5000)
        >>> res = rvea.run(problem, seed=1234)
        >>> print(res.bestObjs)

    References:
        [1] R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff, A reference vector guided
            evolutionary algorithm for many-objective optimization, IEEE Transactions
            on Evolutionary Computation, vol. 20, no. 5, pp. 773-791, 2016.
    """
    name="RVEA"
    alg_type="MOEA"
    
    def __init__(self, alpha: float=2.0, fr: float=0.1,
                nPop: int=50,
                maxFEs: int = 50000, 
                maxIters: int = 1000, 
                maxTolerates=None, tolerate=1e-6, 
                verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = True, saveFlag: bool = True,
                saveFreq: int = 100, hvRefPoint=None, historyFreq: int = 10):
        """
        Initialize the algorithm.

        :param alpha: Angle penalty parameter.
        :param fr: Reference vector adaptation frequency.
        :param nPop: Population size.
        :param maxFEs: Maximum number of function evaluations.
        :param maxIters: Maximum number of iterations.
        :param maxTolerates: Maximum tolerated non-improving iterations.
        :param tolerate: Improvement tolerance.
        :param verboseFlag: Whether to print terminal output.
        :param verboseFreq: Summary output frequency.
        :param logFlag: Whether to save full text logs.
        :param saveFlag: Whether to save sqlite results.
        :param saveFreq: SQLite snapshot save frequency.
        :param historyFreq: Full in-memory snapshot interval; None keeps only the final snapshot.
        """
        super().__init__(maxFEs, maxIters, maxTolerates, tolerate, 
                         verboseFlag, verboseFreq, logFlag, saveFlag, saveFreq, hvRefPoint=hvRefPoint, historyFreq=historyFreq)
        
        # Set user-defined parameters
        self.set('alpha', alpha)
        self.set('fr', fr)
        self.set('nPop', nPop)
    
    def run(self, problem, seed: Optional[int] = None, initialPop=None):
        """
        Run the algorithm on the given problem.

        :param problem: Problem instance.
        :param seed: Random seed.
        :param initialPop: Optional initial population or decision matrix.
        :return OptResult: Final optimization result.
        """
        # setup algorithm
        self.setup(problem, seed)
        
        # Parameters setting
        alpha, fr = self.get('alpha', 'fr')
        nPop = self.get('nPop')
    
        # Generate initial reference vectors
        V0, nPop = uniformPoint(nPop, problem.nOutput)
        V = np.copy(V0)
        self.referenceScale = np.ones(problem.nOutput)
        
        # Generate initial population
        pop = self.initPop(nPop, initialPop=initialPop)
        self.update(pop)
        
        # Iterative process
        while self.checkTermination(pop):
            
            # Select mating pool randomly
            matingPoolIdx = self.rng.integers(0, len(pop), nPop)
            matingPool = pop[matingPoolIdx]
            # Generate offspring using genetic operations
            offspringDecs = gaOperator(matingPool.decs, self.searchUb, self.searchLb, rng=self.rng)
            offspring = Population(offspringDecs)
            
            # Evaluate the offspring
            self.evaluate(offspring)
            
            # Environmental selection
            pop.merge(offspring)
            nextIdx = self.environmentSelection(pop.objs, V, (self.FEs/self.maxFEs)**alpha, pop.cons, pop.conWgt)
            pop = pop[nextIdx]
            
            # Check if reference vectors need to be updated
            condition = not (np.ceil(self.FEs / nPop) % np.ceil(fr * self.maxFEs / nPop))
            
            if condition:
                # Update reference vectors
                V = self.updateReferenceVector(pop.objs, V0)
            self.update(pop, completed=True)
                        
        # Return the final result
        return self.finalize()
    
    def updateReferenceVector(self, popObjs, V):
        """
        Update the reference vectors based on the current population.

        :param pop: Current population.
        :param V: Initial reference vectors.
        
        :return: Updated reference vectors.
        """
        # Calculate scaling factors based on the population's objective values
        span = np.ptp(popObjs, axis=0)
        previous = getattr(self, "referenceScale", np.ones(popObjs.shape[1]))
        self.referenceScale = np.where(np.isfinite(span) & (span > 0), span, previous)

        # Retain the last useful scale for constant objectives or singleton fronts.
        V = V * self.referenceScale
        
        return V
    
    def environmentSelection(self, popObjs, V, theta, popCons=None, conWgt=None):
        """
        Perform environmental selection to choose the next generation.

        :param pop: Merged population of current and offspring.
        :param V: Reference vectors.
        :param theta: Angle control parameter.
        
        :return: Selected population for the next generation.
        """
        if popCons is not None:
            violation = calcConstraintViolation(popCons, conWgt)
            feasible = np.flatnonzero(violation <= 0)
            if not feasible.size:
                return np.argsort(violation, kind="stable")[:V.shape[0]]
            # Preserve reference-vector selection among feasible solutions.
            selected = self.environmentSelection(popObjs[feasible], V, theta)
            return feasible[selected]

        M = popObjs.shape[1]
        
        nV = V.shape[0]
        
        # Normalize the objective values
        popObjs = popObjs - np.min(popObjs, axis=0)
        
        vectorNorm = np.linalg.norm(V, axis=1, keepdims=True)
        if np.any(vectorNorm <= 0) or not np.all(np.isfinite(vectorNorm)):
            raise ValueError("Reference vectors must be finite and nonzero.")
        unitV = V / vectorNorm
        cosine = np.clip(unitV @ unitV.T, -1.0, 1.0)
        np.fill_diagonal(cosine, -1.0)
        gamma = np.maximum(np.min(np.arccos(cosine), axis=1), 1e-12)
        objectiveNorm = np.linalg.norm(popObjs, axis=1, keepdims=True)
        unitObjs = np.divide(popObjs, objectiveNorm, out=np.zeros_like(popObjs),
                             where=objectiveNorm > 0)
        angle = np.arccos(np.clip(unitObjs @ unitV.T, -1.0, 1.0))
        # The ideal point has zero APD in every direction, without undefined angles.
        angle[objectiveNorm[:, 0] == 0] = 0.0

        # Associate each solution with a reference vector
        associate = np.argmin(angle, axis=1)
        
        next = np.ones(nV, dtype=np.int32)*-1
        
        for i in np.unique(associate):
            current1 = np.where(associate == i)[0]
            
            if len(current1) > 0:
                # Calculate the APD value for each solution
                APD = (1 + M * theta * angle[current1, i] / gamma[i]) * np.sqrt(np.sum(popObjs[current1, :]**2, axis=1))
                # Select the one with the minimum APD value
                best = np.argmin(APD)
                next[i] = current1[best]
        
        nextIdx = next[next != -1].astype(int)
        
        return nextIdx 
