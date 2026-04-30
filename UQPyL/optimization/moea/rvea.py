# Reference vector guided evolutionary algorithm (RVEA) <Multi>
import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional

from ..base import AlgorithmABC
from ..core import uniformPoint, gaOperator
from ..population import Population

class RVEA(AlgorithmABC):
    """
    Multi-objective reference vector guided evolutionary algorithm.
    """
    name="RVEA"
    alg_type="MOEA"
    
    def __init__(self, alpha: float=2.0, fr: float=0.1,
                nPop: int=50,
                maxFEs: int = 50000, 
                maxIters: int = 1000, 
                maxTolerates=None, tolerate=1e-6, 
                verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = True, saveFlag: bool = True,
                saveFreq: int = 100):
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
        :param saveFreq: Snapshot save frequency.
        """
        super().__init__(maxFEs, maxIters, maxTolerates, tolerate, 
                         verboseFlag, verboseFreq, logFlag, saveFlag, saveFreq)
        
        # Set user-defined parameters
        self.setParaVal('alpha', alpha)
        self.setParaVal('fr', fr)
        self.setParaVal('nPop', nPop)
    
    def run(self, problem, seed: Optional[int] = None):
        """
        Run the algorithm on the given problem.

        :param problem: Problem instance.
        :param seed: Random seed.
        :return OptResult: Final optimization result.
        """
        # setup algorithm
        self.setup(problem, seed)
        
        # Parameters setting
        alpha, fr = self.getParaVal('alpha', 'fr')
        nPop = self.getParaVal('nPop')
    
        # Generate initial reference vectors
        V0, nPop = uniformPoint(nPop, problem.nOutput)
        V = np.copy(V0)
        
        # Generate initial population
        pop = self.initPop(nPop)
        self.update(pop)
        
        # Iterative process
        while self.checkTermination(pop):
            
            # Select mating pool randomly
            matingPoolIdx = np.random.randint(0, len(pop), nPop)
            matingPool = pop[matingPoolIdx]
            # Generate offspring using genetic operations
            offspringDecs = gaOperator(matingPool.decs, problem.ub, problem.lb)
            offspring = Population(offspringDecs)
            
            # Evaluate the offspring
            self.evaluate(offspring)
            
            # Environmental selection
            pop.merge(offspring)
            nextIdx = self.environmentSelection(pop.objs, V, (self.FEs/self.maxFEs)**alpha)
            pop = pop[nextIdx]
            
            # Check if reference vectors need to be updated
            condition = not (np.ceil(self.FEs / nPop) % np.ceil(fr * self.maxFEs / nPop))
            
            if condition:
                # Update reference vectors
                V = self.updateReferenceVector(pop.objs, V0)
            self.update(pop)
                        
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
        scaling_factors = np.max(popObjs, axis=0) - np.min(popObjs, axis=0)
        
        # Scale the reference vectors
        V = V * scaling_factors
        
        return V
    
    def environmentSelection(self, popObjs, V, theta):
        """
        Perform environmental selection to choose the next generation.

        :param pop: Merged population of current and offspring.
        :param V: Reference vectors.
        :param theta: Angle control parameter.
        
        :return: Selected population for the next generation.
        """
        M = popObjs.shape[1]
        
        nV = V.shape[0]
        
        # Normalize the objective values
        popObjs = popObjs - np.min(popObjs, axis=0)
        
        # Calculate cosine similarity between reference vectors
        cosine = 1-cdist(V, V, metric='cosine')
        
        np.fill_diagonal(cosine, 0)
        
        # Calculate the minimum angle between reference vectors
        gamma = np.min(np.arccos(cosine), axis=1)
        
        # Calculate the angle between population objectives and reference vectors
        angle = np.arccos(1-cdist(popObjs, V, metric="cosine"))
        
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
