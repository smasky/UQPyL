# Reference vector guided evolutionary algorithm (RVEA) <Multi>
import numpy as np
from scipy.spatial.distance import cdist
from typing import Optional

from ..base import AlgorithmABC, Verbose
from ..util import uniformPoint, gaOperator
from ..population import Population

class RVEA(AlgorithmABC):
    """
    Reference vector guided evolutionary algorithm (RVEA) <Multi>
    -------------------------------------------------------------
    This class implements the RVEA, a multi-objective evolutionary algorithm
    that uses reference vectors to guide the search process.

    Methods:
        run(problem): 
            Executes the RVEA on a given multi-objective problem.
            - problem: Problem
                The problem to solve, which includes attributes like nInput, ub, lb, and evaluate.

    References:
        [1] R. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff, "A Reference Vector Guided Evolutionary Algorithm for Many-Objective Optimization," IEEE Transactions on Evolutionary Computation, vol. 20, no. 5, pp. 773-791, 2016.
    -------------------------------------------------------------
    """
    name="RVEA"
    alg_type="MOEA"
    
    def __init__(self, alpha: float=2.0, fr: float=0.1,
                nPop: int=50,
                maxFEs: int = 50000, 
                maxIters: int = 1000, 
                maxTolerates=None, tolerate=1e-6, 
                verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = True, saveFlag: bool = True):
        """
        Initialize the RVEA with user-defined parameters.
        
        :param alpha: Controls the convergence speed of the algorithm.
        :param fr: Frequency of reference vector adaptation.
        :param nPop: Population size.
        :param maxFEs: Maximum number of function evaluations.
        :param maxIterTimes: Maximum number of iterations.
        :param maxTolerateTimes: Maximum number of tolerated iterations without improvement.
        :param tolerate: Tolerance for improvement.
        :param verbose: Flag to enable verbose output.
        :param verboseFreq: Frequency of verbose output.
        :param logFlag: Flag to enable logging.
        :param saveFlag: Flag to enable saving results.
        """
        super().__init__(maxFEs, maxIters, maxTolerates, tolerate, 
                         verboseFlag, verboseFreq, logFlag, saveFlag)
        
        # Set user-defined parameters
        self.setParaVal('alpha', alpha)
        self.setParaVal('fr', fr)
        self.setParaVal('nPop', nPop)
    
    @Verbose.run
    def run(self, problem, seed: Optional[int] = None):
        """
        Execute the RVEA on the specified multi-objective problem.

        :param problem: An instance of a class derived from ProblemABC.
                        This object defines the optimization problem, including
                        the number of inputs (nInput), number of outputs (nOutput),
                        upper bounds (ub), lower bounds (lb), and evaluation methods.
        
        :return Result: An instance of the Result class, which contains the
                        optimization results, including the best decision variables,
                        objective values, and constraint violations encountered during
                        the optimization process.
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
                        
        # Return the final result
        return self.result
    
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