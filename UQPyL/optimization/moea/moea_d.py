# Multi-objective Evolutionary Algorithm based on Decomposition (MOEAD) <Multi>
import math
import numpy as np
from typing import Literal, Optional
from scipy.spatial import distance

from ..base import AlgorithmABC
from ..population import Population
from ..core import uniformPoint, gaOperatorHalf, calcConstraintViolation

class MOEAD(AlgorithmABC):
    """
    Multi-objective evolutionary algorithm based on decomposition.

    Examples:
        >>> moead = MOEAD(nPop=100, maxFEs=5000)
        >>> res = moead.run(problem, seed=1234)
        >>> print(res.bestObjs)

    References:
        [1] Q. Zhang and H. Li, MOEA/D: A multiobjective evolutionary algorithm based
            on decomposition, IEEE Transactions on Evolutionary Computation,
            vol. 11, no. 6, pp. 712-731, 2007.
    """
    
    name = "MOEA_D"
    alg_type = "MOEA"
    
    def __init__(self, aggregation: Literal['PBI', 'TCH', 'TCH_N', 'TCH_M'] = 'TCH',
                 nPop: int = 50,
                 maxFEs: int = 50000, 
                 maxIters: int = 1000, 
                 maxTolerates = None, tolerate = 1e-6, 
                 verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = True, saveFlag: bool = True,
                 saveFreq: int = 100):
        """
        Initialize the algorithm.

        :param aggregation: Aggregation method.
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
        
        # Initialize the base class with common parameters
        super().__init__(maxFEs, maxIters, maxTolerates, tolerate, 
                         verboseFlag, verboseFreq, logFlag, saveFlag, saveFreq)
        
        # Set specific parameters for MOEAD
        self.set('aggregation', aggregation)
        self.set('nPop', nPop)
        
    #-------------------Public Functions-----------------------#
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
        aggregation = self.get('aggregation')
        
        nPop = self.get('nPop')
        
        # Determine the number of neighbors
        T = math.ceil(nPop / 10)
        
        # Generate uniform weight vectors
        W, N = uniformPoint(nPop, problem.nOutput)
        
        # Adjust population size
        nPop = N
        
        # Calculate the distance matrix and sort neighbors
        B = distance.cdist(W, W, metric='euclidean')
        B = np.argsort(B, axis=1)
        B = B[:, 0:T]
        
        # Generate initial population
        pop = self.initPop(nPop)
        self.update(pop)
        
        # Initialize the ideal point
        Z = np.min(pop.objs, axis=0).reshape(1, -1)
         
        # Main loop of the algorithm
        while self.checkTermination(pop):
            
            for i in range(nPop):
                
                # Select parents from the neighborhood
                P = B[i, self.rng.permutation(B.shape[1])].ravel()

                # Generate offspring using genetic operations
                subPop = pop[P[0:2]]
                offspringDecs = gaOperatorHalf(subPop.decs, problem.ub, problem.lb, 1, 20, 1, 20, rng=self.rng)
                offspring = Population(offspringDecs)
                # Evaluate the offspring
                self.evaluate(offspring)
                
                # Update the ideal point
                Z = np.min(np.vstack((Z, offspring.objs)), axis=0).reshape(1, -1)
                
                # Extract objective values for parents and offspring
                popObjs = pop.objs[P]
                offspringObjs = offspring.objs
                
                # Calculate aggregation values based on the selected method
                if aggregation == 'PBI':
                    # Penalty-based Boundary Intersection
                    normW = np.sqrt(np.sum(W[P, :]**2, axis=1))
                    normP = np.sqrt(np.sum((popObjs - np.tile(Z, (T, 1)))**2, axis=1))
                    normO = np.sqrt(np.sum((offspringObjs - Z)**2, axis=1))
                    CosineP = np.sum((pop.objs[P] - np.tile(Z, (T, 1))) * W[P, :], axis=1) / normW / normP
                    CosineO = np.sum(np.tile(offspringObjs - Z, (T, 1)) * W[P, :], axis=1) / normW / normO
                    g_old = normP * CosineP + 5 * normP * np.sqrt(1 - CosineP**2)
                    g_new = normO * CosineO + 5 * normO * np.sqrt(1 - CosineO**2)
                    
                elif aggregation == 'TCH':
                    # Tchebycheff approach
                    g_old = np.max(np.abs(popObjs - np.tile(Z, (T, 1))) * W[P, :], axis=1)
                    g_new = np.max(np.tile(np.abs(offspringObjs - Z), (T, 1)) * W[P, :], axis=1)
                    
                elif aggregation == 'TCH_N':
                    # Normalized Tchebycheff approach
                    Zmax = np.max(pop.objs, axis=0)
                    g_old = np.max(np.abs(popObjs - np.tile(Z, (T, 1))) / np.tile(Zmax - Z, (T, 1)) * W[P, :], axis=1)
                    g_new = np.max(np.tile(np.abs(offspringObjs - Z) / (Zmax - Z), (T, 1)) * W[P, :], axis=1)
                    
                elif aggregation == 'TCH_M':
                    # Modified Tchebycheff approach
                    g_old = np.max(np.abs(popObjs - np.tile(Z, (T, 1))) / W[P, :], axis=1)
                    g_new = np.max(np.tile(np.abs(offspringObjs - Z), (T, 1)) / W[P, :], axis=1)
                
                parentCons = None if pop.cons is None else pop.cons[P]
                offspringCons = None if offspring.cons is None else np.repeat(offspring.cons, len(P), axis=0)
                parentCV = calcConstraintViolation(parentCons, pop.conWgt)
                offspringCV = calcConstraintViolation(offspringCons, pop.conWgt)

                if parentCV is None:
                    replaceMask = g_old >= g_new
                else:
                    parentFeasible = parentCV <= 0
                    offspringFeasible = offspringCV <= 0
                    replaceMask = np.zeros(len(P), dtype=bool)
                    replaceMask[offspringFeasible & ~parentFeasible] = True
                    bothInfeasible = ~offspringFeasible & ~parentFeasible
                    replaceMask[bothInfeasible] = offspringCV[bothInfeasible] < parentCV[bothInfeasible]
                    bothFeasible = offspringFeasible & parentFeasible
                    replaceMask[bothFeasible] = g_old[bothFeasible] >= g_new[bothFeasible]

                # Replace individuals in the population based on feasibility/CV first, then aggregation
                pop.replace(P[replaceMask], offspring)
            self.update(pop)
                    
        # Return the final result
        return self.finalize()    
