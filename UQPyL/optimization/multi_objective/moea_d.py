# Multi-objective Evolutionary Algorithm based on Decomposition (MOEAD) <Multi>
import numpy as np
import math
from typing import Literal
from scipy.spatial import distance

from ..algorithmABC import Algorithm
from ..population import Population
from ..utility_functions import uniformPoint, NDSort
from ..utility_functions.operation_GA import operationGAHalf
from ...utility import Verbose

class MOEAD(Algorithm):
    '''
    Multi-objective Evolutionary Algorithm based on Decomposition <Multi>
    ---------------------------------------------------------------------
    This class implements the MOEAD algorithm, which is used for solving
    multi-objective optimization problems by decomposing them into simpler
    subproblems.

    References:
        Zhang, Q., & Li, H. (2007). MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition. 
        IEEE Transactions on Evolutionary Computation, 11(6), 712-731.
        DOI: 10.1109/TEVC.2007.892759
    '''
    
    name = "MOEA_D"
    type = "MOEA"
    
    def __init__(self, aggregation: Literal['PBI', 'TCH', 'TCH_N', 'TCH_M'] = 'PBI',
                 nPop: int = 50,
                 maxFEs: int = 50000, 
                 maxIterTimes: int = 1000, 
                 maxTolerateTimes = None, tolerate = 1e-6, 
                 verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = True, saveFlag: bool = True):
        '''
        Initialize the MOEAD algorithm with user-defined parameters.
        
        :param aggregation: The aggregation method to use.
        :param nPop: Population size.
        :param maxFEs: Maximum number of function evaluations.
        :param maxIterTimes: Maximum number of iterations.
        :param maxTolerateTimes: Maximum number of tolerated iterations without improvement.
        :param tolerate: Tolerance for improvement.
        :param verbose: Flag to enable verbose output.
        :param verboseFreq: Frequency of verbose output.
        :param logFlag: Flag to enable logging.
        :param saveFlag: Flag to enable saving results.
        '''
        
        # Initialize the base class with common parameters
        super().__init__(maxFEs, maxIterTimes, maxTolerateTimes, tolerate, 
                         verboseFlag, verboseFreq, logFlag, saveFlag)
        
        # Set specific parameters for MOEAD
        self.setPara('aggregation', aggregation)
        self.setPara('nPop', nPop)
        
    #-------------------Public Functions-----------------------#
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem):
        '''
        Execute the MOEAD algorithm on the specified problem.

        :param problem: An instance of a class derived from ProblemABC.
                        This object defines the optimization problem, including
                        the number of inputs (nInput), number of outputs (nOutput),
                        upper bounds (ub), lower bounds (lb), and evaluation methods.
        
        :return Result: An instance of the Result class, which contains the
                        optimization results, including the best decision variables,
                        objective values, and constraint violations encountered during
                        the optimization process.
        '''
        
        # Retrieve parameter values
        aggregation = self.getParaVal('aggregation')
        nPop = self.getParaVal('nPop')
        
        # Set the problem to solve
        self.setProblem(problem)
        
        # Initialize termination conditions
        self.FEs = 0
        self.iters = 0
        
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
        pop = self.initialize(nPop)
        
        # Initialize the ideal point
        Z = np.min(pop.objs, axis=0).reshape(1, -1)
         
        # Main loop of the algorithm
        while self.checkTermination():
            
            for i in range(nPop):
                
                # Select parents from the neighborhood
                P = B[i, np.random.permutation(B.shape[1])].ravel()

                # Generate offspring using genetic operations
                offspring = operationGAHalf(pop[P[0:2]], problem.ub, problem.lb, 1, 20, 1, 20)
                
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
                
                # Replace individuals in the population based on aggregation values
                pop.replace(P[g_old >= g_new], offspring)
                
            # Record the current state of the population
            self.record(pop)
    
        # Return the final result
        return self.result     