# Non-dominated Sorting Genetic Algorithm III (NSGA-III) <Multi>
import numpy as np

from ..algorithmABC import Algorithm
from ..population import Population
from ..utility_functions import tournamentSelection, uniformPoint, NDSort, crowdingDistance
from ..utility_functions.operation_GA import operationGA
from ...utility import Verbose
class NSGAIII(Algorithm):
    '''
    Non-dominated Sorting Genetic Algorithm III <Multi>
    This class implements a multi-objective genetic algorithm for optimization.
    
    Methods:
        run(problem, xInit=None, yInit=None): 
            Executes the NSGA-III algorithm on a given problem.
            - problem: Problem
                The problem to solve, which includes attributes like nInput, ub, lb, and evaluate.
    
    References:
        [1] K. Deb and H. Jain, An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems with Box Constraints, 2014.
    '''
    
    name = "NSGAIII"
    type = "MOEA"
    
    def __init__(self, proC: float=1.0, disC: float=20.0, proM: float=1.0, disM: float=20.0,
                 nPop: int=50,
                 maxFEs=50000, maxIterTimes=1000, 
                 maxTolerateTimes=None, tolerate=1e-6, 
                 verboseFlag: bool = True, verboseFreq: int = 10, 
                 logFlag: bool = True, saveFlag: bool = True):
        '''
        Initialize the NSGA-III algorithm with user-defined parameters.
        
        :param proC: Crossover probability.
        :param disC: Crossover distribution index.
        :param proM: Mutation probability.
        :param disM: Mutation distribution index.
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
        
        super().__init__(maxFEs, maxIterTimes, maxTolerateTimes, tolerate, 
                         verboseFlag, verboseFreq, logFlag, saveFlag)
        
        # Set user-defined parameters
        self.setPara('proC', proC)
        self.setPara('disC', disC)
        self.setPara('proM', proM)
        self.setPara('disM', disM)
        self.setPara('nPop', nPop)
        
    #-------------------------Public Functions------------------------#
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem):
        '''
        Execute the NSGA-III algorithm on the specified problem.

        :param problem: An instance of a class derived from ProblemABC.
                        This object defines the optimization problem, including
                        the number of inputs (nInput), number of outputs (nOutput),
                        upper bounds (ub), lower bounds (lb), and evaluation methods.
        
        :return Result: An instance of the Result class, which contains the
                        optimization results, including the best decision variables,
                        objective values, and constraint violations encountered during
                        the optimization process.
        '''
        
        # Parameter Setting
        proC, disC, proM, disM = self.getParaVal('proC', 'disC', 'proM', 'disM')
        nPop = self.getParaVal('nPop')
        
        # Set the problem to solve
        self.setProblem(problem)
        
        # Generate uniform reference points
        Z, nPop = uniformPoint(nPop, problem.nOutput)
        
        # Initialize termination conditions
        self.FEs = 0; self.iters = 0
        
        # Generate initial population
        pop = self.initialize(nPop)
        
        # Iterative process
        while self.checkTermination():
            
            # Perform non-dominated sorting
            frontNo, _ = NDSort(pop)
            # Calculate crowding distance
            crowdDis = crowdingDistance(pop, frontNo)
            # Select mating pool using tournament selection
            matingPool = tournamentSelection(pop, 2, len(pop), frontNo, -crowdDis)
            
            # Generate offspring using genetic operations
            offspring = operationGA(matingPool, self.problem.ub, self.problem.lb, proC, disC, proM, disM)
            
            # Evaluate the offspring
            self.evaluate(offspring)
            
            # Merge offspring with current population
            pop.merge(offspring)
            
            # Update the minimum objective values
            Zmin = np.min(pop.objs, axis=0).reshape(1,-1)
            
            # Select the best individuals to form the new population
            pop = self.environmentSelection(pop, Z, Zmin)
            
            # Record the current state of the population
            self.record(pop)
        
        # Return the final result
        return self.result
    
    def environmentSelection(self, pop, Z, Zmin):
        '''
        Perform environmental selection to choose the next generation.

        :param pop: Current population.
        :param Z: Reference points.
        :param Zmin: Minimum objective values.
        
        :return: Selected offspring for the next generation.
        '''
        
        N = Z.shape[0]
        
        # Perform non-dominated sorting
        frontNo, maxFNo = NDSort(pop, N)
        
        # Determine which individuals to keep
        next = frontNo < maxFNo
        
        # Identify the last front
        last = np.where(frontNo == maxFNo)[0]
        
        # Separate the population into selected and last front individuals
        popObjs1 = pop.objs[next]
        popObjs2 = pop.objs[last]
        
        # Select individuals from the last front
        choose = self.lastSelection(popObjs1, popObjs2, N - popObjs1.shape[0], Z, Zmin)
        
        # Update the selection
        next[last[choose]] = True
        
        # Return the selected offspring
        offSpring = pop[next]
        
        return offSpring
        
    def lastSelection(self, PopObj1, PopObj2, K, Z, Zmin):
        '''
        Select individuals from the last front based on reference points.

        :param PopObj1: Objective values of selected individuals.
        :param PopObj2: Objective values of individuals in the last front.
        :param K: Number of individuals to select.
        :param Z: Reference points.
        :param Zmin: Minimum objective values.
        
        :return: Boolean array indicating selected individuals.
        '''
        
        from scipy.spatial.distance import cdist
        PopObj = np.vstack((PopObj1, PopObj2)) - Zmin
        N, M = PopObj.shape
        N1 = PopObj1.shape[0]
        N2 = PopObj2.shape[0]
        NZ = Z.shape[0]

        # Normalization
        # Detect the extreme points
        Extreme = np.zeros(M, dtype=int)
        w = np.zeros((M, M)) + 1e-6 + np.eye(M)
        for i in range(M):
            Extreme[i] = np.argmin(np.max(PopObj / w[i], axis=1))

        # Calculate the intercepts of the hyperplane constructed by the extreme points
        try:
            Hyperplane = np.linalg.solve(PopObj[Extreme, :], np.ones(M))
        except np.linalg.LinAlgError:
            Hyperplane = np.ones(M)
        a = 1 / Hyperplane
        if np.any(np.isnan(a)):
            a = np.max(PopObj, axis=0)
        
        # Normalize PopObj
        PopObj = PopObj / a

        # Associate each solution with one reference point
        # Calculate the cosine similarity
        Cosine = 1 - cdist(PopObj, Z, 'cosine')
        Distance = np.sqrt(np.sum(PopObj**2, axis=1)).reshape(-1, 1) * np.sqrt(1 - Cosine**2)
        
        # Find the nearest reference point for each solution
        d = np.min(Distance, axis=1)
        pi = np.argmin(Distance, axis=1)

        # Calculate the number of associated solutions for each reference point
        rho = np.histogram(pi[:N1], bins=np.arange(NZ + 1))[0]

        # Environmental selection
        Choose = np.zeros(N2, dtype=bool)
        Zchoose = np.ones(NZ, dtype=bool)

        # Select K solutions one by one
        while np.sum(Choose) < K:
            # Find the least crowded reference point
            Temp = np.where(Zchoose)[0]
            if Temp.size == 0:
                break
            Jmin = Temp[np.where(rho[Temp] == np.min(rho[Temp]))[0]]
            j = Jmin[np.random.randint(len(Jmin))]

            # Find unselected solutions associated with this reference point
            I = np.where((~Choose) & (pi[N1:] == j))[0]

            if I.size > 0:
                if rho[j] == 0:
                    s = np.argmin(d[N1 + I])
                else:
                    s = np.random.choice(I.size)
                Choose[I[s]] = True
                rho[j] += 1
            else:
                Zchoose[j] = False

        return Choose
        
        
        
        
