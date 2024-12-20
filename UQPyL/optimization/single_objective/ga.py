# Genetic Algorithm <Single>

import numpy as np
import math

from ..algorithmABC import Algorithm, Verbose, Result
from ..population import Population
class GA(Algorithm):
    '''
        Genetic Algorithm <single> <real>/<mix>
        -------------------------------
        Attributes:
        
            nPop: int, default=50
                the population size of the algorithm
            proC: float, default=1
                the probability of crossover
            disC: float, default=20
                the distribution index of crossover
            proM: float, default=1
                the probability of mutation
            disM: float, default=20
                the distribution index of mutation
                
            maxIterTimes: int, default=10000
                the maximum iteration times
            maxFEs: int, default=2000000
                the maximum function evaluations
            maxTolerateTimes: int, default=1000
                the maximum tolerate times which the best objective value does not change
            tolerate: float, default=1e-6
                the tolerate value which the best objective value does not change
        
        Methods:
            run(problem): 
                run the algorithm
                - problem: Problem
                the problem you want to solve, including the following attributes:
                
                    nInput: int
                        the input number of the problem
                    ub: 1d-np.ndarray or float
                        the upper bound of the problem
                    lb: 1d-np.ndarray or float
                        the lower bound of the problem
                    evaluate: Callable
                        the function to evaluate the input
                        
                    Optional:
                    var_type: np.array
                        the type of variables of the problem
                    var_set: list
                        the sets of discrete variables of the problem
        
        References:
            [1] D. E. Goldberg, Genetic Algorithms in Search, Optimization, and Machine Learning, 1989.
            [2] M. Mitchell, An Introduction to Genetic Algorithms, 1998.
            [3] D. Simon, Evolutionary Optimization Algorithms, 2013.
            [4] J. H. Holland, Adaptation in Natural and Artificial Systems, MIT Press, 1992.
    '''
    
    name = "GA"
    type = "EA"
    
    def __init__(self, nPop: int = 50,
                 proC: float = 1, disC: float = 20, proM: float = 1, disM: float = 20,
                 maxIterTimes: int = 1000,
                 maxFEs: int = 50000,
                 maxTolerateTimes: int = 1000, tolerate: float = 1e-6,
                 verbose: bool = True, verboseFreq: int = 10, logFlag: bool = False, saveFlag = True):
        
        super().__init__(maxFEs=maxFEs, maxIterTimes=maxIterTimes, 
                         maxTolerateTimes=maxTolerateTimes, tolerate=tolerate,
                         verbose=verbose, verboseFreq=verboseFreq, logFlag=logFlag, saveFlag=saveFlag)
        
        #user-define setting
        self.setParameters('proC', proC)
        self.setParameters('disC', disC)
        self.setParameters('proM', proM)
        self.setParameters('disM', disM)
        self.setParameters('nPop', nPop)
        
    #--------------------Public Functions---------------------#
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem):
        
        #Initialization
        #Parameter Setting
        proC, disC, proM, disM = self.getParaValue('proC', 'disC', 'proM', 'disM')
        nPop = self.getParaValue('nPop')
        
        #Problem
        self.problem = problem
        
        #Termination Condition Setting
        self.FEs = 0; self.iters = 0; self.tolerateTimes = 0
        
        #Population Generation
        pop = self.initialize(nPop)
        
        #Record
        self.record(pop) 
        
        #Iterative
        while self.checkTermination():
            
            matingPool = self._tournamentSelection(pop, 2)
            
            offspring = self._operationGA(matingPool, proC, disC, proM, disM)
            
            self.evaluate(offspring)
            
            pop = pop.merge(offspring)
            
            pop = pop.getBest(nPop)
            
            self.record(pop)
            
        return self.result
    
    #--------------------Private Functions--------------------#         
    def _tournamentSelection(self, pop, K: int=2):
        '''
            K-tournament selection
        '''
        
        rankIndex = pop.argsort()
        rank = np.argsort(rankIndex, axis=0)
        tourSelection = np.random.randint(0, high=len(pop), size=(len(pop), K))
        winner = np.min(rank[tourSelection].ravel().reshape(len(pop), K), axis=1)
        winnerIndex = rankIndex[winner]
        
        return pop[winnerIndex]
        
    def _operationGA(self, matingPool, proC, disC, proM, disM ):
        '''
            GA Operation: crossover and mutation
        '''
        popDec = matingPool.decs
        
        NN = len(matingPool)
        
        # Crossover
        parent1 = popDec[:math.floor(NN/2)]
        parent2 = popDec[math.floor(NN/2):math.floor(NN/2)*2]
        
        N, D = parent1.shape
        beta = np.zeros(shape=(N, D))
        mu = np.random.rand(N, D)

        beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1 / (disC + 1))
        beta[mu > 0.5] = np.power(2 - 2 * mu[mu > 0.5], -1 / (disC + 1))
        beta = beta * (-1) ** np.random.randint(0, 2, size=(N, D))
        beta[np.random.rand(N, D) < 0.5] = 1
        beta[np.repeat(np.random.rand(N, 1) > proC, D, axis=1)] = 1

        off1 = (parent1 + parent2) / 2 + (parent1 - parent2) * beta / 2
        off2 = (parent1 + parent2) / 2 - (parent1 - parent2) * beta / 2 
        
        offspring=np.vstack((off1, off2))
        
        # Polynomial mutation
        lower = np.repeat(self.problem.lb, 2 * N, axis=0)
        upper = np.repeat(self.problem.ub, 2 * N, axis=0)
        sita = np.random.rand(2 * N, D) < proM / D
        mu = np.random.rand(2 * N, D)
        
        np.clip(offspring, lower, upper, out=offspring)
        
        temp = sita & (mu <= 0.5)        
        t1 = (1 - 2 * mu[temp]) * np.power(1 - (offspring[temp] - lower[temp]) / (upper[temp] - lower[temp]), disM + 1)
        offspring[temp] = offspring[temp] + (np.power(2 * mu[temp] + t1, 1 / (disM + 1)) - 1) *(upper[temp] - lower[temp])
        
        temp = sita & (mu > 0.5)
        t2 = 2 * (mu[temp] - 0.5) * np.power(1 - (upper[temp] - offspring[temp]) / (upper[temp] - lower[temp]), disM + 1)
        offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * (1 - np.power(2 * (1 - mu[temp]) + t2, 1 / (disM + 1)))
        
        return Population(offspring)