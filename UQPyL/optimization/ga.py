# Genetic Algorithm <Single>

import numpy as np
import math

from .algorithm import Algorithm, Population, Verbose
class GA(Algorithm):
    '''
        Genetic Algorithm <Single>
        -------------------------------
        Attributes:
            problem: Problem
                the problem you want to solve, including the following attributes:
                n_input: int
                    the input number of the problem
                ub: 1d-np.ndarray or float
                    the upper bound of the problem
                lb: 1d-np.ndarray or float
                    the lower bound of the problem
                evaluate: Callable
                    the function to evaluate the input
            n_samples: int, default=50
                the number of samples as the population
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
            run: run the Genetic Algorithm
        
        References:
            [1] D. E. Goldberg, Genetic Algorithms in Search, Optimization, and Machine Learning, 1989.
            [2] M. Mitchell, An Introduction to Genetic Algorithms, 1998.
            [3] D. Simon, Evolutionary Optimization Algorithms, 2013.
            [4] J. H. Holland, Adaptation in Natural and Artificial Systems, MIT Press, 1992.
    '''
    name= "Genetic Algorithm"
    type= "EA" #Evolutionary Algorithm
    target= "Single"
    def __init__(self, nInit: int=50, nPop: int=50,
                 proC: float=1, disC: float=20, proM: float=1, disM: float=20,
                 maxIterTimes: int=1000,
                 maxFEs: int=50000,
                 maxTolerateTimes: int=1000, tolerate: float=1e-6, 
                 verbose: bool=True, verboseFreq: int=100, logFlag: bool=False):
        
        super().__init__(maxFEs=maxFEs, maxIterTimes=maxIterTimes, 
                         maxTolerateTimes=maxTolerateTimes, tolerate=tolerate, 
                         verbose=verbose, verboseFreq=verboseFreq, logFlag=logFlag)
        
        #user-define setting
        self.setParameters('proC', proC)
        self.setParameters('disC', disC)
        self.setParameters('proM', proM)
        self.setParameters('disM', disM)
        self.setParameters('nInit', nInit)
        self.setParameters('nPop', nPop)
        
    #--------------------Public Functions---------------------#
    @Verbose.decoratorRun
    def run(self, problem, xInit=None, yInit=None):
        
        #Initialization
        #Parameter Setting
        proC, disC, proM, disM = self.getParaValue('proC', 'disC', 'proM', 'disM')
        nInit, nPop = self.getParaValue('nInit', 'nPop')
        
        #Problem
        self.problem=problem
        
        #Termination Condition Setting
        self.FEs=0; self.iters=0; self.tolerateTimes=0
        
        #Population Generation
        if xInit is not None:
            if yInit is not None:
                pop=Population(xInit, yInit)
            else:
                pop=Population(xInit)
                self.evaluate(pop)
        else:
            pop=self.initialize(nInit)
        
        pop=pop.getTop(nPop)
        
        #Record
        self.record(pop) 
        
        while self.checkTermination():
            
            matingPool=self._tournamentSelection(pop, 2)
            offspring=self._operationGA(matingPool, proC, disC, proM, disM)
            self.evaluate(offspring)
            
            pop=pop.merge(offspring)
            pop=pop.getTop(nPop)
            self.record(pop)
            
        return self.result
    #--------------------Private Functions--------------------#         
    def _tournamentSelection(self, pop, K: int=2):
        '''
            K-tournament selection
        '''
        
        rankIndex=pop.argsort()
        rank=np.argsort(rankIndex,axis=0)
        tourSelection=np.random.randint(0, high=len(pop), size=(len(pop), K))
        winner=np.min(rank[tourSelection].ravel().reshape(len(pop), K), axis=1)
        winnerIndex=rankIndex[winner]
        
        return pop[winnerIndex]
        
    def _operationGA(self, matingPool, proC, disC, proM, disM ):
        '''
            GA Operation: crossover and mutation
        '''
        
        n_samples=len(matingPool)
        # Crossover
        parent1=matingPool[:math.floor(n_samples/2)]
        parent2=matingPool[math.floor(n_samples/2):math.floor(n_samples/2)*2]
        
        n, d = parent1.size()
        beta = np.zeros(shape=(n,d))
        mu = np.random.rand(n, d)

        beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1 / (disC + 1))
        beta[mu > 0.5] = np.power(2 - 2 * mu[mu > 0.5], -1 / (disC + 1))
        beta = beta * (-1) ** np.random.randint(0, 2, size=(n, d))
        beta[np.random.rand(n, d) < 0.5] = 1
        beta[np.repeat(np.random.rand(n, 1) > proC, d, axis=1)] = 1

        off1=(parent1 + parent2) / 2 + (parent1 - parent2) * beta / 2
        off2=(parent1 + parent2) / 2 - (parent1 - parent2) * beta / 2 
        offspring=off1.merge(off2)
        
        # Polynomial mutation
        lower = np.repeat(self.problem.lb, 2 * n, axis=0)
        upper = np.repeat(self.problem.ub, 2 * n, axis=0)
        sita = np.random.rand(2 * n, d) < proM / d
        mu = np.random.rand(2 * n, d)
        
        offspring.clip(lower, upper)
        temp = sita & (mu <= 0.5)        
        t1 = (1 - 2 * mu[temp]) * np.power(1 - (offspring.decs[temp] - lower[temp]) / (upper[temp] - lower[temp]), disM + 1)
        offspring.decs[temp] = offspring.decs[temp] + (np.power(2 * mu[temp] + t1, 1 / (disM + 1)) - 1) *(upper[temp] - lower[temp])
        
        temp = sita & (mu > 0.5)
        t2 = 2 * (mu[temp] - 0.5) * np.power(1 - (upper[temp] - offspring.decs[temp]) / (upper[temp] - lower[temp]), disM + 1)
        offspring.decs[temp] = offspring.decs[temp] + (upper[temp] - lower[temp]) * (1 - np.power(2 * (1 - mu[temp]) + t2, 1 / (disM + 1)))
        
        return offspring