import numpy as np
import math

from ..problems import Problem
from ..DoE import LHS
from .optimizer import Optimizer, verboseForRun 
class GA(Optimizer):
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
    name="Genetic Algorithm"
    type="EA" #Evolutionary Algorithm
    def __init__(self, problem, nInit: int=50, nPop: int=50,
                 x_init=None, y_init=None,
                 proC: float=1, disC: float=20, proM: float=1, disM: float=20,
                 maxIterTimes: int=1000,
                 maxFEs: int=50000,
                 maxTolerateTimes: int=1000,
                 tolerate=1e-6,
                 verbose=True,
                 logFlag=False):
        #problem setting
        self.n_input=problem.n_input
        self.ub=problem.ub.reshape(1,-1);self.lb=problem.lb.reshape(1,-1)
        
        #algorithm setting
        self.proC=proC;self.disC=disC
        self.proM=proM;self.disM=disM
        self.tolerate=tolerate
        self.nInit=nInit
        self.nPop=nPop
        
        self.x_init=x_init
        self.y_init=y_init
        
        #termination setting
        self.maxTolerateTimes=maxTolerateTimes
        self.maxIterTimes=maxIterTimes
        self.maxFEs=maxFEs
        
        #setting record
        setting={}
        setting["nPop"]=nPop
        setting["nInit"]=nInit
        setting["proC"]=proC
        setting["disC"]=disC
        setting["proM"]=proM
        setting["proC"]=proC
        setting["maxFEs"]=maxFEs
        setting["maxIterTimes"]=maxIterTimes
        setting["maxTolerateTimes"]=maxTolerateTimes
        self.setting=setting
        
        super().__init__(problem=problem, maxFEs=maxFEs, maxIter=maxIterTimes, 
                         maxTolerateTimes=maxTolerateTimes, tolerate=tolerate, verbose=verbose, logFlag=logFlag)
    #--------------------Public Functions---------------------#
    @verboseForRun  
    def run(self, xInit=None, yInit=None):
        
        if xInit is None:
            lhs=LHS('classic', problem=self.problem)
            xInit=lhs.sample(self.nInit, self.n_input)
        
        if yInit is None:
            yInit=self.evaluate(xInit)
            
        self.update(xInit, yInit)
        decs=xInit; objs=yInit
        
        while self.checkTermination():
            decs, objs=self.evolve(decs, objs)
            self.update(decs, objs)
        return self.database
    #--------------------Private Functions--------------------# 
    def evolve(self, decs, objs):
        
        matingPool=self._tournamentSelection(decs,objs,2)
        matingDecs=self._operationGA(matingPool)
        matingObjs=self.evaluate(matingDecs)
        self.FEs+=matingDecs.shape[0]
        
        decs=np.vstack((decs,matingDecs))
        objs=np.vstack((objs,matingObjs))
        sorted_indices=np.argsort(objs.ravel())
        objs=objs[sorted_indices[:self.nPop], :]
        decs=decs[sorted_indices[:self.nPop], :]
        
        return decs, objs
        
    def _tournamentSelection(self, decs: np.ndarray, objs: np.ndarray, K: int=2):
        '''
            K-tournament selection
        '''
        rankIndex=np.argsort(objs,axis=0)
        rank=np.argsort(rankIndex,axis=0)
        
        tourSelection=np.random.randint(0,high=objs.shape[0],size=(objs.shape[0],K))
        winner=np.min(rank[tourSelection,:].ravel().reshape(objs.shape[0],2),axis=1)
        winIndex=rankIndex[winner]
        
        return decs[winIndex.ravel(),:]
        
    def _operationGA(self,decs: np.ndarray):
        '''
            GA Operation: crossover and mutation
        '''
        n_samples=decs.shape[0]
        parent1=decs[:math.floor(n_samples/2),:]
        parent2=decs[math.floor(n_samples/2):math.floor(n_samples/2)*2,:]
        
        n, d = parent1.shape
        beta = np.zeros_like(parent1)
        mu = np.random.rand(n, d)

        beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1 / (self.disC + 1))
        beta[mu > 0.5] = np.power(2 - 2 * mu[mu > 0.5], -1 / (self.disC + 1))
        beta = beta * (-1) ** np.random.randint(0, 2, size=(n, d))
        beta[np.random.rand(n, d) < 0.5] = 1
        beta[np.repeat(np.random.rand(n, 1) > self.proC, d, axis=1)] = 1

        offspring = np.concatenate(( (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2,
                              (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2 ), axis=0)

        lower = np.repeat(self.lb, 2 * n, axis=0)
        upper = np.repeat(self.ub, 2 * n, axis=0)
        site = np.random.rand(2 * n, d) < self.proM / d
        mu = np.random.rand(2 * n, d)
        
        temp = site & (mu <= 0.5)
        offspring = np.clip(offspring, lower, upper)
        t1 = (1 - 2 * mu[temp]) * np.power(1 - (offspring[temp] - lower[temp]) / (upper[temp] - lower[temp]), self.disM + 1)
        offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * (np.power(2 * mu[temp] + t1, 1 / (self.disM + 1)) - 1)
        
        temp = site & (mu > 0.5)
        t2 = 2 * (mu[temp] - 0.5) * np.power(1 - (upper[temp] - offspring[temp]) / (upper[temp] - lower[temp]), self.disM + 1)
        offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * (1 - np.power(2 * (1 - mu[temp]) + t2, 1 / (self.disM + 1)))
        
        return offspring        