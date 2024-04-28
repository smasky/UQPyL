import numpy as np
import math
from typing import Callable

class GA():
    '''
        Genetic Algorithm
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
    type="EA" #Evolutionary Algorithm
    def __init__(self, problem, n_samples: int=50,
                 proC: float=1, disC: float=20, proM: float=1, disM: float=20,
                 maxIterTimes: int=1000,
                 maxFEs: int=50000,
                 maxTolerateTimes: int=1000,
                 tolerate=1e-6):
        
        self.n_input=problem.n_input
        self.ub=problem.ub.reshape(1,-1);self.lb=problem.lb.reshape(1,-1)
        self.evaluate=problem.evaluate
        
        self.proC=proC;self.disC=disC
        self.proM=proM;self.disM=disM
        self.tolerate=1e-6
        self.maxTolerateTimes=maxTolerateTimes
        self.n_samples=n_samples
        self.maxIterTimes=maxIterTimes
        self.maxFEs=maxFEs
        self.tolerate=tolerate
    #--------------------------Public Functions--------------------------#
    def run(self) -> dict:
        '''
            Run the Genetic Algorithm
            -------------------------------
            Returns:
                Result: dict
                    the result of the Genetic Algorithm, including the following keys:
                    best_decs: 2d-np.ndarray
                        the decision variables of the best solution
                    best_objs: 2d-np.ndarray
                        the objective values of the best solution
                    history_decs: 2d-np.ndarray
                        the best decision variables of each iteration
                    history_objs: 2d-np.ndarray
                        the best objective values of each iteration
                    iter: int
                        the iteration times of the Genetic Algorithm
                    fe: int
                        the function evaluations of the Genetic Algorithm
        '''
        best_objs=np.inf
        best_decs=None
        time=1
        iter=0
        fe=0
        
        decs=np.random.random((self.n_samples,self.n_input))*(self.ub-self.lb)+self.lb
        objs=self.evaluate(decs)
        fe+=objs.shape[0]
        
        history_decs=[]
        history_objs=[]
        Result={}
        while iter<self.maxIterTimes and fe<self.maxFEs and time<=self.maxTolerateTimes:
            
            matingPool=self._tournamentSelection(decs,objs,2)
            matingDecs=self._operationGA(matingPool)
            matingObjs=self.evaluate(matingDecs)
            fe+=matingObjs.shape[0]
            
            tempObjs=np.vstack((objs,matingObjs))
            tempDecs=np.vstack((decs,matingDecs))
            rank=np.argsort(tempObjs,axis=0)
            decs=tempDecs[rank[:self.n_samples,0],:]
            objs=tempObjs[rank[:self.n_samples,0],:]
            
            if(abs(best_objs-np.min(objs))>self.tolerate):
                best_objs=np.min(objs)
                best_decs=decs[np.argmin(objs,axis=0),:]
                time=0
            else:
                time+=1
            
            history_decs.append(best_decs)
            history_objs.append(best_objs)
                  
            iter+=1
        
        Result['best_dec']=best_decs
        Result['best_obj']=best_objs
        Result['history_decs']=np.vstack(history_decs)
        Result['history_objs']=np.array(history_objs).reshape(-1,1)
        Result['iters']=iter
        Result['FEs']=fe
        
        return Result
    #--------------------Private Functions--------------------# 
    def _tournamentSelection(self,decs: np.ndarray, objs: np.ndarray, K: int=2):
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
    
    
            

        