#multiobjective evolutionary algorithm based on decomposition (MOEAD) <Multi>
import numpy as np
import math
from typing import Optional, Literal
from scipy.spatial import distance

from ..algorithmABC import Algorithm
from ..population import Population
from ..utility_functions import uniformPoint, operationGAHalf, NDSort
from ...utility import Verbose
class MOEAD(Algorithm):
    '''
        Multi_objective Evolutionary Algorithm based on Decomposition <Multi>
        -------------------------------------------
        Parameters:
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
                the number of initial samples
            x_init: 2d-np.ndarray, default=None
                the initial input samples
            y_init: 2d-np.ndarray, default=None
                the initial output samples
            aggregation_type: Literal['PBI', 'TCH', 'TCH_N', 'TCH_M'], default='PBI'
                the type of aggregation function
            maxFEs: int, default=50000
                the maximum number of function evaluations
            maxIters: int, default=1000
                the maximum number of iterations
    '''
    
    name="MOEA/D"
    type="Multi"
    
    def __init__(self, aggregation: Literal['PBI', 'TCH', 'TCH_N', 'TCH_M']= 'PBI',
                nInit: int=50, nPop: int=50,
                maxFEs: int = 50000, 
                maxIterTimes: int = 1000, 
                maxTolerateTimes=None, tolerate=1e-6, 
                verbose=True, verboseFreq=10, logFlag=True, saveFlag=False):
        #problem setting
        super().__init__(maxFEs, maxIterTimes, maxTolerateTimes, tolerate, verbose, verboseFreq, logFlag, saveFlag)
        
        self.setParameters('aggregation', aggregation)
        self.setParameters('nInit', nInit)
        self.setParameters('nPop', nPop)
        
    #-------------------Public Functions-----------------------#
    @Verbose.decoratorRun
    def run(self, problem, xInit=None, yInit=None):
        
        #Parameter Setting
        aggregation=self.getParaValue('aggregation')
        nInit, nPop=self.getParaValue('nInit', 'nPop')
        #Problem
        self.setProblem(problem)
        #Termination Condition Setting
        self.FEs=0; self.iters=0
        #Population Generation
        if xInit is not None:
            pop = Population(xInit, yInit) if yInit is not None else Population(xInit)
            if yInit is None:
                self.evaluate(pop)
        else:
            pop = self.initialize(nInit)
        
        pop=pop.getTop(nPop)
        
        T = math.ceil(nPop / 10)
        W, N=uniformPoint(nPop, problem.nOutput)
        
        B=distance.cdist(W, W, metric='euclidean')
        B=np.argsort(B, axis=1)
        B=B[:,0:T]
        
        Z=np.min(pop.objs, axis=0).reshape(1,-1)
        
        while self.checkTermination():
            for i in range(N):
                
                P = B[i, np.random.permutation(B.shape[1])].ravel()

                offspring=operationGAHalf(pop[P[0:2]], problem.ub, problem.lb, 1, 20, 1, 20)
                self.evaluate(offspring)
                
                Z=np.min(np.vstack((Z, offspring.objs)), axis=0).reshape(1, -1)
                
                #PBI
                if(aggregation=='PBI'):
                    
                    normW = np.sqrt(np.sum(W[P, :]**2, axis=1))
                    normP = np.sqrt(np.sum((pop.objs[P] - np.tile(Z, (T, 1)))**2, axis=1))
                    normO = np.sqrt(np.sum((offspring.objs - Z)**2, axis=1))
                    CosineP = np.sum((pop.objs[P] - np.tile(Z, (T, 1))) * W[P, :], axis=1) / normW / normP
                    CosineO = np.sum(np.tile(offspring.objs - Z, (T, 1)) * W[P, :], axis=1) / normW / normO
                    g_old = normP * CosineP + 5 * normP * np.sqrt(1 - CosineP**2)
                    g_new = normO * CosineO + 5 * normO * np.sqrt(1 - CosineO**2)
                    
                elif(aggregation=='TCH'):
                    
                    g_old = np.max(np.abs(pop.objs[P] - np.tile(Z, (T, 1))) * W[P, :], axis=1)
                    g_new = np.max(np.tile(np.abs(offspring.objs- Z), (T, 1)) * W[P, :], axis=1)
                    
                elif(aggregation=='TCH_N'):
                    
                    Zmax = np.max(pop.objs, axis=0)
                    g_old = np.max(np.abs(pop.objs[P] - np.tile(Z, (T, 1))) / np.tile(Zmax - Z, (T, 1)) * W[P, :], axis=1)
                    g_new = np.max(np.tile(np.abs(offspring.objs - Z) / (Zmax - Z), (T, 1)) * W[P, :], axis=1)
                    
                elif(aggregation=='TCH_M'):
                    
                    g_old = np.max(np.abs(pop.objs[P] - np.tile(Z, (T, 1))) / W[P, :], axis=1)
                    g_new = np.max(np.tile(np.abs(offspring.objs - Z), (T, 1)) / W[P, :], axis=1)
                
                pop.replace(P[g_old >= g_new], offspring)
                
            self.record(pop)
    
        return self.result     