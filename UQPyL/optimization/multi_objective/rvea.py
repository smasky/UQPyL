# Reference vector guided evolutionary algorithm (RVEA) <Multi>
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from ..utility_functions import uniformPoint, operationGA
from ..algorithmABC import Algorithm
from ..population import Population
from ...utility import Verbose
class RVEA(Algorithm):
    """
    Reference vector guided evolutionary algorithm (RVEA) <Multi>
    """
    name="RVEA"
    type="MOEA"
    
    def __init__(self, alpha: float=2.0, fr: float=0.1,
                nInit: int=50, nPop: int=50,
                maxFEs: int = 50000, 
                maxIterTimes: int = 1000, 
                maxTolerateTimes=None, tolerate=1e-6, 
                verbose=True, verboseFreq=10, logFlag=True, saveFlag=False):
        
        super().__init__(maxFEs, maxIterTimes, maxTolerateTimes, tolerate, verbose, verboseFreq, logFlag, saveFlag)
        
        self.setParameters('alpha', alpha)
        self.setParameters('fr', fr)
        self.setParameters('nInit', nInit)
        self.setParameters('nPop', nPop)
    
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem, xInit=None, yInit=None):
        
        #Parameters setting
        alpha, fr=self.getParaValue('alpha', 'fr')
        nInit, nPop=self.getParaValue('nInit', 'nPop')
        
        #Problem
        self.setProblem(problem)
        #Termination Condition Setting
        self.FEs=0; self.iters=0
        if xInit is not None:
            pop = Population(xInit, yInit) if yInit is not None else Population(xInit)
            if yInit is None:
                self.evaluate(pop)
        else:
            pop = self.initialize(nInit)
        
        pop=pop.getTop(nPop)
        V0, N=uniformPoint(nPop, problem.nOutput)
        V = np.copy(V0)
        
        while self.checkTermination():
            
            matingPool=np.random.randint(0, len(pop), N)
            
            offspring = operationGA(pop[matingPool], problem.ub, problem.lb)
            
            self.evaluate(offspring)
            
            pop = self.environmentalSelection(pop.merge(offspring), V, (self.FEs/self.maxFEs)**alpha)
            
            condition= not (np.ceil(self.FEs / N) % np.ceil(fr * self.maxFEs / N))
            
            if condition:
                
                V = self.updateReferenceVector(pop, V0)
            
            self.record(pop)
        
        return self.result
    
    def updateReferenceVector(self, pop, V):
        
        scaling_factors = np.max(pop.objs, axis=0) - np.min(pop.objs, axis=0)
        
        V = V * scaling_factors
        
        return V
    
    def environmentalSelection(self, pop, V, theta):
        
        popObjs=pop.objs
        
        m=pop.nOutput
        
        nV=V.shape[0]
        
        popObjs=popObjs - np.min(popObjs, axis=0)
        
        cosine=1-cdist(V, V, metric='cosine')
        
        np.fill_diagonal(cosine, 0)
        
        gamma= np.min(np.arccos(cosine), axis=1)
        
        angle= np.arccos(1-cdist(popObjs, V, metric="cosine"))
        
        associate = np.argmin(angle, axis=1)
        
        next = np.ones(nV, dtype=np.int32)*-1
        
        for i in np.unique(associate):
            current1 = np.where(associate == i)[0]
            # current2 = np.where(associate == i)[0]
            
            if len(current1) > 0:
                # Calculate the APD value for each solution
                APD = (1 + m * theta * angle[current1, i] / gamma[i]) * np.sqrt(np.sum(popObjs[current1, :]**2, axis=1))
                # Select the one with the minimum APD value
                best = np.argmin(APD)
                next[i] = current1[best]
                
        return pop[next[next != -1].astype(int)]    