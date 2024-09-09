# Reference vector guided evolutionary algorithm (RVEA) <Multi>
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from ..utility_functions import uniformPoint, operationGA
from ..algorithm import Algorithm
from ..population import Population

class REVA(Algorithm):
    """
    Reference vector guided evolutionary algorithm (RVEA) <Multi>
    """
    
    def __init__(self, alpha: float=2.0, fr: float=0.1,
                nInit: int=50, nPop: int=50,
                maxFEs: int = 50000, 
                maxIterTimes: int = 1000, 
                maxTolerateTimes=None, tolerate=1e-6, 
                verbose=True, verboseFreq=10, logFlag=True):
        
        super().__init__(maxFEs, maxIterTimes, maxTolerateTimes, tolerate, verbose, verboseFreq, logFlag)
        
        self.setParameters('alpha', alpha)
        self.setParameters('fr', fr)
        self.setParameters('nInit', nInit)
        self.setParameters('nPop', nPop)
        
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
        V = V0
        
        while self.checkTermination():
            matingPool=np.random.randint(0, N, N)
            offspring = operationGA(matingPool, problem.ub, problem.lb)
    
    def environmentalSelection(self, pop, V, rate):
        popObjs=pop.objs
        n, m=pop.size()
        
        nV=V.shape[0]
        
        popObjs=popObjs - np.min(popObjs, axis=0)
        
        cosine=1-squareform(pdist(V, metric='cosine'))
        
        np.fill_diagonal(cosine, 0)
        
        gamma= np.min(np.arccos(cosine), axis=1)
        
        angle= np.arccos(1-cdist(popObjs, V, metric="cosine"))
        
        associate = np.argmin(angle, axis=1)
        
        
            
        
        
        