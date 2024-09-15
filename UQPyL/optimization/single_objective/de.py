# Differential Evolution <Single>

import numpy as np

from ..algorithmABC import Algorithm, Population
from ...utility import Verbose

class DE(Algorithm):
    """
    
    Reference:
    [1] Storn R , Price K .Differential Evolution (1997). A Simple and Efficient Heuristic for global Optimization over Continuous Spaces[J].
        Journal of Global Optimization, 11(4):341-359.DOI:10.1023/A:1008202821328.
    """
    
    name = "DE"
    type = "EA"
    
    def __init__(self, cr: float=0.9, f: float=0.5,
                 nInit: int=50, nPop: int=50,
                 maxFEs: int = 50000, 
                 maxIterTimes: int = 1000, 
                 maxTolerateTimes=None, tolerate=1e-6, 
                 verbose=True, verboseFreq=10, logFlag=True, saveFlag=False):
        
        super().__init__(maxFEs, maxIterTimes, maxTolerateTimes, tolerate, verbose, verboseFreq, logFlag, saveFlag)
        
        self.setParameters('cr', cr)
        self.setParameters('f', f)
        self.setParameters('nInit', nInit)
        self.setParameters('nPop', nPop)
        
    @Verbose.decoratorRun
    def run(self, problem, xInit=None, yInit=None):
        #Parameter Setting
        cr, f=self.getParaValue('cr', 'f')
        nInit, nPop=self.getParaValue('nInit', 'nPop')
        self.FEs=0; self.iters=0; self.tolerateTimes=0
        #Problem
        self.setProblem(problem)
        #Population Generation
        if xInit is not None:
            pop = Population(xInit, yInit) if yInit is not None else Population(xInit)
            if yInit is None:
                self.evaluate(pop)
        else:
            pop = self.initialize(nInit)
        pop=pop.getTop(nPop)
        
        while self.checkTermination():
            
            matingPool=self._tournamentSelection(pop, len(pop)*2, 2)
            offspring=self._operateDE(pop, matingPool[:len(pop)], matingPool[len(pop):], cr, f)
            self.evaluate(offspring)
            
            idx= offspring.objs.ravel()<pop.objs.ravel()
            pop.replace(idx, offspring[idx])
            self.record(pop)
        
        return self.result
            
    def _operateDE(self, pop1, pop2, pop3, cr, f):
        
        n, d=pop1.size()
        
        #DE
        sita = np.random.random((n,d)) < cr
        offspring = pop1.copy()
        offspring.decs[sita]= pop1.decs[sita] + (pop2.decs[sita] - pop3.decs[sita])*f
        
        return offspring
        
    def _tournamentSelection(self, pop, N, K: int=2):
        '''
            K-tournament selection
        '''
        
        rankIndex=pop.argsort()
        rank=np.argsort(rankIndex,axis=0)
        tourSelection=np.random.randint(0, high=len(pop), size=(N, K))
        winner=np.min(rank[tourSelection].ravel().reshape(N, K), axis=1)
        winnerIndex=rankIndex[winner]
        
        return pop[winnerIndex]