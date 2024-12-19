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
    
    def __init__(self, cr: float = 0.9, f: float = 0.5,
                 nPop: int = 50,
                 maxFEs: int = 50000, 
                 maxIterTimes: int = 1000, 
                 maxTolerateTimes: int = 1000, tolerate: float = 1e-6, 
                 verbose=True, verboseFreq = 10, logFlag = False, saveFlag = True):
        
        super().__init__(maxFEs, maxIterTimes, maxTolerateTimes, tolerate, verbose, verboseFreq, logFlag, saveFlag)
        
        self.setParameters('cr', cr)
        self.setParameters('f', f)
        self.setParameters('nPop', nPop)
        
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem):
        
        #Parameter Setting
        cr, f = self.getParaValue('cr', 'f')
        nPop = self.getParaValue('nPop')
        
        #Termination Condition Setting
        self.FEs = 0; self.iters = 0; self.tolerateTimes = 0
        
        #Problem
        self.setProblem(problem)
        
        #Population Generation
        pop = self.initialize(nPop)
        
        while self.checkTermination():
            
            matingPool = self._tournamentSelection(pop, len(pop)*2, 2)
            
            offspring = self._operateDE(pop, matingPool[:len(pop)], matingPool[len(pop):], cr, f)
            
            self.evaluate(offspring)
            
            idx= offspring.objs.ravel()<pop.objs.ravel()
            pop.replace(idx, offspring[idx])
            
            self.record(pop)
        
        return self.result
            
    def _operateDE(self, pop1, pop2, pop3, cr, f):
        
        N, D = pop1.size()
        
        popDecs1 = pop1.decs
        popDecs2 = pop2.decs
        popDecs3 = pop3.decs
        
        #DE
        sita = np.random.random((N, D)) < cr
        offspringDecs = np.copy(popDecs1)
        offspringDecs[sita] = popDecs1[sita] + (popDecs2[sita] - popDecs3[sita]) * f
        
        return Population(offspringDecs)
        
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