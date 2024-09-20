# Efficient global optimization
import numpy as np
from scipy.stats import norm

from .ga import GA
from ..algorithmABC import Algorithm, Population, Verbose
from ...problems import PracticalProblem
from ...surrogates import Surrogate
from ...surrogates.kriging import KRG
from ...utility.scalers import StandardScaler

class EGO(Algorithm):
    
    name="EGO"
    type="EA"
    def __init__(self, nInit: int=50,
                 maxFEs: int=1000,
                 maxTolerateTimes: int=100,
                 verbose: bool=True, verboseFreq: int=1, logFlag: bool=False, saveFlag=False
                ):
        super().__init__(maxFEs=maxFEs, maxTolerateTimes=maxTolerateTimes, verbose=verbose, verboseFreq=verboseFreq, logFlag=logFlag, saveFlag=saveFlag)
        
        self.setParameters('nInit', nInit)

        scaler=(StandardScaler(0, 1), StandardScaler(0, 1))
        surrogate=KRG()
        self.surrogate=surrogate
        
        optimizer=GA(maxFEs=10000, verbose=False)
        self.optimizer=optimizer
        self.optimizer.verbose=False
        
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem, xInit=None, yInit=None):
        
        #Initialization
        nInit=self.getParaValue('nInit')
        
        #Problem
        self.problem=problem
        
        #SubProblem
        subProblem=PracticalProblem(self.EI, problem.nInput, 1, problem.ub, problem.lb)
        
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
        
        self.record(pop)
        
        while self.checkTermination():
            # Build surrogate model
            self.surrogate.fit(pop.decs, pop.objs)
            res=self.optimizer.run(subProblem)
            
            offSpring=Population(decs=res.bestDec)
            self.evaluate(offSpring)
            pop.add(offSpring)
            
            self.record(pop)
    
        return self.result
    
    def EI(self, X):
        
        objs, mses=self.surrogate.predict(X, only_value=False)
        
        s=np.sqrt(mses)
        
        bestObj=self.result.bestObj
        
        ei= -(bestObj - objs) * norm.cdf((bestObj - objs) / s) - s * norm.pdf((bestObj - objs) / s)
        
        return ei
        
