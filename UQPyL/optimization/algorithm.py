import numpy as np

from .population import Population
from ..DoE import LHS
from ..utility import Verbose

class Algorithm():
    """
    This is a baseclass for algorithms
    """
    def __init__(self, maxFEs, maxIterTimes, maxTolerateTimes=None, tolerate=1e-6, 
                 verbose=True, verboseFreq=10,
                 logFlag=True):
        
        self.setting=Setting()
        self.result=Result()
        
        self.problem=None
        self.maxFEs=maxFEs
        self.maxIter=maxIterTimes
        self.maxTolerateTimes=maxTolerateTimes
        self.tolerate=tolerate
        
        self.verbose=verbose
        self.verboseFreq=verboseFreq
        self.logFlag=logFlag

    def initialize(self, nInit):
        
        lhs=LHS('classic', problem=self.problem)
        xInit=lhs.sample(nInit, self.problem.n_input)
        pop=Population(xInit)
        self.evaluate(pop); 
            
        return pop
    
    def evaluate(self, pop):
        
        pop.evaluate(self.problem)
        self.FEs+=pop.nPop
    
    def checkTermination(self):
        
        if self.FEs<=self.maxFEs:
            if self.iters<=self.maxIter:
                if self.maxTolerateTimes is None or self.tolerateTimes<=self.maxTolerateTimes:
                    self.iters+=1
                    return True
        return False
    
    def setProblem(self, problem):
        self.problem=problem
    
    @Verbose.decoratorRecord
    def record(self, pop):
        
        self.result.update(pop, self.FEs, self.iters)

    def setParameters(self, key, value):
        
        self.setting.setParameter(key, value)
    
    def getParaValue(self, *args):
        
        return self.setting.getParaValue(*args)
    
class Setting():
    """
    Save the parameter setting of the algorithm
    """
    def __init__(self):
        self.keys=[]
        self.values=[]
        self.dicts={}
    
    def setParameter(self, key, value):
        
        self.dicts[key]=value
        self.keys.append(key)
        self.values.append(value)
    
    def getParaValue(self, *args):
        
        values=[]
        for arg in args:
            values.append(self.dicts[arg])
                
        return tuple(values)