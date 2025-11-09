import abc
import numpy as np

from .population import Population
from .result import Result
from ..doe import LHS
from ..util import Verbose

class AlgorithmABC(metaclass = abc.ABCMeta):
    """
    Baseclass for algorithms
    """
    def __init__(self, maxFEs: int = None, maxIters: int = None, maxTolerates: int = None, tolerate: float = 1e-6, 
                 verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = True, saveFlag: bool = False):
        
        self.setting = Setting()
        self.result = Result(self)
        
        self.problem = None
        self.maxFEs = maxFEs
        self.maxIter = maxIters
        self.maxTolerates = maxTolerates
        self.tolerate = tolerate
        
        self.verboseFlag = verboseFlag
        self.verboseFreq = verboseFreq
        self.logFlag = logFlag
        self.saveFlag = saveFlag
    
    def reset(self):
        
        self.FEs = 0; self.iters = 0; self.tolerateTimes = 0
        
        self.result.reset()
    
    def setup(self, problem, seed):
        
        self.setProblem(problem)
        
        self.reset()
        
        if seed is not None:
            np.random.seed(seed)
        else:
            seed = np.random.randint(0, 1000000)
            np.random.seed(seed)
        
        self.setParaVal('seed', seed)
    
    def initPop(self, nInit):
        
        lhs = LHS('classic')
        
        # TODO
        seed = np.random.randint(0, 1000000)
        xInit = lhs.sample(self.problem, nInit, seed)
        
        xInit = self.problem._transform_unit_X(xInit, IFlag = False, DFlag = False)
        
        pop = Population(xInit)
        
        self.evaluate(pop)
        
        return pop
    
    def setProblem(self, problem):
        
        self.problem = problem
        
        self.setParaVal('optType', problem.optType)
    
    def evaluate(self, pop):
        
        pop.evaluate(self.problem)
        
        self.FEs += pop.nPop
    
    def checkTermination(self, pop):
        
        signalFlag = False
        
        if self.FEs < self.maxFEs:
            if self.maxIter is None or self.iters <= self.maxIter:
                if self.maxTolerates is None or self.tolerateTimes <= self.maxTolerates:
                    
                    signalFlag = True
                    
                    # For GUI version
                    if hasattr(self.problem, 'GUI'):
                        self.problem.iterEmit.send()
                        if self.problem.isStop == True:
                            return False
                
        if self.verboseFlag > 0 or self.logFlag > 0 or self.saveFlag > 0 or not signalFlag or self.alg_type == 'EA':
            self.record(pop)
            
        # Check termination for single-objective optimization
        if self.problem.nOutput == 1 and self.tolerate is not None:
            if abs(self.result.bestObjs - pop.getBest().objs) > self.tolerate: 
                self.tolerateTimes = 0
            else:
                self.tolerateTimes += 1
        
        self.iters += 1
        
        return signalFlag
    
    def setProblem(self, problem):
        
        self.problem = problem
        self.optType = self.problem.optType
    
    def saveResult(self):
        
        if self.problem.nOutput > 1:
            self.result.save(alg_type = 1)
        else:
            self.result.save()
    
    @Verbose.record
    def record(self, pop):

        if self.problem.nOutput == 1:
            self.result.update(pop, self.problem, self.FEs, self.iters, 'EA')
        else:
            self.result.update(pop, self.problem, self.FEs, self.iters, 'MOEA')
                    
    def setParaVal(self, key, value):
        
        self.setting.setPara(key, value)
    
    def getParaVal(self, *args):
        
        return self.setting.getVal(*args)
    
class Setting():
    """
    Save the parameter setting of the algorithm
    """
    
    def __init__(self):
        self.keys = []
        self.values = []
        self.dicts = {}
    
    def setPara(self, key, value):
        
        self.dicts[key] = value
        self.keys.append(key)
        self.values.append(value)
    
    def getVal(self, *args):
        
        values = []
        for arg in args:
            values.append(self.dicts[arg])
        
        if len(args) > 1:
            return tuple(values)
        else:
            return values[0]