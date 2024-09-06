import numpy as np
from datetime import datetime
from prettytable import PrettyTable

from ..DoE import LHS
from ..problems import ProblemABC
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
    
class Population():
    def __init__(self, decs=None, objs=None):
        
        self.decs=np.copy(decs)
        self.objs=np.copy(objs)
        
        if decs is not None:
            self.nPop, self.D=decs.shape
        else:
            self.nPop=0; self.D=0
        if objs is None:
            self.evaluated=None
       
    def __add__(self, otherPop):
        # self.checkSameStatus(otherPop)
        if isinstance(otherPop, np.ndarray):
            return Population(self.decs+otherPop)
        
        return Population(self.decs+otherPop.decs)
    
    def __sub__(self, otherPop):
        
        if isinstance(otherPop, np.ndarray):
            return Population(self.decs-otherPop)
        
        return Population(self.decs-otherPop.decs)
    
    def __mul__(self, number):
        
        return Population(self.decs*number)
    
    def __rmul__(self, number):
        
        return Population(self.decs*number)
    
    def __truediv__(self, number):
        
        return Population(self.decs/number)
    
    def add(self, decs, objs):
        
        otherPop=Population(decs, objs)
        self.add(otherPop)

    def checkSameStatus(self, otherPop):
        
        if self.evaluated != otherPop.evaluated:
            raise Exception("The population evaluation status is different.")
    
    def checkEvaluated(self):
        
        if self.evaluate is False:
            raise Exception("The population is not evaluated yet.")
    
    def initialize(self, decs, objs):
        
        self.decs=decs
        self.objs=objs
        self.nPop, self.D=decs.shape
        
    def getTop(self, k):
        
        args=np.argsort(self.objs.ravel())

        return Population(self.decs[args[:k], :], self.objs[args[:k], :])
    
    def argsort(self):
        
        args=np.argsort(self.objs.ravel())
        
        return args
    
    def clip(self, lb, ub):
        
        self.decs=np.clip(self.decs, lb, ub)
    
    def replace(self, index, pop):
        
        self.decs[index, :]=pop.decs
        self.objs[index, :]=pop.objs
        
    def size(self):
        
        return self.nPop, self.D
    
    def evaluate(self, problem):
        
        self.objs=problem.evaluate(self.decs)
        self.evaluated=True
        
    def add(self, otherPop):
        
        if self.decs is not None:
            self.decs=np.vstack((self.decs, otherPop.decs))
            self.objs=np.vstack((self.objs, otherPop.objs))
        else:
            self.decs=otherPop.decs
            self.objs=otherPop.objs
            
        self.nPop=self.decs.shape[0]
    
    def merge(self, otherPop):
        
        self.add(otherPop)
        
        return self
    
    def __getitem__(self, index):
        
        if isinstance(index, (slice, list, np.ndarray)):
            decs = self.decs[index]
            objs = self.objs[index] if self.objs is not None else None
        elif isinstance(index, (int, np.integer)):
            decs = self.decs[index:index+1]
            objs = self.objs[index:index+1] if self.objs is not None else None
        else:
            raise TypeError("Index must be int, slice, list, or ndarray")
        
        return Population(decs, objs)

    def __len__(self):
        return self.nPop

class Setting():
    """
    Save the setting of the algorithm
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
    
class Result():
    def __init__(self):
        
        self.bestDec=None
        self.bestObj=None
        self.appearFEs=None
        self.appearIters=None
        self.historyBestDecs={}
        self.historyBestObjs={}
        self.historyDecs={}
        self.historyObjs={}
        self.historyFEs={}
    
    def update(self, pop, FEs, Iters):
        
        decs=np.copy(pop.decs); objs=np.copy(pop.objs)
        if self.bestObj==None or np.min(objs)<self.bestObj:
            ind=np.where(objs==np.min(objs))
            self.bestDec=decs[ind[0][0], :]
            self.bestObj=objs[ind[0][0], :]
            self.appearFEs=FEs
            self.appearIters=Iters
            
        self.historyFEs[FEs]=Iters
        self.historyDecs[FEs]=decs
        self.historyObjs[FEs]=objs
        self.historyBestDecs[FEs]=self.bestDec
        self.historyBestObjs[FEs]=self.bestObj
    
    def reset(self):
        
        self.bestDec=None; self.bestObj=None
        self.appearFEs=None; self.appearIters=None
        self.historyBestDecs={}; self.historyBestObjs={}
        self.historyDecs={}; self.historyObjs={}
        self.historyFEs={}