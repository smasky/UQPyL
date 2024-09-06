import numpy as np

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