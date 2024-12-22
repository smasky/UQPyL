import numpy as np
import copy

from .utility_functions.ndsort import NDSort
from .utility_functions.crowding_distance import crowdingDistance
class Population():
    
    def __init__(self, decs, objs=None, cons=None):
        
        self.decs = np.atleast_2d(np.copy(decs))
        
        if objs is not None: 
            self.objs = np.atleast_2d(np.copy(objs))
            self.nOutput=self.objs.shape[1]
        
        if cons is not None:
            self.cons = np.atleast_2d(np.copy(cons))
        
        self.nPop, self.D = self.decs.shape
            
    def __add__(self, otherPop):
        
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
    
    def copy(self):
        
        return copy.deepcopy(self)
    
    def add(self, decs, objs):
        
        otherPop = Population(decs, objs)
        self.add(otherPop)
    
    def getBest(self, k=None):
        
        if k is None:
            if self.nOutput==1:
                iMax = np.argmax(self.objs)
                obj = self.objs[iMax]
                decs = self.decs[iMax]
                cons = self.cons[iMax]
                return Population(decs, obj, cons)
            
            else:
                frontNo, _ = NDSort(self)
                objs = self.objs[frontNo==1]
                decs = self.decs[frontNo==1]
                cons = self.cons[frontNo==1]
                return Population(decs, objs, cons)
            
        else:
            if self.nOutput==1:
                idx = self.argsort()
                return self[idx[:k]]
            
            else:
                frontNo, _ = NDSort(self)
                crowDis = crowdingDistance(self, frontNo)
                indices = np.lexsort((-crowDis, frontNo))
                objs = self.objs[indices[:k]]
                decs = self.decs[indices[:k]]
                cons = self.cons[indices[:k]] 
                return Population(decs, objs, cons)
            
    def argsort(self):
        
        if self.nOutput==1:
            
            popSumCon=np.sum(self.cons, axis=1).reshape(-1, 1)
            
            infeasible = (popSumCon > 0).astype(int)
            
            integration = self.objs + infeasible * popSumCon
            
            args = np.argsort(integration.ravel())
            
        else:
            
            frontNo, _ = NDSort(self)
            
            crowDis = crowdingDistance(self, frontNo)
            
            args = np.lexsort((-crowDis, frontNo))
        
        return args
    
    def clip(self, lb, ub):
        
        self.decs = np.clip(self.decs, lb, ub, out=self.decs)
    
    def replace(self, index, pop):
        
        self.decs[index, :] = pop.decs
        
        if pop.objs is not None:
            self.objs[index, :] = pop.objs
        
    def size(self):
        
        return self.nPop, self.D
    
    def evaluate(self, problem):
        
        decs=self.decs
        if problem.encoding=='mix':
            decs = problem._transform_discrete_var(np.copy(decs))
        
        self.nOutput=problem.nOutput
        
        self.objs = problem.evaluate(decs)
        self.cons = problem.constraint(decs)
        
    def add(self, otherPop):
        
        if self.decs is not None:
            self.decs=np.vstack((self.decs, otherPop.decs))
            self.objs=np.vstack((self.objs, otherPop.objs))
            self.cons=np.vstack((self.cons, otherPop.cons))
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
            cons = self.cons[index] if self.cons is not None else None
            
        elif isinstance(index, (int, np.integer)):
            decs = self.decs[index:index+1]
            objs = self.objs[index:index+1] if self.objs is not None else None
            cons = self.cons[index:index+1] if self.cons is not None else None
            
        else:
            raise TypeError("Index must be int, slice, list, or ndarray")
        
        return Population(decs, objs, cons)

    def __len__(self):
        
        return self.nPop