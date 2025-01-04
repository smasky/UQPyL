import numpy as np
import copy

from .utility_functions.ndsort import NDSort
from .utility_functions.crowding_distance import crowdingDistance

class Population():
    
    def __init__(self, decs, objs = None, cons = None, conWgt = None):
        
        self.conWgt = conWgt
        
        self.decs = np.atleast_2d(np.copy(decs))
        
        if objs is not None: 
            self.objs = np.atleast_2d(np.copy(objs))
            self.nOutput = self.objs.shape[1]
        else:
            self.objs = None
            
        if cons is not None:
            self.cons = np.atleast_2d(np.copy(cons))
        else:
            self.cons = None
            
        self.nPop, self.D = self.decs.shape
    
    def copy(self):
        
        return copy.deepcopy(self)
    
    def add(self, decs, objs, cons = None):
        
        otherPop = Population(decs, objs, cons, self.conWgt)
        
        self.add(otherPop)
    
    def add(self, otherPop):
        
        if self.decs is not None:
            self.decs = np.vstack((self.decs, otherPop.decs))
            self.objs = np.vstack((self.objs, otherPop.objs))
            self.cons = np.vstack((self.cons, otherPop.cons)) if self.cons is not None else None
        else:
            self.decs = otherPop.decs
            self.objs = otherPop.objs
            self.cons = otherPop.cons
            
        self.nPop=self.decs.shape[0]
    
    def getBest(self, k=None):
        
        if k is None:
            if self.nOutput==1:
                iMax = np.argmax(self.objs)
                obj = self.objs[iMax]
                decs = self.decs[iMax]
                if cons != None:
                    cons = self.cons[iMax]
                return Population(decs, obj, cons, self.conWgt)
            
            else:
                frontNo, _ = NDSort(self)
                objs = self.objs[frontNo==1]
                decs = self.decs[frontNo==1]
                cons = self.cons[frontNo==1] if self.cons is not None else None
                return Population(decs, objs, cons, self.conWgt)
            
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
                return Population(decs, objs, cons, self.conWgt)
            
    def argsort(self):
        
        if self.nOutput == 1:
            
            if self.cons is not None:
                
                popCons_ = np.maximum(0 , self.cons)
                
                popCons = popCons_ * self.conWgt if self.conWgt is not None else popCons_ * 1e6
                
                popSumCon = np.sum( popCons, axis=1 ).reshape(-1, 1)
                
                infeasible = (popSumCon > 0).astype(int)
                
            integration = self.objs + infeasible * popSumCon if self.cons is not None else self.objs
                
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
        
        decs = self.decs
        
        if problem.encoding == 'mix':
            decs = problem._transform_discrete_var(np.copy(decs))
        
        self.nOutput = problem.nOutput
        
        res = problem.evaluate(decs)
        
        self.objs, self.cons = res['objs'], res['cons']
        
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
        
        return Population(decs, objs, cons, self.conWgt)

    def __len__(self):
        
        return self.nPop