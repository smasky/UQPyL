import numpy as np
import copy

from .util import NDSort, crowdingDist

class Population():
    
    def __init__(self, decs, objs = None, cons = None, conWgt = None, optType: str = 'min'):
        
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
        
        self.frontNo = None
        self.crowdDis = None
    
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
    
    def getBest(self, k: int = 1):
        
        '''
        Get the `k` best individual in the population.
        '''
        
        if self.nOutput == 1:
            return self._bestSingle(k)
        else:
            return self._bestMulti()
    
    def _bestSingle(self, k: int = None):
                
        args = self.argsort()
        
        Idx = args[:k] if k is not None else args[:1]
        
        return Population(self.decs[Idx],
                          self.objs[Idx],
                          self.cons[Idx] if self.cons is not None else None,
                          self.conWgt)
    
    def _bestMulti(self, k: int = None):
        
        if self.frontNo is None:
            frontNo, _ = NDSort(self.objs, self.cons)
        else:
            frontNo = self.frontNo
        
        if self.cons is not None:
            
            CV = self.conWgt * self.cons if self.conWgt is not None else self.cons
            CV = np.sum(np.maximum(0, CV), axis=1)
            feasible = CV <= 0
            
            feasiblePop = self[feasible]
            
            if len(feasiblePop) > 0:
                nonDominated = frontNo == 1
                bestPop = feasiblePop[nonDominated]
            
            else:
                sortedIdx = np.argsort(CV)
                k = 10 if k is None else k
                bestPop = self[sortedIdx[:k]]
                return bestPop
        else:
            nonDominated = frontNo == 1
            bestPop = self[nonDominated]
        
        if k is not None and len(bestPop) > k:
            if self.crowdDis is None:
                crowDis = crowdingDist(self.objs, frontNo)
            else:
                crowDis = self.crowdDis
            sortedIdx = np.lexsort((-crowDis, frontNo))
            bestPop = self[sortedIdx[:k]]
        
        return bestPop
    
    def getParetoFront(self):
        
        if self.cons is not None:
            
            CV = self.conWgt * self.cons if self.conWgt is not None else self.cons
            CV = np.sum(np.maximum(0, CV), axis=1)
            feasible = CV <= 0
            
            feasiblePop = self[feasible]
            
            if len(feasiblePop) > 0:
                frontNo, _ = NDSort(feasiblePop.objs, feasiblePop.cons)
                nonDominated = frontNo == 1
                bestPop = feasiblePop[nonDominated]
                
            else:
                sortedIdx = np.argsort(CV)
                bestPop = self[sortedIdx[:10]]
                return bestPop
        else:
            frontNo, _ = NDSort(self.objs, self.cons)
            nonDominated = frontNo == 1
            bestPop = self[nonDominated]

        return bestPop
    
    def argsort(self):
        
        if self.nOutput == 1:
            
            if self.cons is not None:
                # TODO: validate this
                infeasible = (self.cons > 0).any(axis=1).astype(int).reshape(-1, 1)
                
                viol = np.maximum(0.0, self.cons)
                violMax = np.max(viol, axis=0); violMin = np.min(viol, axis=0)
                denom = violMax - violMin + 1e-12
                viol_norm = (viol - violMin) / denom * 1e6
                              
                viol_weighted = viol_norm * self.conWgt if self.conWgt is not None else viol_norm
                
                popSumCon = viol_weighted.sum(axis=1, keepdims=True)
                 
            integration = self.objs + infeasible * popSumCon if self.cons is not None else self.objs
                
            args = np.argsort(integration.ravel())
            
        else:
            
            frontNo, _ = NDSort(self.objs, self.cons)
            
            crowDis = crowdingDist(self.objs, frontNo)
            
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
        
        decs = np.copy(self.decs)
        
        if problem.encoding == 'mix':
            decs = problem._transform_discrete_var(decs)
            decs = problem._transform_int_var(decs)

        self.nOutput = problem.nOutput
        
        res = problem.evaluate(decs)
        
        self.objs, self.cons = res['objs'], res['cons']
        
        self.objs = self.objs * problem.opt  # TODO: min
        
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