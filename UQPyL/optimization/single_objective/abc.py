# Artificial Bee Colony Algorithm <Single>
import numpy as np

from ..algorithmABC import Algorithm, Population
from ...utility import Verbose

class ABC(Algorithm):
    """
    Artificial Bee Colony Algorithm (ABC) <Single>
    """
    
    name="ABC"
    type="EA"
    
    def __init__(self, employedRate: float=0.3,  limit: int=50,
                 nInit: int=50, nPop: int=50, 
                 maxFEs: int = 50000, 
                 maxIterTimes: int = 1000, 
                 maxTolerateTimes=None, tolerate=1e-6, 
                 verbose=True, verboseFreq=10, logFlag=True, saveFlag=False):
                
        super().__init__(maxFEs, maxIterTimes, maxTolerateTimes, tolerate, verbose, verboseFreq, logFlag, saveFlag)
        
        self.setParameters('employedRate', employedRate)
        self.setParameters('limit', limit)
        self.setParameters('nInit', nInit)
        self.setParameters('nPop', nPop)
    
    @Verbose.decoratorRun
    def run(self, problem, xInit=None, yInit=None):
        #Parameter Setting
        employedRate, limit = self.getParaValue('employedRate', 'limit')
        nInit, nPop = self.getParaValue('nInit', 'nPop')
        #Problem
        self.setProblem(problem)
        self.FEs=0; self.iters=0
        #Population Generation
        if xInit is not None:
            pop = Population(xInit, yInit) if yInit is not None else Population(xInit)
            if yInit is None:
                self.evaluate(pop)
        else:
            pop = self.initialize(nInit)
            
        pop=pop.getTop(nPop)
        beeType=np.zeros(nPop, dtype=np.int32)
        limitCount=np.zeros(nPop)
        
        while self.checkTermination():
            
            pop, beeType=self.setEmployedBees(beeType, pop, employedRate)
            
            pop, limitCount=self.updateEmployedBees(pop, beeType, limitCount)
            
            pop, beeType, limitCount=self.updateUnemployedBees(pop, beeType, limitCount)
            
            pop, beeType, limitCount=self.updateOnlookerBees(pop, beeType, limitCount, employedRate)
            
            beeType=self.checkLimitTimes(beeType, limitCount, limit)
            
            self.record(pop)
         
        return self.result
            
    def checkLimitTimes(self, beeType:np.ndarray, limitCount: np.ndarray, limit: int):
        onlookerBees=np.where(limitCount>limit)[0]
        beeType[onlookerBees]=2
        
        return beeType        
    
    def updateOnlookerBees(self, pop: Population, beeType: np.ndarray, limitCount: np.ndarray, employedRate: float):
        maxNEmployed=int(len(pop)*employedRate)
        if(np.sum(beeType==2)>0):
            onlookerIdx=np.where(beeType==2)[0]
            onlookerBees=pop[onlookerIdx]
            n, d=onlookerBees.size()
            
            onlookerBees.decs=np.random.random((n,d))*(self.problem.ub-self.problem.lb)+self.problem.lb
            
            self.evaluate(onlookerBees)
            
            onlookerBees=onlookerBees[onlookerBees.argsort()]
            
            pop.replace(beeType==2, onlookerBees)
            
            nEmployed=np.sum(beeType==1)
            
            limitCount[onlookerIdx]=0
            beeType[onlookerIdx]=0
            if nEmployed<maxNEmployed:
                pop.replace(onlookerIdx, onlookerBees)
                beeType[onlookerIdx[:maxNEmployed-nEmployed]]=1
        
        return pop, beeType, limitCount
            
    def updateUnemployedBees(self, pop: Population, beeType: np.ndarray, limitCount: np.ndarray):
        
        n, d=pop.size()
        
        employedType=np.where(beeType==1)[0]
        unemployedType=np.where(beeType==0)[0]
        
        employedBees=pop[employedType]
        unemployedBees=pop[unemployedType]
        
        idx=employedBees.argsort()
        nEmployed=len(employedBees)
        p=2*(nEmployed+1.0-np.linspace(1, nEmployed, nEmployed))/((nEmployed+1)*nEmployed)
        p[idx]=p/np.sum(p)
        
        globalIdx=np.random.choice(len(employedBees), len(unemployedBees), p=p)

        idx=np.arange(len(pop))
        while True:
            randIdx=np.random.permutation(idx)
            if(np.all(randIdx[beeType==0]!=idx[beeType==1][globalIdx])):
                break
        
        rnd=np.random.random((len(unemployedBees), d))*2-1
        newBees=employedBees[globalIdx]+(employedBees[globalIdx]-pop[randIdx[beeType==0]])*rnd
        newBees.clip(self.problem.lb, self.problem.ub)
        
        self.evaluate(newBees)
        
        pop.replace(beeType==0, newBees)
        
        replaceIdx=np.where(newBees.objs<employedBees[globalIdx].objs)[0]
        limitCount[unemployedType[replaceIdx]]=0
        beeType[unemployedType[replaceIdx]]=1
        limitCount[employedType[globalIdx][replaceIdx]]=0
        beeType[employedType[globalIdx][replaceIdx]]=0
        
        updateIdx=np.where(newBees.objs>employedBees[globalIdx].objs)[0]
        limitCount[employedType[globalIdx][updateIdx]]+=1
        
        return pop, beeType, limitCount
        
    def updateEmployedBees(self, pop: Population, beeType: np.ndarray,  limitCount: np.ndarray):
        
        n, d=pop.size()
        employedBeesType=np.where(beeType==1)[0]
        nEmployBees=np.sum(beeType==1)
        idx=np.arange(len(pop))
        while True:
            randIdx=np.random.permutation(idx)
            if(np.all(randIdx[employedBeesType]!=idx[employedBeesType])):
                break
            
        rnd=np.random.random((nEmployBees, d))*2-1
        
        newBees=pop[employedBeesType]+(pop[randIdx[employedBeesType]]-pop[employedBeesType])*rnd
        newBees.clip(self.problem.lb, self.problem.ub)
        
        self.evaluate(newBees)
        
        countIdx=np.where(newBees.objs>=pop[employedBeesType].objs)[0]
        limitCount[employedBeesType[countIdx]]+=1
        
        updateIdx=np.where(newBees.objs<pop[employedBeesType].objs)[0]
        pop.replace(employedBeesType[updateIdx], newBees[updateIdx])
        
       
        
        return pop, limitCount
    
    def setEmployedBees(self, beeType: np.ndarray, pop: Population, employedRate: float):
        
        nEmployBees=np.sum(beeType==1)
        
        if nEmployBees==0:
            idx=pop.argsort()
            beeType[idx[:int(len(pop)*employedRate)]]=1
        
        return pop, beeType