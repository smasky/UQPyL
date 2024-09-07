# Artificial Bee Colony Algorithm <Single>
import numpy as np

from ..algorithm import Algorithm, Population
from ...utility import Verbose

class ABC(Algorithm):
    """
    Artificial Bee Colony Algorithm (ABC) <Single>
    """
    
    def __init__(self, employedRate: float=0.3,  limit: int=50,
                 nInit: int=50, nPop: int=50, 
                 maxFEs: int = 50000, 
                 maxIterTimes: int = 1000, 
                 maxTolerateTimes=None, tolerate=1e-6, 
                 verbose=True, verboseFreq=10, logFlag=True):
                
        super().__init__(maxFEs, maxIterTimes, maxTolerateTimes, tolerate, verbose, verboseFreq, logFlag)
        
        self.setParameters('employedRate', employedRate)
        self.setParameters('limit', limit)
        self.setParameters('nInit', nInit)
        self.setParameters('nPop', nPop)
    
    @Verbose.decoratorRun
    def run(self, problem, xInit=None, yInit=None):
        #Parameter Setting
        employedRate, limit = self.getParaValue('percent', 'limit')
        nInit, nPop = self.getParaValue('nInit', 'nPop')
        #Problem
        self.setProblem(problem)
        #Population Generation
        if xInit is not None:
            pop = Population(xInit, yInit) if yInit is not None else Population(xInit)
            if yInit is None:
                self.evaluate(pop)
        else:
            pop = self.initialize(nInit)
            
        pop=pop.getTop(nPop)
        beeType=np.zeros(nPop)
        limitCount=np.zeros(nPop)
        
        while self.checkTermination():
            
            pop, beeType=self.setEmployedBees(beeType, pop, employedRate)
            
            pop, limitCount=self.updateEmployedBees(pop, beeType, limitCount)
            
            pop, beeType, limitCount=self.updateUnemployedBees(pop, beeType, limitCount)
            
            pop, beeType, limitCount=self.updateOnlookerBees(pop, beeType, limitCount, employedRate)
            
            beeType=self.checkLimitTimes(beeType, limitCount, limit)
            
    def checkLimitTimes(self, beeType:np.ndarray, limitCount: np.ndarray, limit: int):
        onlookerBees=np.where(limitCount>limit)[0]
        beeType[onlookerBees]=2
        
        return beeType        
    
    def updateOnlookerBees(self, pop: Population, beeType: np.ndarray, limitCount: np.ndarray, employedRate: float):
        maxNEmployed=int(len(pop)*employedRate)
        onlookerIdx=np.where(beeType==2)[0]
        onlookerBees=pop[onlookerIdx]
        n, d=onlookerBees
        
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
        
        employedBees=pop[beeType==1]
        unemployedBees=pop[beeType==0]
        
        offObjs=employedBees.objs-np.min(employedBees.objs)+1
        weights=1/offObjs
        p=weights/np.sum(weights)
        
        globalIdx=np.random.choice(len(employedBees), len(unemployedBees), p=p)

        idx=np.arange(len(pop))
        while True:
            randIdx=np.random.permutation(idx)
            if(np.all(randIdx[beeType==0]!=idx[beeType==1][globalIdx])):
                break
        
        rnd=np.random.random((len(unemployedBees), d))*2-1
        newBees=employedBees[globalIdx]+(employedBees[globalIdx]-pop[randIdx])*rnd
        newBees.clip(self.problem.lb, self.problem.ub)
        
        self.evaluate(newBees)
        
        pop.replace(beeType==0, newBees)
        
        replaceIdx=np.where(newBees.objs<employedBees[globalIdx].objs)[0]
        limitCount[beeType==0][replaceIdx]=0
        beeType[beeType==0][replaceIdx]=1
        limitCount[beeType==1][replaceIdx]=0
        beeType[beeType==1][replaceIdx]=0
        
        updateIdx=np.where(newBees.objs>employedBees[globalIdx].objs)[0]
        limitCount[beeType==1][updateIdx]+=1
        
        return pop, beeType, limitCount
        
    def updateEmployedBees(self, pop: Population, beeType: np.ndarray,  limitCount: np.ndarray):
        
        n, d=pop.size()
        employedBeesType=beeType==1
        nEmployBees=np.sum(employedBeesType)
        idx=np.arange(len(pop))
        while True:
            randIdx=np.random.permutation(idx)
            if(np.all(randIdx[employedBeesType]!=idx[employedBeesType])):
                break
            
        rnd=np.random.random((nEmployBees, d))*2-1
        
        newBees=pop[employedBeesType]+(pop[randIdx[employedBeesType]]-pop[employedBeesType])*rnd
        newBees=newBees.clip(self.problem.lb, self.problem.ub)
        
        self.evaluate(newBees)
        
        updateIdx=np.where(newBees.objs<pop[employedBeesType].objs)[0]
        pop.replace(employedBeesType, newBees[updateIdx])
        
        countIdx=np.where(newBees.objs>=pop[employedBeesType].objs)[0]
        limitCount[employedBeesType][countIdx]+=1
        
        return pop, limitCount
    
    def setEmployedBees(self, beeType: np.ndarray, pop: Population, employedRate: float):
        
        nEmployBees=np.sum(beeType==1)
        
        if nEmployBees==0:
            pop=pop[pop.argsort()]
            beeType[:int(len(pop)*employedRate)]=1
        
        return pop, beeType