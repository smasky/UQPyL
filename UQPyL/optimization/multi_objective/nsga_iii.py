# Non-dominated Sorting Genetic Algorithm III (NSGA-III) <Multi>
import numpy as np

from ..algorithmABC import Algorithm
from ..population import Population
from ...utility import Verbose
from ..utility_functions import tournamentSelection, operationGA, uniformPoint, NDSort, crowdingDistance

class NSGAIII(Algorithm):
    '''
    Non-dominated Sorting Genetic Algorithm III <Multi>
    '''
    
    name = "NSGAIII"
    type = "MOEA"
    
    def __init__(self, proC: float=1.0, disC: float=20.0, proM: float=1.0, disM: float=20.0,
                 nInit: int=50, nPop: int=50,
                 maxFEs=50000, maxIterTimes=1000, 
                 maxTolerateTimes=None, tolerate=1e-6, 
                 verbose=True, verboseFreq=10, 
                 logFlag=True, saveFlag=False):
        
        super().__init__(maxFEs, maxIterTimes, maxTolerateTimes, tolerate, verbose, verboseFreq, logFlag, saveFlag)
        
        self.setParameters('proC', proC)
        self.setParameters('disC', disC)
        self.setParameters('proM', proM)
        self.setParameters('disM', disM)
        self.setParameters('nInit', nInit)
        self.setParameters('nPop', nPop)
        
        #-------------------------Public Functions------------------------#
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem, xInit=None, yInit=None):
        
        #Parameter Setting
        proC, disC, proM, disM=self.getParaValue('proC', 'disC', 'proM', 'disM')
        nInit, nPop=self.getParaValue('nInit', 'nPop')
        
        #Problem
        self.setProblem(problem)
        Z, N=uniformPoint(nPop, problem.nOutput)
        nInit=N; nPop=N
        #Termination Condition Setting
        self.FEs=0; self.iters=0
        #Population Generation
        if xInit is not None:
            pop = Population(xInit, yInit) if yInit is not None else Population(xInit)
            if yInit is None:
                self.evaluate(pop)
        else:
            pop = self.initialize(nInit)
            
        pop=pop.getTop(nPop)
        
        # Zmin=np.min(pop.objs, axis=0).reshape(1,-1)
        
        while self.checkTermination():
            
            frontNo, _ = NDSort(pop)
            crowdDis = crowdingDistance(pop, frontNo)
            # _, frontNo, CrowdDis=self.environmentalSelection(pop, nPop)
            selectIdx=tournamentSelection(2, nPop, frontNo, -crowdDis)
            
            offspring=operationGA(pop[selectIdx], self.problem.ub, self.problem.lb, proC, disC, proM, disM)
            
            self.evaluate(offspring)
            
            pop.merge(offspring)
            
            Zmin=np.min(pop.objs, axis=0).reshape(1,-1)
            
            pop=self.environmentSelection(pop, Z, Zmin)
            
            self.record(pop)
        
        return self.result
    
    def environmentSelection(self, pop, Z, Zmin):
        
        N=Z.shape[0]
        
        frontNo, maxFNo=NDSort(pop, N)
        
        next=frontNo < maxFNo
        
        last=np.where(frontNo==maxFNo)[0]
        
        popObjs1=pop.objs[next]
        popObjs2=pop.objs[last]
        
        choose=self.lastSelection(popObjs1, popObjs2, N-popObjs1.shape[0], Z, Zmin)
        
        next[last[choose]]=True
        
        offSpring=pop[next]
        
        return offSpring
        
    def lastSelection(self, PopObj1, PopObj2, K, Z, Zmin):
        from scipy.spatial.distance import cdist
        PopObj = np.vstack((PopObj1, PopObj2)) - Zmin
        N, M = PopObj.shape
        N1 = PopObj1.shape[0]
        N2 = PopObj2.shape[0]
        NZ = Z.shape[0]

        # Normalization
        # Detect the extreme points
        Extreme = np.zeros(M, dtype=int)
        w = np.zeros((M,M))+ 1e-6 + np.eye(M)
        for i in range(M):
            Extreme[i] = np.argmin(np.max(PopObj / w[i], axis=1))

        # Calculate the intercepts of the hyperplane constructed by the extreme points
        try:
            Hyperplane = np.linalg.solve(PopObj[Extreme, :], np.ones(M))
        except np.linalg.LinAlgError:
            Hyperplane = np.ones(M)
        a = 1 / Hyperplane
        if np.any(np.isnan(a)):
            a = np.max(PopObj, axis=0)
        
        # Normalize PopObj
        PopObj = PopObj / a

        # Associate each solution with one reference point
        # Calculate the cosine similarity
        Cosine = 1 - cdist(PopObj, Z, 'cosine')
        Distance = np.sqrt(np.sum(PopObj**2, axis=1)).reshape(-1, 1) * np.sqrt(1 - Cosine**2)
        
        # Find the nearest reference point for each solution
        d = np.min(Distance, axis=1)
        pi = np.argmin(Distance, axis=1)

        # Calculate the number of associated solutions for each reference point
        rho = np.histogram(pi[:N1], bins=np.arange(NZ + 1))[0]

        # Environmental selection
        Choose = np.zeros(N2, dtype=bool)
        Zchoose = np.ones(NZ, dtype=bool)

        # Select K solutions one by one
        while np.sum(Choose) < K:
            # Find the least crowded reference point
            Temp = np.where(Zchoose)[0]
            if Temp.size == 0:
                break
            Jmin = Temp[np.where(rho[Temp] == np.min(rho[Temp]))[0]]
            j = Jmin[np.random.randint(len(Jmin))]

            # Find unselected solutions associated with this reference point
            I = np.where((~Choose) & (pi[N1:] == j))[0]

            if I.size > 0:
                if rho[j] == 0:
                    s = np.argmin(d[N1 + I])
                else:
                    s = np.random.choice(I.size)
                Choose[I[s]] = True
                rho[j] += 1
            else:
                Zchoose[j] = False

        return Choose
        
        
        
        
