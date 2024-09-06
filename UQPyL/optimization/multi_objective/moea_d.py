#multiobjective evolutionary algorithm based on decomposition
import numpy as np
import math
from typing import Optional, Literal
from scipy.spatial import distance

from ..DoE import LHS
from ..problems import Problem
from .utility_functions._uniformPoint import _NBI 

class MOEA_D():
    '''
        Multi_objective Evolutionary Algorithm based on Decomposition <Multi-objective>
        -------------------------------------------
        Parameters:
            problem: Problem
                the problem you want to solve, including the following attributes:
                    n_input: int
                        the input number of the problem
                    ub: 1d-np.ndarray or float
                        the upper bound of the problem
                    lb: 1d-np.ndarray or float
                        the lower bound of the problem
                    evaluate: Callable
                        the function to evaluate the input
            n_samples: int, default=50
                the number of initial samples
            x_init: 2d-np.ndarray, default=None
                the initial input samples
            y_init: 2d-np.ndarray, default=None
                the initial output samples
            aggregation_type: Literal['PBI', 'TCH', 'TCH_N', 'TCH_M'], default='PBI'
                the type of aggregation function
            maxFEs: int, default=50000
                the maximum number of function evaluations
            maxIters: int, default=1000
                the maximum number of iterations
    '''
    def __init__(self, problem: Problem, n_samples: int=50,
                       x_init: Optional[np.ndarray]=None, y_init: Optional[np.ndarray]=None,
                       aggregation_type: Literal['PBI', 'TCH', 'TCH_N', 'TCH_M']= 'PBI',
                       maxFEs: int=50000, maxIters: int=1000):
            #problem setting
            self.evaluate=problem.evaluate
            self.n_input=problem.n_input
            self.lb=problem.lb; self.ub=problem.ub
            self.n_output=problem.n_output
            self.problem=problem
            
            #inital setting
            self.n_samples=n_samples
            self.x_init=x_init
            self.y_init=y_init
            self.aggregation_type=aggregation_type
            
            #termination setting
            self.maxFEs=maxFEs
            self.maxIters=maxIters
            
    #-------------------Public Functions-----------------------#
    def run(self):
            
        maxFEs=self.maxFEs
        maxIters=self.maxIters
        
        n_input=self.n_input
        lb=self.lb
        ub=self.ub
        
        lhs=LHS('classic', problem=self.problem)
        if self.x_init is None:
            self.x_init=lhs(self.n_samples, n_input)
        if self.y_init is None:
            self.y_init=self.evaluate(self.x_init)
        
        XPop=self.x_init
        YPop=self.y_init
        
        FE=YPop.shape[0]
        Iter=0
        
        T = math.ceil(self.n_samples / 10)
        W, N=_NBI(self.n_samples, self.n_output)
        
        B=distance.cdist(W, W, metric='euclidean')
        B=np.argsort(B, axis=1)
        B=B[:,0:T]
        
        Z=np.min(YPop, axis=0).reshape(1,-1)
        
        history_pareto_x=[]
        history_pareto_y=[]
        
        while FE<maxFEs and Iter<maxIters:
            for i in range(N):
                
                P = B[i, np.random.permutation(B.shape[1])]

                XOffspring=self._operationGA(XPop[P[0:2],:])
                YOffspring=self.evaluate(XOffspring)
                
                Z=np.min(np.vstack((Z, YOffspring)), axis=0).reshape(1, -1)
                
                #PBI
                if(self.aggregation_type=='PBI'):
                    normW = np.sqrt(np.sum(W[P, :]**2, axis=1))
                    normP = np.sqrt(np.sum((YPop[P] - np.tile(Z, (T, 1)))**2, axis=1))
                    normO = np.sqrt(np.sum((YOffspring - Z)**2, axis=1))
                    CosineP = np.sum((YPop[P] - np.tile(Z, (T, 1))) * W[P, :], axis=1) / normW / normP
                    CosineO = np.sum(np.tile(YOffspring - Z, (T, 1)) * W[P, :], axis=1) / normW / normO
                    g_old = normP * CosineP + 5 * normP * np.sqrt(1 - CosineP**2)
                    g_new = normO * CosineO + 5 * normO * np.sqrt(1 - CosineO**2)
                elif(self.aggregation_type=='TCH'):
                    g_old = np.max(np.abs(YPop[P] - np.tile(Z, (T, 1))) * W[P, :], axis=1)
                    g_new = np.max(np.tile(np.abs(YOffspring - Z), (T, 1)) * W[P, :], axis=1)
                elif(self.aggregation_type=='TCH_N'):
                    Zmax = np.max(YPop, axis=0)
                    g_old = np.max(np.abs(YPop[P] - np.tile(Z, (T, 1))) / np.tile(Zmax - Z, (T, 1)) * W[P, :], axis=1)
                    g_new = np.max(np.tile(np.abs(YOffspring - Z) / (Zmax - Z), (T, 1)) * W[P, :], axis=1)
                elif(self.aggregation_type=='TCH_M'):
                    g_old = np.max(np.abs(YPop[P] - np.tile(Z, (T, 1))) / W[P, :], axis=1)
                    g_new = np.max(np.tile(np.abs(YOffspring - Z), (T, 1)) / W[P, :], axis=1)
                    
                XPop[P[g_old >= g_new]] = XOffspring
                YPop[P[g_old >= g_new]] = YOffspring
                
            FrontNo, MaxFNo=self._NDSort(YPop, self.n_samples)
            idx=np.where(FrontNo==1.0)
            history_pareto_x.append(XPop[idx])
            history_pareto_y.append(YPop[idx])
                
            Iter+=1
            FE+=2*N
        
        FrontNo, MaxFNo=self._NDSort(YPop, self.n_samples)
        crowDis=self._CrowdingDistance(YPop, FrontNo)
        idx=np.where(FrontNo==1.0)
        
        Result={}
        Result['pareto_x']=XPop[idx]
        Result['pareto_y']=YPop[idx]
        Result['crowdDis']=crowDis
        Result['history_pareto_x']=history_pareto_x
        Result['history_pareto_y']=history_pareto_y
        
        return XPop, YPop
                
    def _operationGA(self,decs: np.ndarray):
        '''
            GA Operation
        '''
        proC=1; disC=20; proM=1; disM=20
        n_samples=decs.shape[0]
        parent1=decs[:math.floor(n_samples/2),:]
        parent2=decs[math.floor(n_samples/2):math.floor(n_samples/2)*2,:]
        
        N, D = parent1.shape
        beta = np.zeros((N, D))
        mu = np.random.rand(N, D)
        beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / (disC + 1))
        beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / (disC + 1))
        beta = beta * (-1) ** np.random.randint(2, size=(N, D))
        beta[np.random.rand(N, D) < 0.5] = 1
        beta[(np.random.rand(N, 1) > proC).repeat(D, axis=1)] = 1
        Offspring = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2

        # Polynomial mutation
        Lower = np.tile(self.lb, (N, 1))
        Upper = np.tile(self.ub, (N, 1))
        Site = np.random.rand(N, D) < proM / D
        mu = np.random.rand(N, D)
        temp = Site & (mu <= 0.5)
        Offspring = np.minimum(np.maximum(Offspring, Lower), Upper)
        Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * \
                            (1 - (Offspring[temp] - Lower[temp]) / (Upper[temp] - Lower[temp])) ** (disM + 1)) ** (1 / (disM + 1)) - 1)
        temp = Site & (mu > 0.5)
        Offspring[temp] = Offspring[temp] + (Upper[temp] - Lower[temp]) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * \
                            (1 - (Upper[temp] - Offspring[temp]) / (Upper[temp] - Lower[temp])) ** (disM + 1)) ** (1 / (disM + 1)))
        
        return Offspring
    
    def _CrowdingDistance(self, PopObj, FrontNo):
        N, M = PopObj.shape

        # 如果未提供FrontNo，默认所有解决方案属于同一前沿
        if FrontNo is None:
            FrontNo = np.ones(N)
        
        CrowdDis = np.zeros(N)
        Fronts = np.setdiff1d(np.unique(FrontNo), np.inf)
        
        for f in Fronts:
            Front = np.where(FrontNo == f)[0]
            Fmax = np.max(PopObj[Front, :], axis=0)
            Fmin = np.min(PopObj[Front, :], axis=0)
            
            for i in range(M):
                # 对第i个目标排序，获取排序后的索引
                Rank = np.argsort(PopObj[Front, i])
                CrowdDis[Front[Rank[0]]] = np.inf
                CrowdDis[Front[Rank[-1]]] = np.inf
                
                for j in range(1, len(Front) - 1):
                    CrowdDis[Front[Rank[j]]] += (PopObj[Front[Rank[j+1]], i] - PopObj[Front[Rank[j-1]], i]) / (Fmax[i] - Fmin[i])
                        
        return CrowdDis
    
    def _NDSort(self, YPop, NSort):
        '''
            Non-dominated Sorting
        '''
        
        PopObj, indices = np.unique(YPop, axis=0, return_inverse=True)
       
        Table = np.bincount(indices)
        N, M = PopObj.shape
        FrontNo = np.inf * np.ones(N)
        MaxFNo = 0

        while np.sum(Table[FrontNo < np.inf]) < min(NSort, len(indices)):
            MaxFNo += 1
            for i in range(N):
                if FrontNo[i] == np.inf:
                    Dominated = False
                    for j in range(i-1, -1, -1):
                        if FrontNo[j] == MaxFNo:
                            m = 1
                            while m < M and PopObj[i, m] >= PopObj[j, m]:
                                m += 1
                            Dominated = m == M
                            if Dominated or M == 2:
                                break
                    if not Dominated:
                        FrontNo[i] = MaxFNo

        FrontNo = FrontNo[indices]

        return FrontNo, MaxFNo       
        
            
        