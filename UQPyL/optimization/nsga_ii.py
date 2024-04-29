# Non-dominated Sorting Genetic Algorithm II (NSGA-II)
import numpy as np
import math
from typing import Optional

from ..DoE import LHS
from ..problems import Problem
lhs=LHS('classic')

class NSGAII():
    '''
    Non-dominated Sorting Genetic Algorithm II
    ------------------------------------------------
    Attributes:
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
            the number of samples for each generation
        x_init: np.ndarray, default=None
            the initial input
        y_init: np.ndarray, default=None
            the initial output
        proC: float, default=1
            the probability of crossover
        disC: float, default=20
            the distribution index of crossover
        proM: float, default=1
            the probability of mutation
        disM: float, default=20
            the distribution index of mutation
        maxFEs: int, default=50000
            the maximum number of function evaluations
        maxIters: int, default=1000
            the maximum number of iterations
    Methods:
        run: run the NSGA-II
        
    References:
        [1] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, 2002.
    '''
    def __init__(self, problem: Problem, n_samples: int=50, 
                 x_init: Optional[np.ndarray]=None, y_init: Optional[np.ndarray]=None,
                 proC=1, disC=20, proM=1, disM=20,
                 maxFEs: int=50000, maxIters: int=1000
                 ):
        #problem setting
        self.evaluate=problem.evaluate
        self.n_input=problem.n_input
        self.lb=problem.lb;self.ub=problem.ub
        self.n_output=problem.n_output
        
        #initial setting
        self.n_samples=n_samples
        self.x_init=x_init
        self.y_init=y_init
        
        #GA setting
        self.proC=proC
        self.disC=disC
        self.proM=proM
        self.disM=disM
        
        #termination setting
        self.maxFEs=maxFEs
        self.maxIters=maxIters
    #-------------------------Public Functions------------------------#
    def run(self):
        
        maxFEs=self.maxFEs
        maxIters=self.maxIters
        
        n_input=self.n_input
        lb=self.lb
        ub=self.ub
        
        if self.x_init is None:
            self.x_init=(ub-lb)*lhs(self.n_samples, n_input)+lb
        if self.y_init is None:
            self.y_init=self.evaluate(self.x_init)
        
        XPop=self.x_init
        YPop=self.y_init
        
        FE=YPop.shape[0]
        Iter=0
        
        _, _, FrontNo, CrowdDis=self.EnvironmentalSelection(XPop, YPop, self.n_samples)
        
        while FE<maxFEs and Iter<maxIters:
            
            SelectIndex=self.TournamentSelection(2, self.n_samples, FrontNo, -CrowdDis)
            XOffSpring=self._operationGA(XPop[SelectIndex,:])
            YOffSpring=self.evaluate(XOffSpring)
            
            XPop=np.vstack((XPop, XOffSpring))
            YPop=np.vstack((YPop, YOffSpring))
            
            XPop, YPop, FrontNo, CrowdDis=self.EnvironmentalSelection(XPop, YPop, self.n_samples)
            
            #Update the termination criteria
            FE+=YOffSpring.shape[0]
            Iter+=1
            
        idx=np.where(FrontNo==1.0)
        
        return XPop[idx], YPop[idx], FrontNo[idx], CrowdDis[idx]
    
    #-------------------------Private Functions--------------------------#
    def _operationGA(self,decs: np.ndarray):
        '''
            GA Operation
        '''
        n_samples=decs.shape[0]
        parent1=decs[:math.floor(n_samples/2),:]
        parent2=decs[math.floor(n_samples/2):math.floor(n_samples/2)*2,:]
        
        n, d = parent1.shape
        beta = np.zeros_like(parent1)
        mu = np.random.rand(n, d)

        beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1 / (self.disC + 1))
        beta[mu > 0.5] = np.power(2 - 2 * mu[mu > 0.5], -1 / (self.disC + 1))
        beta = beta * (-1) ** np.random.randint(0, 2, size=(n, d))
        beta[np.random.rand(n, d) < 0.5] = 1
        beta[np.repeat(np.random.rand(n, 1) > self.proC, d, axis=1)] = 1

        offspring = np.concatenate(( (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2,
                              (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2 ), axis=0)

        lower = np.repeat(self.lb, 2 * n, axis=0)
        upper = np.repeat(self.ub, 2 * n, axis=0)
        site = np.random.rand(2 * n, d) < self.proM / d
        mu = np.random.rand(2 * n, d)
        
        temp = site & (mu <= 0.5)
        offspring = np.clip(offspring, lower, upper)
        t1 = (1 - 2 * mu[temp]) * np.power(1 - (offspring[temp] - lower[temp]) / (upper[temp] - lower[temp]), self.disM + 1)
        offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * (np.power(2 * mu[temp] + t1, 1 / (self.disM + 1)) - 1)
        
        temp = site & (mu > 0.5)
        t2 = 2 * (mu[temp] - 0.5) * np.power(1 - (upper[temp] - offspring[temp]) / (upper[temp] - lower[temp]), self.disM + 1)
        offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * (1 - np.power(2 * (1 - mu[temp]) + t2, 1 / (self.disM + 1)))
        
        return offspring

        return Offspring
        
    
    def TournamentSelection(self, K, N, fitness1, fitness2):
        """
        Perform K-tournament selection based on two fitness criteria.

        Parameters:
        - K: The number of candidates to compete in each tournament.
        - N: The number of selections to make.
        - fitness1: The primary fitness values of the candidates.
        - fitness2: The secondary fitness values of the candidates.

        Returns:
        - indices of the selected N solutions.
        """
        # Ensure fitness values are numpy arrays
        fitness1 = np.array(fitness1).reshape(-1, 1)
        fitness2 = np.array(fitness2).reshape(-1, 1)
        
        # Combine the fitness values and sort candidates based on fitness1, then fitness2
        # fitness_combined = np.hstack([fitness1, fitness2])
        rankIndex = np.lexsort((fitness2.ravel(), fitness1.ravel())).reshape(-1,1)
        rank=np.argsort(rankIndex,axis=0).ravel()
        
        tourSelection=np.random.randint(0,high=fitness1.shape[0],size=(N,K))

        winner_indices_in_tournament = np.argmin(rank[tourSelection], axis=1).ravel()
        winners_original_order = tourSelection[np.arange(N), winner_indices_in_tournament]
        # winner=np.min(rank[tourSelection,:].ravel().reshape(fitness1.shape[0],2),axis=1)
        # winIndex=rankIndex[winner]
        
        return winners_original_order.ravel()
        # Conduct the tournament
        # selected_indices = []
        # for _ in range(N):
        #     # Randomly pick K candidates
        #     candidates_indices = np.random.choice(range(len(fitness1)), K, replace=False)
        #     candidates_ranks = ranks[candidates_indices]
            
        #     # Select the candidate with the best (lowest) rank
        #     best_candidate_index = candidates_indices[np.argmin(candidates_ranks)]
        #     selected_indices.append(best_candidate_index)
        
        # return np.array(selected_indices)

    def EnvironmentalSelection(self, XPop, YPop, N):
        # 非支配排序
        FrontNo, MaxFNo = self.NDSort(YPop, N)
        
        # 初始化下一代的选择
        Next = FrontNo < MaxFNo
        
        # 计算拥挤距离
        CrowdDis = self.CrowdingDistance(YPop, FrontNo)
        
        # 选择最后一前沿基于拥挤距离的个体
        Last = np.where(FrontNo == MaxFNo)[0]
        Rank = np.argsort(-CrowdDis[Last])
        NumSelected = N - np.sum(Next)
        Next[Last[Rank[:NumSelected]]] = True
        
        # 创建下一代
        NextXPop = np.copy(XPop[Next,:])
        NextYPop=np.copy(YPop[Next,:])
        NextFrontNo = np.copy(FrontNo[Next])
        NextCrowdDis = np.copy(CrowdDis[Next])
        
        return NextXPop, NextYPop, NextFrontNo, NextCrowdDis
        
    def CrowdingDistance(self, PopObj, FrontNo):
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
        

    def NDSort(self, YPop, NSort):
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
        
        