# Non-dominated Sorting Genetic Algorithm II (NSGA-II)

import numpy as np
from ..Experiment_Design import LHS

lhs=LHS('center')

class NSGAII():
    def __init__(self, evaluator, NInput, NOutput, LB, UB, NInit, XInit=None, YInit=None,
                 proC=1, disC=20, proM=1, disM=20):
        self.evaluator=evaluator
        self.NInput=NInput
        self.NOutput=NOutput
        self.LB=LB
        self.UB=UB
        self.NInit=NInit
        self.XInit=XInit
        self.YInit=YInit
        # self.NPop=NPop
        # self.NGen=NGen
        ####GA setting
        self.proC=proC
        self.disC=disC
        self.proM=proM
        self.disM=disM
    
    def run(self, maxFE=100000):
        
        NInput=self.NInput
        NInit=self.NInit
        LB=self.LB
        UB=self.UB
            
        if self.XInit is None:
            self.XInit=(UB-LB)*lhs(self.NInit, NInput)+LB
        if self.YInit is None:
            self.YInit=self.evaluator(self.XInit)
        
        XPop=self.XInit
        YPop=self.YInit
        FE=YPop.shape[0]
        
        _,_, FrontNo, CrowdDis=self.EnvironmentalSelection(XPop, YPop, self.NInit)
        
        while FE<maxFE:
            
            SelectIndex=self.TournamentSelection(2, NInit, FrontNo, -CrowdDis)
            XOffSpring=self._operationGA(XPop[SelectIndex,:])
            YOffSpring=self.evaluator(XOffSpring)
            FE+=YOffSpring.shape[0]
            
            XPop=np.vstack((XPop,XOffSpring))
            YPop=np.vstack((YPop,YOffSpring))
            
            XPop, YPop, FrontNo, CrowdDis=self.EnvironmentalSelection(XPop, YPop, self.NInit)
        idx=np.where(FrontNo==1.0)
        return XPop[idx], YPop[idx], FrontNo[idx], CrowdDis[idx]
    
    def _operationGA(self,decs: np.ndarray):
        '''
            GA Operation
        '''
        n_samples=decs.shape[0]
        Parent1=decs[:np.floor(n_samples/2).astype(int),:]
        Parent2=decs[np.floor(n_samples/2).astype(int):np.floor(n_samples/2).astype(int)*2,:]
        
        N,D=Parent1.shape
        
        beta=np.zeros((N,D))
        mu=np.random.random((N,D))
        
        beta[mu<=0.5]=np.power(2*mu[mu<=0.5],1/(self.disC+1))
        beta[mu>0.5]=np.power(2-2*mu[mu>0.5],-1/(self.disC+1))
        beta=beta*np.power(-1,np.random.randint(0,high=2,size=(N,D)))
        beta[np.random.random((N,D))<0.5]=1
        beta[np.repeat(np.random.random((N,1))>self.proC,D,axis=1)]=1
        
        off1=(Parent1+Parent2)/2+beta*(Parent1-Parent2)/2
        off2=(Parent1+Parent2)/2-beta*(Parent1-Parent2)/2
        Offspring=np.vstack((off1,off2))
        
        Lower=np.repeat(self.LB,2*N,axis=0)
        Upper=np.repeat(self.UB,2*N,axis=0)
        Site=np.random.random((2*N,D))<self.proM/D
        mu=np.random.random((2*N,D)) 
        temp=np.zeros((2*N,D),dtype=np.bool_)
        temp[Site * mu<=0.5]=1
        Offspring=np.minimum(np.maximum(Offspring,Lower),Upper)
        
        t1=(1-2*mu[temp])*np.power(1-(Offspring[temp]-Lower[temp])/(Upper[temp]-Lower[temp]),self.disM+1)
        Offspring[temp]=Offspring[temp]+(Upper[temp]-Lower[temp])*(np.power(2*mu[temp]+t1,1/(self.disM+1))-1)
        
        temp=np.zeros((2*N,D),dtype=np.bool_);temp[Site * mu>0.5]=1
        t2=2*(mu[temp]-0.5)*np.power(1-(Upper[temp]-Offspring[temp])/(Upper[temp]-Lower[temp]),self.disM+1)
        
        Offspring[temp]=Offspring[temp]+(Upper[temp]-Lower[temp])*(1-np.power(2*(1-mu[temp])+t2,1/(self.disM+1)))
        
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
        ranks = np.lexsort((fitness2.ravel(), fitness1.ravel()))
        
        # Conduct the tournament
        selected_indices = []
        for _ in range(N):
            # Randomly pick K candidates
            candidates_indices = np.random.choice(range(len(fitness1)), K, replace=False)
            candidates_ranks = ranks[candidates_indices]
            
            # Select the candidate with the best (lowest) rank
            best_candidate_index = candidates_indices[np.argmin(candidates_ranks)]
            selected_indices.append(best_candidate_index)
        
        return np.array(selected_indices)

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
        
    def CrowdingDistance(self, YPop, FrontNo):
        N, M = YPop.shape
        CrowdDis = np.zeros(N)
        Fronts = np.unique(FrontNo[FrontNo != np.inf])
        
        for f in Fronts:
            Front = np.where(FrontNo == f)[0]
            Fmax = np.max(YPop[Front, :], axis=0)
            Fmin = np.min(YPop[Front, :], axis=0)
            D = np.zeros((len(Front), M))
            
            for i in range(M):
                order = np.argsort(YPop[Front, i])
                sortedFront = Front[order]
                D[:, i] = np.inf  # Initialize distances to infinity
                
                # Edge points always have infinite distance
                D[0, i] = np.inf
                D[-1, i] = np.inf
                
                # Compute crowding distance for each point
                for j in range(1, len(Front) - 1):
                    if (Fmax[i] - Fmin[i]) > 0:
                        D[j, i] = (YPop[sortedFront[j + 1], i] - YPop[sortedFront[j - 1], i]) / (Fmax[i] - Fmin[i])
            
            # Sum up the distances for all objectives
            CrowdDis[Front] = np.sum(D, axis=1)
    
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
        
        