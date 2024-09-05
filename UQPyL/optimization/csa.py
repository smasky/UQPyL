# Cooperation search algorithm <Single>
import numpy as np
import math
import copy

from .optimizer import Optimizer, Population, Verbose
class CSA(Optimizer):
    """
    Cooperative Search Algorithm (CSA) <Single>
    -------------------------------------------------
    Attributes:
        problems : Problem
            the problem you want to solve, including the following attributes:
                n_input: int
                    the input number of the problem
                ub: 1d-np.ndarray or float
                    the upper bound of the problem
                lb: 1d-np.ndarray or float
                    the lower bound of the problem
                evaluate: Callable
                    the function to evaluate the input
    References:
        [1]	Z. Feng, W. Niu, and S. Liu (2021), Cooperation search algorithm: A novel metaheuristic evolutionary intelligence algorithm for numerical optimization and engineering optimization problems, Appl. Soft. Comput., vol. 98, p. 106734, Jan.  doi: 10.1016/j.asoc.2020.106734.
    """
    name="Cooperative Search Algorithm"
    type="EA" #Evolutionary Algorithm
    target="Single"
    def __init__(self, alpha: float = 0.10, beta: float = 0.15, M: int = 3,
                 nInit: int = 50, nPop: int = 50,
                 maxIterTimes: int=1000,
                 maxFEs: int=50000,
                 maxTolerateTimes: int=1000, tolerate: float=1e-6, 
                 verbose: bool=True, verboseFreq: int=100, logFlag: bool=False):
        
        super().__init__(maxFEs=maxFEs, maxIterTimes=maxIterTimes, 
                         maxTolerateTimes=maxTolerateTimes, tolerate=tolerate, 
                         verbose=verbose, verboseFreq=verboseFreq, logFlag=logFlag)
        
        #user-define setting
        self.alpha=alpha
        self.beta=beta
        self.M=M
        self.nInit=nInit; self.nPop=nPop
        #record
        self.setting["alpha"]=self.alpha
        self.setting["beta"]=self.beta
        self.setting["M"]=self.M
        self.setting["nInit"]=nInit
        self.setting["nPop"]=nPop
    
    #------------------Public Function------------------#
    @Verbose.decoratorRun
    def run(self, problem, xInit=None, yInit=None):
        
        self.problem=problem
        self.FEs=0; self.iters=0; self.tolerateTimes=0
        
        if xInit is not None:
            if yInit is not None:
                pop=Population(xInit, yInit)
            else:
                pop=Population(xInit)
                self.evaluate(pop)
        else:
            pop=self.initialize()
        
        pop=pop.getTop(self.nPop)
        
        #Initial directors and supervisors
        pBest=copy.deepcopy(pop)
        
        gBest=pBest[pBest.argsort()[:self.M]]
    
        while self.checkTermination():
            
            #Team communication operator
            uPop=self.teamCommunicationOperator(pop, pBest, gBest)
            
            #Reflective learning operator 
            vPop=self.reflectiveLearningOperator(uPop)
            
            #Internal competition operator
            self.evaluate(uPop)
            self.evaluate(vPop)
            
            newPop=Population(decs=np.where(uPop.objs<vPop.objs, uPop.decs, vPop.decs), objs=np.minimum(uPop.objs, vPop.objs))
            self.record(newPop)

            #Update person best and global best
            tmp=newPop[newPop.argsort()[:self.M]]
            pBest=Population(decs=np.where(newPop.objs<pBest.objs, newPop.decs, pBest.decs), objs=np.minimum(newPop.objs, pBest.objs) )
            # gBest=pBest[pBest.argsort()[:self.M]]
            gBest.add(tmp)
            gBest=gBest[gBest.argsort()[:self.M]]
            pop=copy.deepcopy(newPop)
            
        return self.result
    
    def reflectiveLearningOperator(self, pop):
        
        n, d=pop.size()
        c=(self.problem.ub+self.problem.lb)/2
        c_n=np.repeat(c, n, axis=0)
        lb_n=np.repeat(self.problem.lb, n, axis=0)
        ub_n=np.repeat(self.problem.ub, n, axis=0)
        fai_1=self.problem.ub+self.problem.lb-pop.decs
        
        gailv=np.abs(pop.decs-c)/(self.problem.ub-self.problem.lb)
        #calculate r
        t1=np.random.random((n, d))*np.abs(c-fai_1)+np.where(c_n>fai_1, fai_1, c_n)
        t2=np.random.random((n, d))*np.abs(fai_1-self.problem.lb)+np.where(fai_1>lb_n, lb_n, fai_1)
        seed=np.random.random((n,d))
        r=np.where(gailv<seed, t1, t2)
        
        #calculate p
        t3=np.random.random((n, d))*np.abs(fai_1-c)+np.where(c_n>fai_1, fai_1, c_n)
        t4=np.random.random((n, d))*np.abs(self.problem.ub-fai_1)+np.where(fai_1>ub_n, ub_n, fai_1)
        seed=np.random.random((n,d))
        p=np.where(gailv<seed, t3, t4)
        
        vPop=Population(decs=np.where(pop.decs>=c_n, r, p))
        vPop.clip(self.problem.lb, self.problem.ub)
        return vPop
    
    def teamCommunicationOperator(self, pop, pBest, gBest):
        
        n, d=pop.size()
        
        ind=np.random.randint(0, self.M, (n,d))
        
        A=np.log(1.0/np.random.random((n, d)))*(gBest.decs[ind, np.arange(d)]-pop.decs)
        
        B=self.alpha*np.random.random((n, d))*(np.mean(gBest.decs, axis=0)-pop.decs)
        
        C=self.beta*np.random.random((n, d))*(np.mean(pBest.decs, axis=0)-pop.decs)
        
        uPop=pop+A+B+C
        uPop.clip(self.problem.lb, self.problem.ub)
        
        return uPop

    # def tCO(self, pop, pBest, gBest):
    #     ave_pbest=np.mean(pBest.decs, axis=0)
    #     ave_gbest=np.mean(gBest.decs, axis=0)
    #     n, d=pop.size()
        
    #     gBestP=gBest.decs
    #     decs=pop.decs
    #     new_decs=np.zeros(shape=decs.shape)
    #     for i in range(n):
    #         for j in range(d):
    #             alpha=0.10
    #             beta=0.15
                
    #             A=np.log(1.0/self.Phi(0,1))*(gBestP[np.random.randint(0, self.M), j]-decs[i,j])

    #             B=alpha*self.Phi(0,1)*(ave_gbest[j]-decs[i,j])
                
    #             C=beta*self.Phi(0,1)*(ave_pbest[j]-decs[i,j])
                
    #             new_decs[i,j]=decs[i,j]+A+B+C
        
    #     new_decs.clip(self.problem.lb, self.problem.ub)
        
    #     return Population(decs=new_decs)
                
    def Phi(self, num1, num2):
        if num1<num2:
            o=num1+np.random.random(1)*abs(num1-num2)
        else:
            o=num2+np.random.random(1)*abs(num1-num2)
        return o