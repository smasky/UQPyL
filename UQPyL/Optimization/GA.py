import numpy as np
import warnings
import math
from typing import Callable
class GA():
    type="EA" #Evolutionary Algorithm
    proC=None
    disC=None
    proM=None
    disM=None
    tolerate=1e-6
    def __init__(self, dim: int, ub: np.ndarray, lb: np.ndarray, n_samples: int,
                 proC: float=1, disC: float=20, proM: float=1, disM: float=20,
                 tolerate_times: int=1000):
        self.__check__(dim,ub,lb)
        self.dim=dim;self.ub=ub.reshape(1,-1);self.lb=lb.reshape(1,-1)
        self.proC=proC;self.disC=disC
        self.proM=proM;self.disM=disM
        self.tolerate=1e-6; self.tolerate_times=tolerate_times
        self.n_samples=n_samples
        self.iterTimes=400
    def _tournamentSelection(self,decs: np.ndarray, objs: np.ndarray, K: int=2):
        '''
            K-tournament selection
        '''
        rankIndex=np.argsort(objs,axis=0)
        rank=np.argsort(rankIndex,axis=0)
        
        tourSelection=np.random.randint(0,high=objs.shape[0],size=(objs.shape[0],K))
        winner=np.min(rank[tourSelection,:].ravel().reshape(objs.shape[0],2),axis=1)
        winIndex=rankIndex[winner]
        
        return decs[winIndex.ravel(),:]
        
    def _operationGA(self,decs: np.ndarray):
        '''
            GA Operation
        '''
        n_samples=decs.shape[0]
        Parent1=decs[:math.floor(n_samples/2),:]
        Parent2=decs[math.floor(n_samples/2):math.floor(n_samples/2)*2,:]
        
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
        
        Lower=np.repeat(self.lb,2*N,axis=0)
        Upper=np.repeat(self.ub,2*N,axis=0)
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
    
    def run(self,func: Callable):
        
        best_objs=np.inf
        best_decs=None
        time=1
        iter=0
        
        decs=np.random.random((self.n_samples,self.dim))*(self.ub-self.lb)+self.lb
        objs=func(decs)
        
        while iter<self.iterTimes:
            
            matingPool=self._tournamentSelection(decs,objs,2)
            matingDecs=self._operationGA(matingPool)
            matingObjs=func(matingDecs)
            
            tempObjs=np.vstack((objs,matingObjs))
            tempDecs=np.vstack((decs,matingDecs))
            rank=np.argsort(tempObjs,axis=0)
            decs=tempDecs[rank[:self.n_samples,0],:]
            objs=tempObjs[rank[:self.n_samples,0],:]
            
            if(abs(best_objs-np.min(objs))>self.tolerate):
                best_objs=np.min(objs)
                best_decs=decs[np.argmin(objs,axis=0),:]
                time=0
            else:
                time+=1
            
            if(time>self.tolerate_times):
                break
            
            iter+=1
            
            return best_decs,best_objs
            
    def __check__(self,dim: int, ub: np.ndarray, lb: np.ndarray):
        if(ub.size==lb.size and dim==ub.size):
            pass
        else:
            raise ValueError("The dimensions should be consistent among dim, ub and lb")

        