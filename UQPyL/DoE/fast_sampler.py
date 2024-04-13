import numpy as np

from .sampler_ABC import Sampler

class FAST_Sampler(Sampler):
    def __init__(self, M):
        
        super().__init__()
        self.M=M
        
    def _generate(self, nt: int, nx: int) -> np.ndarray:
        
        if nt<=4*self.M**2:
            raise ValueError("the number of sample must be greater than 4*M**2!")
        
        w=np.zeros(nx)
        w[0]=np.floor((nt-1)/(2*self.M))
        max_wi=np.floor(w[0]/(2*self.M)) #Saltelli 
        
        if max_wi>=nx-1:
            w[1:]=np.floor(np.linspace(1,max_wi, nx-1))
        else:
            w[1:]=np.arange(nt-1)%max_wi+1
        
        s=(2*np.pi/nt)*np.arange(nt)
        
        X_sa=np.zeros((nt*nx, nx))
        w_tmp=np.zeros(nx)
        
        for i in range(nx):
            w_tmp[i]=w[0]
            idx=list(range(i))+list(range(i+1,nx))
            w_tmp[idx]=w[1:]
            idx=range(i*nt, (i+1)*nt)   
            phi=2*np.pi*np.random.rand()    
            sin_result=np.sin(w_tmp[:,None]*s+phi)
            arsin_result=(1/np.pi)*np.arcsin(sin_result) #saltelli formula
            X_sa[idx, :]=0.5+arsin_result.transpose()
        
        return X_sa