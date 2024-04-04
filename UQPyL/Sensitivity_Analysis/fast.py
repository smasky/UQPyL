# Fourier amplitude sensitivity test, FAST
"""
  References
    ----------
    1. Cukier, R.I., Fortuin, C.M., Shuler, K.E., Petschek, A.G.,
       Schaibly, J.H., 1973.
       Study of the sensitivity of coupled reaction systems to
       uncertainties in rate coefficients. I theory.
       Journal of Chemical Physics 59, 3873-3878.
       doi:10.1063/1.1680571
    2. Saltelli, A., S. Tarantola, and K. P.-S. Chan (1999).
       A Quantitative Model-Independent Method for Global Sensitivity Analysis
       of Model Output.
       Technometrics, 41(1):39-56,
       doi:10.1080/00401706.1999.10485594
    3. extend FAST
"""
import numpy as np
from ..Experiment_Design import LHS
lhs=LHS('center')
class FAST():
    def __init__(self, problem, NSample=100, surrogate=None, M=4, NSurrogate=50, XInit=None, YInit=None):
        self.evaluate=problem.evaluate
        self.lb=problem.lb; self.ub=problem.ub
        self.dim=problem.dim
        self.surrogate=surrogate
        
        self.M=M
        self.NSample=NSample
        
        if self.NSample<=4*self.M**2:
            raise ValueError("the number of sample must be greater than 4*M**2!")
            
        self.NSample=NSample
        
        if self.surrogate:
            if XInit is None:
                self.XInit=lhs(NSurrogate, self.dim)*(self.ub-self.lb)+self.lb
    
    def generate_samples(self):
        
        NS=self.NSample
        M=self.M
        D=self.dim
        
        #select omega
        w=np.zeros(D)
        w[0]=np.floor((NS-1)/(2*M))
        max_wi=np.floor(w[0]/(2*M)) #Saltelli 
        
        if max_wi>=D-1:
            w[1:]=np.floor(np.linspace(1,max_wi, D-1))
        else:
            w[1:]=np.arange(D-1)%max_wi+1
        
        s=(2*np.pi/NS)*np.arange(NS)
        
        X_seq=np.zeros((NS*D,D))
        w_tmp=np.zeros(D)
        
        for i in range(D):
            w_tmp[i]=w[0]
            idx=list(range(i))+list(range(i+1,D))
            w_tmp[idx]=w[1:]
            
            idx=range(i*NS, (i+1)*NS)
            #Random Phase-shift Salt
            phi=2*np.pi*np.random.rand()
            
            sin_result=np.sin(w_tmp[:,None]*s+phi)
            arsin_result=(1/np.pi)*np.arcsin(sin_result)
            X_seq[idx, :]=0.5+arsin_result.transpose()
        
        return (X_seq)*(self.ub-self.lb)+self.lb
    
    def analyze(self):
        
        Si=[]
        ST=[]
        
        NS=self.NSample
        M=self.M
        D=self.dim
        
        X_seq=self.generate_samples()
        w_0=np.floor((NS-1)/(2*M))
        
        if self.surrogate:
            self.YInit=self.evaluate(self.XInit)
            self.surrogate.fit(self.XInit, self.YInit)
            Y_seq=self.surrogate.predict(X_seq)
        else:
            Y_seq=self.evaluate(X_seq)
        
        for i in range(D):
            idx=np.arange(i*NS, (i+1)*NS)
            Y_sub=Y_seq[idx]
            #fft
            f=np.fft.fft(Y_sub)
            Sp = np.power(np.absolute(f[np.arange(1, np.ceil((NS-1)/2), dtype=np.int32)-1])/NS, 2) #TODO 1-(NS-1)/2

            V=2.0*np.sum(Sp)
            Di=2.0*np.sum(Sp[np.int32(np.arange(1, M+1, dtype=np.int32)*w_0-1)]) #pw<=(NS-1)/2 w_0=(NS-1)/M
            Dt=2.0*np.sum(Sp[np.arange(np.floor(w_0/2.0), dtype=np.int32)])
            
            Si.append(Di/V)
            ST.append(1.0-Dt/V)
        
        return Si, ST
        
        
        
        