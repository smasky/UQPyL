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
from .sa_ABC import SA
class FAST(SA):
    def __init__(self, problem, M=4, n_sample=100,
                 scale=None, lhs=None,
                 surrogate=None,  n_surrogate_sample=50, 
                 X_for_surrogate=None, Y_for_surrogate=None):
        
        super().__init__(problem, n_sample,
                         scale, lhs,
                         surrogate, n_surrogate_sample, X_for_surrogate, Y_for_surrogate
                         )

        if self.n_sample<=4*self.M**2:
            raise ValueError("the number of sample must be greater than 4*M**2!")
        
        self.M=M
    
    def generate_samples(self):
                
        #select omega
        w=np.zeros(self.n_input)
        w[0]=np.floor((self.n_sample-1)/(2*self.M))
        max_wi=np.floor(w[0]/(2*self.M)) #Saltelli 
        
        if max_wi>=self.n_input-1:
            w[1:]=np.floor(np.linspace(1,max_wi, self.n_input-1))
        else:
            w[1:]=np.arange(self.n_input-1)%max_wi+1
        
        s=(2*np.pi/self.n_sample)*np.arange(self.n_sample)
        
        X_sa=np.zeros((self.n_sample*self.n_input, self.n_input))
        w_tmp=np.zeros(self.n_input)
        
        for i in range(self.n_input):
            w_tmp[i]=w[0]
            idx=list(range(i))+list(range(i+1,self.n_input))
            w_tmp[idx]=w[1:]
            idx=range(i*self.n_sample, (i+1)*self.n_sample)   
            phi=2*np.pi*np.random.rand()    
            sin_result=np.sin(w_tmp[:,None]*s+phi)
            arsin_result=(1/np.pi)*np.arcsin(sin_result)
            X_sa[idx, :]=0.5+arsin_result.transpose()
        
        return (X_sa)*(self.ub-self.lb)+self.lb
    
    def analyze(self, X_sa=None, Y_sa=None):
        
        if X_sa==None:
            X_sa=self.generate_samples()
        
        X_sa, Y_sa=self.__check_and_scale_x_y__(X_sa, Y_sa)
        
        Si=[]; ST=[]
        
        w_0=np.floor((self.n_sample-1)/(2*self.M))
                
        for i in range(self.n_input):
            idx=np.arange(i*self.n_sample, (i+1)*self.n_sample)
            Y_sub=Y_sa[idx]
            #fft
            f=np.fft.fft(Y_sub)
            Sp = np.power(np.absolute(f[np.arange(1, np.ceil((self.n_sample-1)/2), dtype=np.int32)-1])/self.n_sample, 2) #TODO 1-(NS-1)/2

            V=2.0*np.sum(Sp)
            Di=2.0*np.sum(Sp[np.int32(np.arange(1, self.M+1, dtype=np.int32)*w_0-1)]) #pw<=(NS-1)/2 w_0=(NS-1)/M
            Dt=2.0*np.sum(Sp[np.arange(np.floor(w_0/2.0), dtype=np.int32)])
            
            Si.append(Di/V)
            ST.append(1.0-Dt/V)
            
        return Si, ST
    
    def summary(self):
        pass
        #TODO summary the result of FAST
        
        
        
        