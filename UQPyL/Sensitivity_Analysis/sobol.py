#Sobol
import numpy as np
from scipy.stats import qmc

from .sa_ABC import SA

class Sobol(SA):
    def __init__(self, problem, n_sample=100, scramble=True, skip_value=0, cal_second_order=False, 
                 scale=None, lhs=None,
                 surrogate=None, n_surrogate_sample=50, 
                 X_for_surrogate=None, Y_for_surrogate=None):
        
        super().__init__(problem, n_sample,
                         scale, lhs,
                         surrogate, n_surrogate_sample, X_for_surrogate, Y_for_surrogate
                        )
        
        self.cal_second_order=cal_second_order
        self.scramble=scramble
        self.skip_value=skip_value
            
        if skip_value>0 and isinstance(skip_value, int):
            M=skip_value
            if not((M&(M-1))==0 and (M!=0 and M-1!=0)):
                raise ValueError("skip value must be a power of 2!")
            
            if self.n_sample>M:
                raise ValueError("skip value must be greater than NSample!")
        elif skip_value<0 or not isinstance(skip_value, int):
            raise ValueError("skip value must be a positive integer!")    
        
    def generate_samples(self):
        #Sobol sequence
        qrng=qmc.Sobol(d=2*self.n_input, scramble=self.scramble)
        
        if self.skip_value>0:
            qrng.fast_forward(self.skip_value)
            
        base_sequence=qrng.random(self.n_sample)
        
        if self.cal_second_order:
            saltelli_sequence=np.zeros(((2*self.n_input+2)*self.n_input, self.n_input))
        else:
            saltelli_sequence=np.zeros(((self.n_input+2)*self.n_sample, self.n_input))
        
        index=0
        for i in range(self.n_sample):
            saltelli_sequence[index, :]=base_sequence[i, :self.n_input]

            index+=1
            
            saltelli_sequence[index:index+self.n_input,:]=np.tile(base_sequence[i, self.self.n_input:], (self.n_input, 1))
            saltelli_sequence[index:index+self.n_input,:][np.diag_indices(self.n_input)]=base_sequence[i, :self.n_input]               
            index+=self.n_input
           
            if self.cal_second_order:
                saltelli_sequence[index:index+self.n_input,:]=np.tile(base_sequence[i, :self.dim], (self.n_input, 1))
                saltelli_sequence[index:index+self.n_input,:][np.diag_indices(self.n_input)]=base_sequence[i, self.n_input:] 
                index+=self.n_input
            
            saltelli_sequence[index,:]=base_sequence[i, self.n_input:self.n_input*2]
            index+=1
        
        saltelli_sequence=saltelli_sequence*(self.ub-self.lb)+self.lb
        
        return saltelli_sequence
    
    def analyze(self, X_sa=None, Y_sa=None):
        
        X_sa=self.generate_samples()
        X_sa, Y_sa=self.__check_and_scale_x_y__(X_sa, Y_sa)
        
        cal_second_order=self.cal_second_order
           
        Y_sa=(Y_sa-Y_sa.mean())/Y_sa.std()
        A, B, AB, BA=self.separate_output_values(Y_sa, self.n_input, self.n_sample, cal_second_order)
        
        if cal_second_order:
            S2=[]
        else:
            S2=None
        
        S1=[]; ST=[]
        for j in range(self.n_input):
            S1.append(self.first_order(A, AB[:, j:j+1], B))
            ST.append(self.total_order(A, AB[:, j:j+1], B))

        if cal_second_order:
            for j in range(self.n_input):
                for k in range(j+1, self.n_input):
                    S2.append(self.second_order(A, AB[:, j:j+1], AB[:, k:k+1], BA[:, j:j+1], B))
                    
        return S1, S2, ST
    
    def second_order(self, A, AB1, AB2, BA, B):
        Y=np.r_[A,B]
        
        Vjk=np.mean(BA*AB2- A*B, axis=0)/np.var(Y, axis=0)
        Sj=self.first_order(A, AB1, B)
        Sk=self.first_order(A, AB2, B)
        
        return Vjk-Sj-Sk
       
    def first_order(self, A, AB, B):
        Y=np.r_[A,B]
        
        return np.mean(B*(AB-A), axis=0)/np.var(Y, axis=0)
    
    def total_order(self, A, AB, B):
        Y=np.r_[A,B]
        
        return 0.5*np.mean((A-AB)**2, axis=0)/np.var(Y, axis=0)
            
        
    def separate_output_values(self, Y_sa, D, N, cal_second_order):
        AB=np.zeros((N,D))
        BA=np.zeros((N,D)) if cal_second_order else None
        
        step=2*D+2 if cal_second_order else D+2
        
        total=Y_sa.shape[0]
        
        A=Y_sa[0:total:step, :]
        B=Y_sa[(step-1):total:step, :]
        
        for j in range(D):
            AB[:, j]=Y_sa[(j+1):total:step, 0]
            if cal_second_order:
                BA[:, j]=Y_sa[(j+1+D):total:step, 0]
        
        return A, B, AB, BA
                
        
