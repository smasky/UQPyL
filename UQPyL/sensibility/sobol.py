#Sobol
import numpy as np
from scipy.stats import qmc
from typing import Optional, Tuple

from .sa_ABC import SA
from ..DoE import Sobol_Sequence, LHS, Sampler
from ..problems import ProblemABC as Problem
from ..surrogates import Surrogate
from ..utility import Scaler
class Sobol(SA):
    def __init__(self, problem: Problem, scramble: bool=True, skip_value: int=0, cal_second_order: bool=False, 
                 sampler: Sampler=Sobol_Sequence(), N_within_sampler: int=100,
                 scale: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None), surrogate: Surrogate=None,
                 if_sampling_consistent: bool=False, sampler_for_surrogate: Sampler=LHS('classic'), N_within_surrogate_sampler: int=100,
                 X_for_surrogate: Optional[np.ndarray]=None, Y_for_surrogate: Optional[np.ndarray]=None):
        
        super().__init__(problem, sampler, N_within_sampler,
                         scale, surrogate, if_sampling_consistent,
                         sampler_for_surrogate, N_within_surrogate_sampler,
                         X_for_surrogate, Y_for_surrogate)
        
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
        
#-------------------------Public Functions--------------------------------#
    def analyze(self, base_sa: np.ndarray=None, Y_sa: np.ndarray=None, verbose: bool=False):
        
        if base_sa and base_sa.shape[1]!=self.n_input*2:
            raise ValueError("The shape[1] of X_sa must twice greater than the input dimension of the problem!")
        else:
            base_sa=self.sampler.sample(self.N_within_sampler, self.n_input*2)\
                    *np.tile((self.ub-self.lb), (1,2))+np.tile(self.lb, (1,2))
        ##forward process
        X_sa=self.__check_and_scale_x__(base_sa[:, :self.n_input])
        X_sa=self._forward(base_sa)
        self.__prepare_surrogate__()
        
        Y_sa=self.evaluate(X_sa)
        
        self.Y=Y_sa
          
        Y_sa=(Y_sa-Y_sa.mean())/Y_sa.std()
        
        A, B, AB, BA=self.separate_output_values(Y_sa, self.n_input, self.N_within_sampler, self.cal_second_order)
        
        S2 = [] if self.cal_second_order else None
        S1=[]; ST=[]
        
        #main process
        for j in range(self.n_input):
            S1.append(self.first_order(A, AB[:, j:j+1], B))
            ST.append(self.total_order(A, AB[:, j:j+1], B))
        
        S2=np.full((self.n_input, self.n_input), np.nan)
        if self.cal_second_order:
            for j in range(self.n_input):
                for k in range(j+1, self.n_input):
                    S2[j,k]=(self.second_order(A, AB[:, j:j+1], AB[:, k:k+1], BA[:, j:j+1], B))
        
        self.Si={'S1':np.array(S1).ravel(), 'S2': S2, 'ST': np.array(ST).ravel()}
        
        if verbose:
            self.summary()
        
        return self.Si

    def summary(self):
        if self.Si is None:
            raise ValueError("The sensitivity analysis has not been performed yet!")
        
        print("Sobol Sensitivity Analysis")
        print("-------------------------------------------------")
        print("Input Dimension: %d" % self.n_input)
        print("-------------------------------------------------")
        print("First Order Sensitivity Indices: ")
        print("-------------------------------------------------")
        for label, value in zip(self.x_labels, self.Si['S1']):
            print(f"{label}: {value:.4f}")
        if self.cal_second_order:
            print("-------------------------------------------------")
            print("Second Order Sensitivity Indices: ")
            print("-------------------------------------------------")
            for i in range(self.n_input):
                for j in range(i+1, self.n_input):
                    print(f"{self.x_labels[i]}-{self.x_labels[j]}: {self.Si['S2'][i, j]:.4f}")
        print("-------------------------------------------------")
        print("Total Order Sensitivity Indices: ")
        print("-------------------------------------------------")
        for i in range(self.n_input):
           print(f"{self.x_labels[i]}: {self.Si['ST'][i]:.4f}")
        print("-------------------------------------------------")
        print("-------------------------------------------------")

#-------------------------Private Functions--------------------------------#
    def _forward(self, base_sequence):
        #Sobol sequence

        if self.cal_second_order:
            saltelli_sequence=np.zeros(((2*self.n_input+2)*self.N_within_sampler, self.n_input))
        else:
            saltelli_sequence=np.zeros(((self.n_input+2)*self.N_within_sampler, self.n_input))
        
        index=0
        for i in range(self.N_within_sampler):
            
            saltelli_sequence[index, :]=base_sequence[i, :self.n_input]

            index+=1
            
            saltelli_sequence[index:index+self.n_input,:]=np.tile(base_sequence[i, :self.n_input], (self.n_input, 1))
            saltelli_sequence[index:index+self.n_input,:][np.diag_indices(self.n_input)]=base_sequence[i, self.n_input:]               
            index+=self.n_input
           
            if self.cal_second_order:
                saltelli_sequence[index:index+self.n_input,:]=np.tile(base_sequence[i, self.n_input:], (self.n_input, 1))
                saltelli_sequence[index:index+self.n_input,:][np.diag_indices(self.n_input)]=base_sequence[i, :self.n_input] 
                index+=self.n_input
            
            saltelli_sequence[index,:]=base_sequence[i, self.n_input:self.n_input*2]
            index+=1
             
        return saltelli_sequence
    
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
                
        
