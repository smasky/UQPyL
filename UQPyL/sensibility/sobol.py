#Sobol
import numpy as np
from scipy.stats import qmc
from typing import Optional, Tuple

from .sa_ABC import SA
from ..problems import ProblemABC as Problem
from ..utility import Scaler
class Sobol(SA):
    '''
    Sobol' sensibility analysis
    --------------------------
    Parameters:
        problem: Problem
            the problem you want to analyse
        scaler: Tuple[Scaler, Scaler], default=(None, None)
            used for scaling X or Y
            
        following parameters derived from Objective problem
        n_input: the input number of the problem
        ub: the upper bound of the problem
        lb: the lower bound of the problem
    
    Methods:
        sample: Generate a sample for sobol' analysis
        analyze: perform sobol analyze from the X and Y you provided.
    
    Reference:
    [1] 
    [2]
    
    '''
    def __init__(self, problem: Problem, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                ):
        
        super().__init__(problem, scalers)
                    
#-------------------------Public Functions--------------------------------#
    def sample(self, N: int=500, skip_value: int=0, scramble: bool=True,
                    cal_second_order: bool=False) -> np.ndarray:
        '''
            Generate Sobol_sequence using Saltelli's sampling technique in [2]
            ----------------------
            Parameters:
                N: int default=500
                    the number of based X
                cal_second_order: bool default=False
                    the switch to calculate second order or not
            Returns:
                X: np.ndarray
                    if cal_second_order
                        the size of X is (N*(n_input+2), n_input)
                    else
                        the size of X is (N*(2*n_input+2), n_input)
        '''
        n_input=self.n_input; self.cal_second_order=cal_second_order
        
        M=None
        if skip_value>0 and isinstance(skip_value, int):
            
            M=skip_value
            
            if not((M&(M-1))==0 and (M!=0 and M-1!=0)):
                raise ValueError("skip value must be a power of 2!")
            
            if N>M:
                raise ValueError("skip value must be greater than N you set!")
        
        elif skip_value<0 or not isinstance(skip_value, int):
            raise ValueError("skip value must be a positive integer!")
        
        sampler=qmc.Sobol(n_input, scramble=scramble)
        
        if M:
            sampler.fast_forward(M)
        
        base_sequence=sampler.random(N)
        
        if cal_second_order:
            saltelli_sequence=np.zeros(((2*n_input+2)*N, n_input))
        else:
            saltelli_sequence=np.zeros(((n_input+2)*N, n_input))
        
        base_sequence=qmc.Sobol(N, n_input, )
        
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
        
        X=saltelli_sequence
         
        return X
     
    def analyze(self, X: Optional[np.ndarray]=None, Y: Optional[np.ndarray]=None, 
                      cal_second_order: bool= False, verbose: bool=False):
        '''
            Perform sobol' analyze
            Noted that if the X and Y is None, sample(500) is used for generate data 
                       and use the method problem.evaluate to evaluate them.
            In Sobol method, we recommend to indicate X at least.
        -------------------------
            Parameters:
                X: np.ndarray
                    the input data
                Y: np.ndarray
                    the result data
                cal_second_order: bool default=False
                    the switch to calculate second order or not
                verbose: bool
                    the switch to print analysis summary or not
            Returns:
                Si: dict
                    The type of Si is dict. And it contain 'S1', 'S2', 'ST' key value.   
        '''
        X, Y=self.__check_and_scale_xy__(X, Y)
        
        n_input=self.n_input
        
        if self.cal_second_order:
            N=int(X.shape[0]/(2*self.n_input+2))
        else:
            N=int(X.shape[0]/(self.n_input+2))
        
        Y=(Y-Y.mean())/Y.std()
        
        A, B, AB, BA=self._separate_output_values(Y, n_input, N, cal_second_order)
        
        S2 = [] if cal_second_order else None
        S1=[]; ST=[]
        
        for j in range(self.n_input):
            S1.append(self._first_order(A, AB[:, j:j+1], B))
            ST.append(self._total_order(A, AB[:, j:j+1], B))
        
        S2=np.full((self.n_input, self.n_input), np.nan)
        if self.cal_second_order:
            for j in range(self.n_input):
                for k in range(j+1, self.n_input):
                    S2[j,k]=(self._second_order(A, AB[:, j:j+1], AB[:, k:k+1], BA[:, j:j+1], B))
        
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
    def _default_sample(self):
        return self.sample(500, self.cal_second_order)
        
    def _second_order(self, A, AB1, AB2, BA, B):
        Y=np.r_[A,B]
        
        Vjk=np.mean(BA*AB2- A*B, axis=0)/np.var(Y, axis=0)
        Sj=self.first_order(A, AB1, B)
        Sk=self.first_order(A, AB2, B)
        
        return Vjk-Sj-Sk
       
    def _first_order(self, A, AB, B):
        Y=np.r_[A,B]
        
        return np.mean(B*(AB-A), axis=0)/np.var(Y, axis=0)
    
    def _total_order(self, A, AB, B):
        Y=np.r_[A,B]
        
        return 0.5*np.mean((A-AB)**2, axis=0)/np.var(Y, axis=0)
            
    def _separate_output_values(self, Y_sa, D, N, cal_second_order):
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
                
    def default_sample(self):
        return self.sample(500)
