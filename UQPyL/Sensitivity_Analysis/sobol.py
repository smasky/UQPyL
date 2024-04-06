#Sobol
from ..Experiment_Design import LHS
from scipy.stats import qmc
import numpy as np
lhs=LHS('center')

class SOBOL():
    def __init__(self, problem, NSample=128, scramble=True, skip_value=0, cal_second_order=False, surrogate=None, NSurrogate=50, XInit=None, YInit=None):
        self.evaluate=problem.evaluate
        self.lb=problem.lb; self.ub=problem.ub
        self.dim=problem.dim
        self.surrogate=surrogate
        
        self.NSample=NSample
        self.cal_second_order=cal_second_order
        self.scramble=scramble
        self.skip_value=skip_value
            
        self.NSample=NSample
        
        self.YInit=YInit
        if self.surrogate:
            if XInit is None:
                self.XInit=lhs(NSurrogate, self.dim)*(self.ub-self.lb)+self.lb
        
        if skip_value>0 and isinstance(skip_value, int):
            M=skip_value
            if not((M&(M-1))==0 and (M!=0 and M-1!=0)):
                raise ValueError("skip value must be a power of 2!")
            
            if NSample>M:
                raise ValueError("skip value must be greater than NSample!")
        elif skip_value<0 or not isinstance(skip_value, int):
            raise ValueError("skip value must be a positive integer!")    
        
    def generate_samples(self):
        #Sobol sequence
        D=self.dim
        N=self.NSample
        qrng=qmc.Sobol(d=2*D, scramble=self.scramble)
        
        if self.skip_value>0:
            qrng.fast_forward(self.skip_value)
            
        base_sequence=qrng.random(N)
        
        if self.cal_second_order:
            saltelli_sequence=np.zeros(((2*D+2)*N, D))
        else:
            saltelli_sequence=np.zeros(((D+2)*N, D))
        
        index=0
        for i in range(N):
            saltelli_sequence[index, :]=base_sequence[i, :self.dim]

            index+=1
            
            saltelli_sequence[index:index+D,:]=np.tile(base_sequence[i, self.dim:], (D, 1))
            saltelli_sequence[index:index+D,:][np.diag_indices(D)]=base_sequence[i, :D]               
            index+=D
           
            if self.cal_second_order:
                saltelli_sequence[index:index+D,:]=np.tile(base_sequence[i, :self.dim], (D, 1))
                saltelli_sequence[index:index+D,:][np.diag_indices(D)]=base_sequence[i, D:] 
                index+=D
            
            saltelli_sequence[index,:]=base_sequence[i, D:D*2]
            index+=1
        
        saltelli_sequence=saltelli_sequence*(self.ub-self.lb)+self.lb
        
        return saltelli_sequence
    
    def analyze(self):
        cal_second_order=self.cal_second_order
        D=self.dim
        N=self.NSample
        
        X_seq=self.generate_samples()
        if self.surrogate:
            if self.XInit is None:
                self.XInit=lhs(self.NSurrogate, self.dim)*(self.ub-self.lb)+self.lb
                self.YInit=self.evaluate(self.XInit)
            self.surrogate.fit(self.XInit, self.YInit)
            Y_seq=self.surrogate.predict(X_seq)
        else:
            Y_seq=self.evaluate(X_seq)
        
        Y_seq=(Y_seq-Y_seq.mean())/Y_seq.std()
        A, B, AB, BA=self.separate_output_values(Y_seq, D, N, cal_second_order)
        
        S1=[]; S2=[];ST=[]
        for j in range(D):
            S1.append(self.first_order(A, AB[:, j:j+1], B))
            ST.append(self.total_order(A, AB[:, j:j+1], B))

        if cal_second_order:
            for j in range(D):
                for k in range(j+1, D):
                    S2.append(self.second_order(A, AB[:, j:j+1], AB[:, k:k+1], BA[:, j:j+1], B))

        S={}
        S['Si']=S1
        S['ST']=ST
        if S2:
            S['S2']=S2
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
            
        
    def separate_output_values(self, Y_seq, D, N, cal_second_order):
        AB=np.zeros((N,D))
        BA=np.zeros((N,D)) if cal_second_order else None
        
        step=2*D+2 if cal_second_order else D+2
        
        total=Y_seq.shape[0]
        
        A=Y_seq[0:total:step, :]
        B=Y_seq[(step-1):total:step, :]
        
        for j in range(D):
            AB[:, j]=Y_seq[(j+1):total:step, 0]
            if cal_second_order:
                BA[:, j]=Y_seq[(j+1+D):total:step, 0]
        
        return A, B, AB, BA
                
        
