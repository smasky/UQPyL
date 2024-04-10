import numpy as np
from ..Experiment_Design import LHS

lhs=LHS('center')
class MORRIS():
    def __init__(self, problem, N_trajectories=100, surrogate=None, XInit=None, YInit=None, 
                    num_levels=4, grid_jump=1):
        self.evaluate=problem.evaluate
        self.surrogate=surrogate
        self.lb=problem.lb;self.ub=problem.ub
        self.dim=problem.dim
        
        self.XInit=XInit; self.YInit=YInit
        
        self.N_trajectories=N_trajectories
        
        self.num_levels=num_levels
        self.grid_jump=grid_jump
        
    def set_sampling_params(self, num_levels=4, grid_jump=1):
        
        self.num_levels = num_levels
        self.grid_jump = grid_jump
    
    def generate_samples(self):
        """生成Morris序列样本"""
                
        delta = 1/self.num_levels
        if self.XInit is None:
            base_list=lhs(self.N_trajectories, self.dim)
            self.XInit=np.copy(base_list)
        sequences =[]
        for j in range(self.N_trajectories):
            base = base_list[j, :]
            sequence=[]
            for i in range(self.dim):
                perturbed = np.copy(base)
                perturbed[i] += delta if perturbed[i] + delta <= 1 else -delta
                sequence.append(perturbed)
                base=perturbed
            sequences.append(np.array(sequence))
        
        return sequences
    
    def analyze(self):
        
        sequences=self.generate_samples()
        if self.YInit is None:
            self.YInit=self.evaluate(self.XInit)
            
        if self.surrogate:
                self.surrogate.fit(self.XInit, self.YInit)
                
        EE=np.zeros((self.dim, self.N_trajectories))
        for i in range(self.N_trajectories):
            sequence=sequences[i]
            if self.surrogate:
                samples_Y=self.surrogate.predict(sequence)
            else:
                samples_Y=self.evaluate(sequence)
            
            samples_Y=np.vstack((self.YInit[i,:], samples_Y))
            sequence=np.vstack((self.XInit[i, :], sequence))
            
            Y_diff=np.diff(samples_Y, axis=0)
            delta_diff=np.sum(np.diff(sequence, axis=0), axis=1).reshape(-1,1)
            EE[:, i:i+1]=Y_diff/delta_diff
        
        mean_EEs = np.mean(EE, axis=1)
        std_EEs = np.std(EE, axis=1)
        
        idx=np.argsort(mean_EEs)[::-1]
        
        return idx, mean_EEs, std_EEs
            
        
        
        
        
        
        
                
        
        
        