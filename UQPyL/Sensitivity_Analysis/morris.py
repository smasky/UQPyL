import numpy as np

class Morris():
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
        # levels = np.linspace(0, 1, self.num_levels + 1)
        # samples = np.zeros((self.NSample, self.dim))
        # for idx in range(self.NSample):
        #     sample = []
        #     level=np.random.choice(levels, size=self.dim, replace=True)
        #     delta=(self.ub-self.lb)/self.grid_jump
        #     sample.append(self.lb+level*delta)
        #     samples[idx, :]=sample
        # return samples
        
        delta = self.num_levels / (2 * self.num_levels - 2)
        
        sequences =[]
        for _ in range(self.N_trajectories):
            sequence=[]
            base = np.random.uniform(0, self.num_levels - 1, self.dim) / (self.num_levels - 1)
            base = np.round(base, decimals=2)
            for i in range(self.dim):
                perturbed = np.copy(base)
                perturbed[i] += delta if perturbed[i] + delta <= 1 else -delta
                perturbed = np.round(perturbed, decimals=2)
                sequence.append(perturbed.reshape(1,-1))
            sequences.append(np.array(sequence))
        
        return sequences
    def analyze(self):
        
        samples_X=self.generate_samples()
        
        if self.surrogate:
            self.surrogate.fit(self.XInit, self.YInit)
            samples_Y=self.surrogate.predict(samples_X)
        else:
            samples_Y=self.evaluate(samples_X)
        
        
        
        
        
                
        
        
        