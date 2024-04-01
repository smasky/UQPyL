import numpy as np

class Morris():
    def __init__(self, problem, surrogate=None,NSample=100, XInit=None, YInit=None, 
                    num_levels=4, grid_jump=1):
        self.evaluate=problem.evaluate
        self.surrogate=surrogate
        self.lb=problem.lb;self.ub=problem.ub
        self.dim=problem.dim
        
        self.XInit=XInit; self.YInit=YInit
        self.NSample=NSample
        
        self.num_levels=num_levels
        self.grid_jump=grid_jump
        
    def set_sampling_params(self, num_levels=4, grid_jump=1):
        
        self.num_levels = num_levels
        self.grid_jump = grid_jump
    
    def generate_samples(self):
        """生成Morris序列样本"""
        levels = np.linspace(0, 1, self.num_levels + 1)
        samples = np.zeros((self.NSample, self.dim))
        for idx in range(self.NSample):
            sample = []
            level=np.random.choice(levels, size=self.dim, replace=True)
            delta=(self.ub-self.lb)/self.grid_jump
            sample.append(self.lb+level*delta)
            samples[idx, :]=sample
        return samples
    
    def analyze(self):
        
        samples_X=self.generate_samples()
        if self.surrogate:
            self.surrogate.fit(self.XInit, self.YInit)
            samples_Y=self.surrogate.predict(samples_X)
        else:
            samples_Y=self.evaluate(samples_X)
        
                
        
        
        