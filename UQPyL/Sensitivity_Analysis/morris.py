import numpy as np
from .sa_ABC import SA
class Morris(SA):
    def __init__(self, problem, n_sample=100,
                 num_levels=4, grid_jump=1,
                 scale=None, lhs=None,
                 surrogate=None, n_surrogate_sample=50,
                 X_for_surrogate=None, Y_for_surrogate=None
                ):
        
        super().__init__(problem, n_sample,
                         scale, lhs,
                         surrogate, n_surrogate_sample, X_for_surrogate, Y_for_surrogate
                         )
        
        self.n_trajectories=n_sample
        
        self.num_levels=num_levels
        self.grid_jump=grid_jump
        
    def generate_samples(self):
        """生成Morris序列样本"""
                
        delta = 1/self.num_levels
        # if self.XInit is None:
        #     base_list=self.lhs(self.n_trajectories, self.n_input)
        #     self.XInit=np.copy(base_list)
        
        self.X_base=self.lhs(self.n_sample, self.n_input)
        X_sa =np.zeros((self.n_sample*self.n_input, self.n_input))
        idx=0
        for j in range(self.n_sample):
            base = self.X_base[j, :]
            for i in range(self.n_input):
                perturbed = np.copy(base)
                perturbed[i] += delta if perturbed[i] + delta <= 1 else -delta
                X_sa[idx, :]=perturbed
                base=perturbed
                idx+=1
        return X_sa
    
    def analyze(self):
        
        X_sa=self.generate_samples()
        
        X_sa, Y_sa=self.__check_and_scale_x_y__(X_sa, Y_sa)
            
        if self.surrogate:
            self.Y_base=self.surrogate.predict(self.X_base)
        else:
            self.Y_base=self.evaluate(self.X_base)
                
        EE=np.zeros((self.n_input, self.n_trajectories))
        
        for i in range(self.n_sample):
            X_sub=X_sa[i*self.n_trajectories:(i+1)*self.n_trajectories, :]
            Y_sub=Y_sa[i*self.n_trajectories:(i+1)*self.n_trajectories, :]
            
            Y_sub=np.vstack((self.Y_base[i, :], Y_sub))
            X_sub=np.vstack((self.X_base[i, :], X_sub))
            
            Y_diff=np.diff(Y_sub, axis=0)
            delta_diff=np.sum(np.diff(X_sub, axis=0), axis=1).reshape(-1,1)
            EE[:, i:i+1]=Y_diff/delta_diff
        
        mean_EEs = np.mean(EE, axis=1)
        std_EEs = np.std(EE, axis=1)

        # idx=np.argsort(mean_EEs)[::-1]
        
        return mean_EEs, std_EEs
    
    def summary(self):
        #TODO summary
        pass
            
        
        
        
        
        
        
                
        
        
        