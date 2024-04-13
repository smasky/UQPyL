import numpy as np

from .sa_ABC import SA
from ..utility import MinMaxScaler

class Morris(SA):
    def __init__(self, problem, N_within_sampler=100,
                 num_levels=4, grid_jump=1,
                 scale=None, sampler=None,
                 surrogate=None, if_sampling_consistent=False, 
                 sampler_for_surrogate=None, N_within_surrogate_sampler=50,
                 X_for_surrogate=None, Y_for_surrogate=None
                ):
        
        super().__init__(problem, N_within_sampler,
                         scale, sampler,
                         surrogate, if_sampling_consistent,
                         sampler_for_surrogate, N_within_surrogate_sampler,
                        X_for_surrogate, Y_for_surrogate
                         )
        
        self.num_levels=num_levels
        self.grid_jump=grid_jump
        
    def generate_samples(self, X_base):
        """生成Morris序列样本"""
        scale=MinMaxScaler(0,1)
        X_base=scale.fit_transform(X_base)
        delta = 1/self.num_levels
        # if self.XInit is None:
        #     base_list=self.lhs(self.n_trajectories, self.n_input)
        #     self.XInit=np.copy(base_list)
        
        X_sa =np.zeros((self.N_within_sampler*self.n_input, self.n_input))
        idx=0
        for j in range(self.N_within_sampler):
            base = X_base[j, :]
            for i in range(self.n_input):
                perturbed = np.copy(base)
                perturbed[i] += delta if perturbed[i] + delta <= 1 else -delta
                X_sa[idx, :]=perturbed
                base=perturbed
                idx+=1
                
        return scale.inverse_transform(X_sa)
    
    def analyze(self):
        
        ##forward process
        X_base=self.__check_and_scale_x__(X_sa)
        self.__prepare_surrogate__()
        X_sa=self.generate_samples(X_base)
        Y_sa=self.evaluate(X_sa)
        Y_base=self.evaluate(X_base)
                   
        EE=np.zeros((self.n_input, self.N_within_sampler))
        
        for i in range(self.N_within_sampler):
            X_sub=X_sa[i*self.N_within_sampler:(i+1)*self.N_within_sampler, :]
            Y_sub=Y_sa[i*self.N_within_sampler:(i+1)*self.N_within_sampler, :]
            
            Y_sub=np.vstack((Y_base[i, :], Y_sub))
            X_sub=np.vstack((X_base[i, :], X_sub))
            
            Y_diff=np.diff(Y_sub, axis=0)
            delta_diff=np.sum(np.diff(X_sub, axis=0), axis=1).reshape(-1,1)
            EE[:, i:i+1]=Y_diff/delta_diff
        
        mean_EEs = np.mean(EE, axis=1)
        std_EEs = np.std(EE, axis=1)
        
        return mean_EEs, std_EEs
    
    def summary(self):
        #TODO summary
        pass
            
        
        
        
        
        
        
                
        
        
        