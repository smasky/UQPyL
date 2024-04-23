import numpy as np
from typing import Optional, Tuple

from .sa_ABC import SA
from ..utility import MinMaxScaler, Scaler
from ..DoE import Sampler, Morris_Sequence, LHS
from ..problems import ProblemABC as Problem
from ..surrogates import Surrogate
class Morris(SA):
    def __init__(self, problem: Problem, num_levels: int=4, extend: bool=False,
                 sampler: Sampler=Morris_Sequence(4), N_within_sampler: int=100,
                 scale: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None), 
                 surrogate: Surrogate=None, if_sampling_consistent: bool=False, 
                 sampler_for_surrogate: Sampler=LHS('classic'), N_within_surrogate_sampler: int=50,
                 X_for_surrogate: Optional[np.ndarray]=None, Y_for_surrogate: Optional[np.ndarray]=None
                ):
        
        if not isinstance(sampler, Morris_Sequence):
            raise TypeError("problem must use Morris_Sequence !")
        
        sampler.num_levels=num_levels
        
        super().__init__(problem, sampler, N_within_sampler,
                         scale, surrogate, if_sampling_consistent,
                         sampler_for_surrogate, N_within_surrogate_sampler,
                         X_for_surrogate, Y_for_surrogate
                         )
        
        self.num_levels=num_levels
        self.extend=extend
        
    def _forward(self, X_base):
        
        """生成Morris序列样本"""
        from UQPyL.DoE import FFD
        
        ffd=FFD()
        candidate=ffd.sample(self.n_input, self.num_levels)
        inds=np.random.choice(np.arange(0, candidate.shape[0]), self.N_within_sampler, replace=True)
        
        X_base=candidate[inds, :]
        
        scale=MinMaxScaler(0,1)
        X_base=scale.fit_transform(X_base)
        delta = np.floor(self.num_levels/2)/(self.num_levels-1)
        
        X_sa = np.zeros((self.N_within_sampler, self.n_input + 1, self.n_input))
        for i in range(self.N_within_sampler):
            X_sa[i, 0, :] = X_base[i, :]
            
            sequence=np.random.choice(np.arange(0, self.n_input), self.n_input, replace=False)
            
            for j in range(1, self.n_input + 1):
                X_sa[i, j, :] = X_sa[i, j - 1, :]
                ind=sequence[j-1]
                if self.extend:
                    # Original Morris method: only positive increments
                    X_sa[i, j, ind] += delta if X_sa[i, j, ind] + delta <= 1 else -delta
                else:
                    # Extended Morris method: symmetric changes
                    direction=np.random.choice([-1, 1])
                    if direction>0:
                        X_sa[i, j, ind] +=  delta if X_sa[i, j, ind] + delta<= 1 else -delta
                    else:
                        X_sa[i, j, ind] -=  delta if X_sa[i, j, ind] - delta> 0 else -delta
        X_sa = X_sa.reshape(self.N_within_sampler * (self.n_input + 1), self.n_input)
        X_sa = np.clip(X_sa, 0, 1)  # Ensure values are not outside the boundaries
                
        return scale.inverse_transform(X_sa)
    
    def analyze(self, X_sa: Optional[np.ndarray]=None, Y_sa: Optional[np.ndarray]=None, verbose=False) -> Tuple[np.ndarray, np.ndarray]:
        '''
        '''
        X_sa=self.__check_and_scale_x__(X_sa)
        self.__prepare_surrogate__()
        # X_sa=self._forward(X_base)
        # X_sa=X_sa*(self.ub-self.lb)+self.lb
        Y_sa=self.evaluate(X_sa)
        self.Y=Y_sa; self.X=X_sa
                   
        EE=np.zeros((self.n_input, self.N_within_sampler))
        
        for i in range(self.N_within_sampler):
            X_sub=X_sa[i*(self.n_input+1):(i+1)*(self.n_input+1), :]
            Y_sub=Y_sa[i*(self.n_input+1):(i+1)*(self.n_input+1), :]

            Y_diff=np.diff(Y_sub, axis=0)
            
            inds = list(np.argmax(np.diff(X_sub, axis=0) != 0, axis=1))
            new_ind=[inds.index(i) for i in range(len(inds))]
            delta_diff=np.sum(np.diff(X_sub, axis=0), axis=1).reshape(-1,1)
            ee=Y_diff/delta_diff
            EE[:, i:i+1]=ee[new_ind]
                    
        mu = np.mean(EE, axis=1)
        mu_star= np.mean(np.abs(EE), axis=1)
        sigma = np.std(EE, axis=1, ddof=1)
        
        Si={'mu':mu, 'mu_star': mu_star, 'sigma': sigma}
        self.Si=Si
        
        return Si
    
    def summary(self):
        print('Morris Sensitivity Analysis')
        print("-------------------------------------------------")
        print("Input Dimension: %d" % self.n_input)
        print("-------------------------------------------------")
        print('mu value:')
        print("-------------------------------------------------")
        for label, value in zip(self.x_labels, self.Si['mu']):
            print(f"{label}: {value:.4f}")
        print("-------------------------------------------------")
        print('mu_star value:')
        print("-------------------------------------------------")
        for label, value in zip(self.x_labels, self.Si['mu_star']):
            print(f"{label}: {value:.4f}")
        print("-------------------------------------------------")
        print("-------------------------------------------------")
        print('sigma value:')
        print("-------------------------------------------------")
        for label, value in zip(self.x_labels, self.Si['sigma']):
            print(f"{label}: {value:.4f}")
        print("-------------------------------------------------")
            
        
        
        
        
        
        
                
        
        
        