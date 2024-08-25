import abc
import numpy as np

class Sampler(metaclass=abc.ABCMeta):
    def __init__(self):
        pass
    
    def __call__(self, nt:int, nx: int) -> np.ndarray:
        return self._generate(nt, nx)
    
    def sample(self, nt:int, nx:int) -> np.ndarray:
        return self._generate(nt, nx)
    
    def rescale_to_problems(self, X:np.ndarray):
        if self.problem is not None:
            disc_var=self.problem.disc_var
            disc_range=self.problem.disc_range
            lb=self.problem.lb.reshape(1,-1)
            ub=self.problem.ub.reshape(1,-1)
            
            tmp_X=np.empty_like(X)
            # for continuous variables
            if isinstance(disc_var, list):
                disc_var=np.array(disc_var)
            pos_ind=np.where(disc_var==0)[0]
            tmp_X[:, pos_ind]=X[:, pos_ind]*(ub[:, pos_ind]-lb[:, pos_ind])+lb[:, pos_ind]
            
            # for disc variables
            for i in range(self.problem.n_input):
                if disc_var[i]==1:
                    num_interval=len(disc_range[i])
                    bins=np.linspace(0, 1, num_interval+1)
                    indices = np.digitize(X[:, i], bins[1:])
                    
                    if isinstance(disc_range[i], list):
                        tmp_X[:, i]=np.array(disc_range[i])[indices]
                    else:
                        tmp_X[:, i]=disc_range[i][indices]
            return tmp_X
        return X
    @abc.abstractmethod
    def _generate(self, nt: int, nx: int) -> np.ndarray:
        '''
        nt: the number of sampled points
        nx: the dimensions of decision variables
        
        return:
            ndarry[nt,nx]
        
        '''
        pass
    
    