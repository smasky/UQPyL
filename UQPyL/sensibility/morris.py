import numpy as np
from typing import Optional, Tuple

from .sa_ABC import SA
from ..utility import Scaler
from ..problems import ProblemABC as Problem
class Morris(SA):
    '''
    Morris analysis
    ------------------------------------------------------
    Parameters:
        problem: Problem
            the problem you want to analyse
        scaler: Tuple[Scaler, Scaler], default=(None, None)
            used for scaling X or Y
        num_levels (p): int, default=4, recommended value: 4 to 10
                each x_i would take value on {0, 1/(p-1), 2/(p-1), ..., 1}. 
                Morris et al[1]. recommend the num_levels to be even and range from 4 and 10.
        
        Following parameters derived from the variable 'problem'
        n_input: the input number of the problem
        ub: the upper bound of the problem
        lb: the lower bound of the problem
    
    Methods:
        sample: Generate a sample for morris analysis
        analyze: perform morris analyze from the X and Y you provided.
        
    Examples:
        >>> mor_method=Morris_Sequence(problem)
        >>> X=mor_method.sample(100, 4)
        >>> Y=problem.evaluate(X)
        >>> mor_method.analyze(X, Y)
    
    References:
        [1] Max D. Morris (1991) Factorial Sampling Plans for Preliminary Computational Experiments, 
                                 Technometrics, 33:2, 161-174
                                 doi: 10.2307/1269043
        [2] SALib, https://github.com/SALib/SALib
    '''
    def __init__(self, problem: Problem, scalers: Tuple[Optional[Scaler], Optional[Scaler]]=(None, None),
                       num_levels: int=4):
          
        super().__init__(problem, scalers)
        self.num_levels=num_levels
        
    def sample(self, num_trajectory: int=500) -> np.ndarray:
        '''
        Generate a sample for Morris analysis
        ---------------------------------------
        Parameters:  
            num_trajectory: int, default=500, recommend value: 500 to 1000
                The number of trajectories. In general, the size of each trajectory is n_input+1 
            
        Returns:
            samples: np.ndarray
                Noted that The size of samples are (num_trajectory*(n_input+1), n_input)
        
        '''
        nt=num_trajectory; nx=self.n_input; num_levels=self.num_levels
        
        X=np.zeros((nt*(nx+1), nx))
        
        for i in range(nt):
            X[i*(nx+1):(i+1)*(nx+1), :]=self._generate_trajectory(nx, num_levels)
        
        return self.transform_into_problem(X)
        
    def analyze(self, X: Optional[np.ndarray]=None, Y: Optional[np.ndarray]=None, verbose: bool=False) -> dict:
        '''
            Perform morris analysis
            
            Noted that if the X and Y is None, sample(500, 4) is used for generate data 
                       and use the method problem.evaluate to evaluate them.
            
            -------------------------
            Parameters:
                X: np.ndarray
                    the input data
                Y: np.ndarray
                    the result data
                verbose: bool
                    the switch to print analysis summary or not

            Returns:
                Si: dict
                    The type of Si is dict. And it contain 'mu', 'mu_star', 'sigma' key value.
        '''
        n_input=self.n_input; num_trajectory=int(X.shape[0]/(n_input+1))
        
        X, Y=self.__check_and_scale_xy__(X, Y)

        EE=np.zeros((n_input, num_trajectory))
        
        N=int(X.shape[0]/self.num_levels)
        
        for i in range(N):
            X_sub=X[i*(n_input+1):(i+1)*(n_input+1), :]
            Y_sub=Y[i*(n_input+1):(i+1)*(n_input+1), :]

            Y_diff=np.diff(Y_sub, axis=0)
            
            tmp_indice = list(np.argmax(np.diff(X_sub, axis=0) != 0, axis=1))
            indice=[tmp_indice.index(i) for i in range(len(tmp_indice))]
            delta_diff=np.sum(np.diff(X_sub, axis=0), axis=1).reshape(-1,1)
            ee=Y_diff/delta_diff
            EE[:, i:i+1]=ee[indice]
            
        mu = np.mean(EE, axis=1)
        mu_star= np.mean(np.abs(EE), axis=1)
        sigma = np.std(EE, axis=1, ddof=1)
        
        Si={'mu':mu, 'mu_star': mu_star, 'sigma': sigma}
        self.Si=Si
        
        if verbose:
            self.summary()
        
        return Si
    
    def summary(self):
        '''
            print analysis summary
        '''
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
    #-------------------------Private Function-------------------------------------#
    def _generate_trajectory(self, nx: int, num_levels: int=4) -> np.ndarray:
        '''
            Generate a random trajectory from Reference[1]
        '''
        delta=num_levels/(2*(num_levels-1))
        
        B=np.tril(np.ones([nx + 1, nx], dtype=int), -1)
        
        # from paper[1] page 164
        D_star = np.diag(np.random.choice([-1, 1], nx)) #step1
        J=np.ones((nx+1, nx))
        
        levels_grids=np.linspace(0, 1-delta, int(num_levels / 2))
        x_star=np.random.choice(levels_grids, nx).reshape(1,-1) #step2
        
        P_star=np.zeros((nx,nx))
        cols = np.random.choice(nx, nx, replace=False)
        P_star[np.arange(nx), cols]=1 #step3
        
        element_a = J[0, :] * x_star
        element_b = P_star.T
        element_c = np.matmul(2.0 * B, element_b)
        element_d = np.matmul((element_c - J), D_star)

        B_star = element_a + (delta / 2.0) * (element_d + J)
    
        return B_star
        
    def _default_sample(self):
        return self.sample(500)
        
        
        
        
                
        
        
        