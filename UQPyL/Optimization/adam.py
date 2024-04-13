import numpy as np
from typing import Callable, Tuple

class Adam():
    """Adam Algorithm
    Parameters
    -------------------------
    params
    
    """
    def __init__(self, params: list, learning_rate: float=0.001, 
                    beta_1: float=0.9, beta_2: float=0.999, epsilon: float
                    
                    
                    =1e-8,
                    epoch: int=2000):
        
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.ms = [np.zeros_like(param) for param in params]
        self.vs = [np.zeros_like(param) for param in params]
        self.learning_rate_init=learning_rate
        self.epoch=epoch
        self.best_loss=np.inf
        self.loss_curve=[]
############################Interface Function##########################
    def run(self, params, func: Callable, arg: Tuple):
        for _ in range(self.epoch):
            loss, grad=func(*arg)
            
    def update_params(self, params, grads):
        """Update parameters with given gradients

        Parameters
        ----------
        params : list of length = len(coefs_) + len(intercepts_)
            The concatenated list containing coefs_ and intercepts_ in MLP
            model. Used for initializing velocities and updating params

        grads : list of length = len(params)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params
        """
        updates = self._get_updates(grads)
        for param, update in zip((p for p in params), updates):
            param += update
##########################Private Function################################  
    def _get_updates(self, grads):
        """Get the values used to update params with given gradients

        Parameters
        ----------
        grads : list, length = len(coefs_) + len(intercepts_)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params

        Returns
        -------
        updates : list, length = len(grads)
            The values to add to params
        """
        self.t += 1
        self.ms = [
            self.beta_1 * m + (1 - self.beta_1) * grad
            for m, grad in zip(self.ms, grads)
        ]
        self.vs = [
            self.beta_2 * v + (1 - self.beta_2) * (grad**2)
            for v, grad in zip(self.vs, grads)
        ]
        self.learning_rate = (
            self.learning_rate_init * np.sqrt(1 - self.beta_2**self.t)
            / (1 - self.beta_1**self.t)
        )
        updates = [
            -self.learning_rate * m / (np.sqrt(v) + self.epsilon)
            for m, v in zip(self.ms, self.vs)
        ]
        return updates