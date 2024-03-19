from .base_kernel import Kernel
import numpy as np

class Gaussian(Kernel):
    name="Gaussian"
    def __init__(self,gamma=1):
        self.gamma=gamma
    def evaluate(self,dist):
        return np.exp(-1*self.gamma*np.power(dist,2))