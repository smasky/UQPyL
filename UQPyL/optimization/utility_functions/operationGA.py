import numpy as np
import math

from ..population import Population

def operationGA(self, pop: Population, proC: float, disC: float, proM: float, disM: float):
        '''
            GA Operation
        '''
        n, d=pop.size()
        
        parent1=pop[:math.floor(n/2),:]
        parent2=pop[math.floor(n/2):math.floor(n/2)*2,:]
        
        n, d = parent1.shape
        beta = np.zeros_like(parent1)
        mu = np.random.rand(n, d)
        #Crossover
        beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1 / (disC + 1))
        beta[mu > 0.5] = np.power(2 - 2 * mu[mu > 0.5], -1 / (disC + 1))
        beta = beta * (-1) ** np.random.randint(0, 2, size=(n, d))
        beta[np.random.rand(n, d) < 0.5] = 1
        beta[np.repeat(np.random.rand(n, 1) > proC, d, axis=1)] = 1

        off1=(parent1 + parent2) / 2 + (parent1 - parent2) * beta / 2
        off2=(parent1 + parent2) / 2 - (parent1 - parent2) * beta / 2 
        offspring=off1.merge(off2)
        
        #polynomial mutation
        lower = np.repeat(self.problem.lb, 2 * n, axis=0)
        upper = np.repeat(self.problem.ub, 2 * n, axis=0)
        sita = np.random.rand(2 * n, d) < proM / d
        mu = np.random.rand(2 * n, d)
        
        offspring.clip(lower, upper)
        temp = sita & (mu <= 0.5)        
        t1 = (1 - 2 * mu[temp]) * np.power(1 - (offspring.decs[temp] - lower[temp]) / (upper[temp] - lower[temp]), disM + 1)
        offspring.decs[temp] = offspring.decs[temp] + (np.power(2 * mu[temp] + t1, 1 / (disM + 1)) - 1) *(upper[temp] - lower[temp])
        
        temp = sita & (mu > 0.5)
        t2 = 2 * (mu[temp] - 0.5) * np.power(1 - (upper[temp] - offspring.decs[temp]) / (upper[temp] - lower[temp]), disM + 1)
        offspring.decs[temp] = offspring.decs[temp] + (upper[temp] - lower[temp]) * (1 - np.power(2 * (1 - mu[temp]) + t2, 1 / (disM + 1)))
        
        return offspring