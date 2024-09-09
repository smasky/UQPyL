import numpy as np
import math

def operationGA(matingPool, ub, lb, proC=1, disC=20, proM=1, disM=20):
        '''
            GA Operation: crossover and mutation
        '''
        
        n_samples=len(matingPool)
        # Crossover
        parent1=matingPool[:math.floor(n_samples/2)]
        parent2=matingPool[math.floor(n_samples/2):math.floor(n_samples/2)*2]
        
        n, d = parent1.size()
        beta = np.zeros(shape=(n,d))
        mu = np.random.rand(n, d)

        beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1 / (disC + 1))
        beta[mu > 0.5] = np.power(2 - 2 * mu[mu > 0.5], -1 / (disC + 1))
        beta = beta * (-1) ** np.random.randint(0, 2, size=(n, d))
        beta[np.random.rand(n, d) < 0.5] = 1
        beta[np.repeat(np.random.rand(n, 1) > proC, d, axis=1)] = 1

        off1=(parent1 + parent2) / 2 + (parent1 - parent2) * beta / 2
        off2=(parent1 + parent2) / 2 - (parent1 - parent2) * beta / 2 
        offspring=off1.merge(off2)
        offspring.clip(lb, ub)
        
        # Polynomial mutation
        lower = np.repeat(lb, 2 * n, axis=0)
        upper = np.repeat(ub, 2 * n, axis=0)
        sita = np.random.rand(2 * n, d) < proM / d
        mu = np.random.rand(2 * n, d)
        
        temp = sita & (mu <= 0.5)        
        t1 = (1 - 2 * mu[temp]) * np.power(1 - (offspring.decs[temp] - lower[temp]) / (upper[temp] - lower[temp]), disM + 1)
        offspring.decs[temp] = offspring.decs[temp] + (np.power(2 * mu[temp] + t1, 1 / (disM + 1)) - 1) *(upper[temp] - lower[temp])
        
        temp = sita & (mu > 0.5)
        t2 = 2 * (mu[temp] - 0.5) * np.power(1 - (upper[temp] - offspring.decs[temp]) / (upper[temp] - lower[temp]), disM + 1)
        offspring.decs[temp] = offspring.decs[temp] + (upper[temp] - lower[temp]) * (1 - np.power(2 * (1 - mu[temp]) + t2, 1 / (disM + 1)))
        
        return offspring

def operationGAHalf(matingPool, ub, lb, proC, disC, proM, disM):
    
    n_samples=len(matingPool)
    # Crossover
    parent1=matingPool[:math.floor(n_samples/2)]
    parent2=matingPool[math.floor(n_samples/2):math.floor(n_samples/2)*2]
    
    n, d = parent1.size()
    beta = np.zeros(shape=(n,d))
    mu = np.random.rand(n, d)

    beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1 / (disC + 1))
    beta[mu > 0.5] = np.power(2 - 2 * mu[mu > 0.5], -1 / (disC + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, size=(n, d))
    beta[np.random.rand(n, d) < 0.5] = 1
    beta[np.repeat(np.random.rand(n, 1) > proC, d, axis=1)] = 1

    offspring=(parent1 + parent2) / 2 + (parent1 - parent2) * beta / 2
    
    n, d=offspring.size()
    
    # Polynomial mutation
    lower = np.repeat(lb, n, axis=0)
    upper = np.repeat(ub, n, axis=0)
    sita = np.random.rand(n, d) < proM / d
    mu = np.random.rand(n, d)
    
    offspring.clip(lower, upper)
    temp = sita & (mu <= 0.5)        
    t1 = (1 - 2 * mu[temp]) * np.power(1 - (offspring.decs[temp] - lower[temp]) / (upper[temp] - lower[temp]), disM + 1)
    offspring.decs[temp] = offspring.decs[temp] + (np.power(2 * mu[temp] + t1, 1 / (disM + 1)) - 1) *(upper[temp] - lower[temp])
    
    temp = sita & (mu > 0.5)
    t2 = 2 * (mu[temp] - 0.5) * np.power(1 - (upper[temp] - offspring.decs[temp]) / (upper[temp] - lower[temp]), disM + 1)
    offspring.decs[temp] = offspring.decs[temp] + (upper[temp] - lower[temp]) * (1 - np.power(2 * (1 - mu[temp]) + t2, 1 / (disM + 1)))
        
    return offspring