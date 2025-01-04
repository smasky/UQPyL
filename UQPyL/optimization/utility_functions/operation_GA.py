import numpy as np
import math

from ..population import Population

def operationGA(matingPool, ub, lb, proC=1, disC=20, proM=1, disM=20):
    '''
        GA Operation: crossover and mutation
    '''
    
    popDec = np.copy(matingPool.decs)
        
    NN = len(matingPool)
    
    # Crossover
    parent1 = popDec[:math.floor(NN/2)]
    parent2 = popDec[math.floor(NN/2):math.floor(NN/2)*2]
    
    N, D = parent1.shape
    beta = np.zeros(shape=(N, D))
    mu = np.random.rand(N, D)

    beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1 / (disC + 1))
    beta[mu > 0.5] = np.power(2 - 2 * mu[mu > 0.5], -1 / (disC + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, size=(N, D))
    beta[np.random.rand(N, D) < 0.5] = 1
    beta[np.repeat(np.random.rand(N, 1) > proC, D, axis=1)] = 1

    off1 = (parent1 + parent2) / 2 + (parent1 - parent2) * beta / 2
    off2 = (parent1 + parent2) / 2 - (parent1 - parent2) * beta / 2 
    
    offspring=np.vstack((off1, off2))
    
    # Polynomial mutation
    lower = np.repeat(lb, 2 * N, axis=0)
    upper = np.repeat(ub, 2 * N, axis=0)
    sita = np.random.rand(2 * N, D) < proM / D
    mu = np.random.rand(2 * N, D)
    
    np.clip(offspring, lower, upper, out=offspring)
    
    temp = sita & (mu <= 0.5)        
    t1 = (1 - 2 * mu[temp]) * np.power(1 - (offspring[temp] - lower[temp]) / (upper[temp] - lower[temp]), disM + 1)
    offspring[temp] = offspring[temp] + (np.power(2 * mu[temp] + t1, 1 / (disM + 1)) - 1) *(upper[temp] - lower[temp])
    
    temp = sita & (mu > 0.5)
    t2 = 2 * (mu[temp] - 0.5) * np.power(1 - (upper[temp] - offspring[temp]) / (upper[temp] - lower[temp]), disM + 1)
    offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * (1 - np.power(2 * (1 - mu[temp]) + t2, 1 / (disM + 1)))
    
    return Population(offspring)
       

def operationGAHalf(matingPool, ub, lb, proC, disC, proM, disM):
    
    popDecs = matingPool.decs
    
    NN = len(matingPool)
    
    # Crossover
    parent1=popDecs[:math.floor(NN/2)]
    parent2=popDecs[math.floor(NN/2):math.floor(NN/2)*2]
    
    N, D = parent1.shape
    
    beta = np.zeros(shape=(N,D))
    mu = np.random.rand(N, D)

    beta[mu <= 0.5] = np.power(2 * mu[mu <= 0.5], 1 / (disC + 1))
    beta[mu > 0.5] = np.power(2 - 2 * mu[mu > 0.5], -1 / (disC + 1))
    beta = beta * (-1) ** np.random.randint(0, 2, size=(N, D))
    beta[np.random.rand(N, D) < 0.5] = 1
    beta[np.repeat(np.random.rand(N, 1) > proC, D, axis=1)] = 1

    offspring=(parent1 + parent2) / 2 + (parent1 - parent2) * beta / 2
    
    N, D=offspring.shape
    
    # Polynomial mutation
    lower = np.repeat(lb, N, axis=0)
    upper = np.repeat(ub, N, axis=0)
    sita = np.random.rand(N, D) < proM / D
    mu = np.random.rand(N, D)
    
    np.clip(offspring, lower, upper, out=offspring)
    
    temp = sita & (mu <= 0.5)        
    t1 = (1 - 2 * mu[temp]) * np.power(1 - (offspring[temp] - lower[temp]) / (upper[temp] - lower[temp]), disM + 1)
    offspring[temp] = offspring[temp] + (np.power(2 * mu[temp] + t1, 1 / (disM + 1)) - 1) *(upper[temp] - lower[temp])
    
    temp = sita & (mu > 0.5)
    t2 = 2 * (mu[temp] - 0.5) * np.power(1 - (upper[temp] - offspring[temp]) / (upper[temp] - lower[temp]), disM + 1)
    offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * (1 - np.power(2 * (1 - mu[temp]) + t2, 1 / (disM + 1)))
    
    return Population(offspring)