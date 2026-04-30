# Particle Swarm Optimization <Single>

import numpy as np

from typing import Optional

from ..base import AlgorithmABC
from ..population import Population
from ..core.constraint import betterMask

class PSO(AlgorithmABC):
    """
    Single-objective particle swarm optimization.
    """
    
    name = "PSO"
    alg_type = "EA"
    
    def __init__(self, w: float = 0.1, c1: float = 0.5, c2: float = 0.5,
                 nPop: int = 50,
                 maxIters: int = 1000,
                 maxFEs: int = 50000,
                 maxTolerates: int = 1000, tolerate: float = 1e-6,
                 verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = False, saveFlag: bool = True,
                 saveFreq: int = 100):
        """
        Initialize the algorithm.

        :param w: Inertia weight.
        :param c1: Cognitive coefficient.
        :param c2: Social coefficient.
        :param nPop: Population size.
        :param maxFEs: Maximum number of function evaluations.
        :param maxIters: Maximum number of iterations.
        :param maxTolerates: Maximum tolerated non-improving iterations.
        :param tolerate: Improvement tolerance.
        :param verboseFlag: Whether to print terminal output.
        :param verboseFreq: Summary output frequency.
        :param logFlag: Whether to save full text logs.
        :param saveFlag: Whether to save sqlite results.
        :param saveFreq: Snapshot save frequency.
        """
        
        super().__init__(maxFEs = maxFEs, maxIters = maxIters, 
                         maxTolerates = maxTolerates, tolerate = tolerate, 
                         verboseFlag = verboseFlag, verboseFreq = verboseFreq, logFlag=logFlag, saveFlag=saveFlag,
                         saveFreq = saveFreq)
        
        # Set user-defined parameters
        self.setParaVal('w', w)
        self.setParaVal('c1', c1)
        self.setParaVal('c2', c2)
        self.setParaVal('nPop', nPop)
                
    def run(self, problem, seed: Optional[int] = None):
        """
        Run the algorithm on the given problem.

        :param problem: Problem instance.
        :param seed: Random seed.
        :return OptResult: Final optimization result.
        """
        
        # setup algorithm
        self.setup(problem, seed)
        
        # Initialization
        # Retrieve parameter values
        w, c1, c2 = self.getParaVal('w', 'c1', 'c2')
        nPop = self.getParaVal('nPop')
        
        # Generate initial population
        pop = self.initPop(nPop)
        self.update(pop)
                
        # Initialize personal best and global best
        pBest = pop  # Personal best
        gBest = pop.getBest(k=1)  # Global best
        vel = pop.decs  # Velocity
        
        # Iterative process
        while self.checkTermination(pop):
            # Perform PSO operation
            popDecs, vel = self._psoOperator(pop.decs, vel, pBest.decs, gBest.decs, w, c1, c2)
            
            # Randomly reinitialize some particles
            popDecs = self._randomParticle(popDecs)
            
            pop = Population(popDecs)
            # Evaluate the population
            self.evaluate(pop)
            
            # Update personal best
            replace = np.where(betterMask(pop.objs, pop.cons, pBest.objs, pBest.cons, pop.conWgt))[0]
            pBest.replace(replace, pop[replace])
            
            # Update global best
            gBest = pBest.getBest(k=1)
            self.update(pop)
            
        # Return the final result
        return self.finalize()
    
    def _psoOperator(self, popDecs, vel, pBestDecs, gBestDecs, w, c1, c2):
        '''
        Perform the particle swarm optimization operation.

        :param pop: Current population.
        :param vel: Current velocity of particles.
        :param pBestPop: Personal best population.
        :param gBestPop: Global best particle.
        :param w: Inertia weight.
        :param c1: Cognitive parameter.
        :param c2: Social parameter.
        
        :return: Updated population and velocity.
        '''
            
        N, D = popDecs.shape
        
        particleVel = vel
        
        # Random coefficients for stochastic behavior
        r1 = np.random.random((N, D))
        r2 = np.random.random((N, D))
        
        # Update velocity
        offVel = w * particleVel + (pBestDecs - popDecs) * c1 * r1 + (gBestDecs - popDecs) * c2 * r2
        
        # Update positions
        offspringDecs = popDecs + offVel
        np.clip(offspringDecs, self.problem.lb, self.problem.ub, out=offspringDecs)
        
        return offspringDecs, offVel
    
    def _randomParticle(self, popDecs):
        '''
        Randomly reinitialize a portion of the population.

        :param pop: Current population.
        
        :return: Population with some particles reinitialized.
        '''
        
        N, D = popDecs.shape
        
        # Determine number of particles to reinitialize
        n_to_reinit = int(0.1 * N)
        n_to_reinit = n_to_reinit if n_to_reinit < D else D
        
        # Randomly select particles and dimensions to mutate
        rows_to_mutate = np.random.choice(N, size=n_to_reinit, replace=False)
        cols_to_mutate = np.random.choice(D, size=n_to_reinit, replace=False)

        offspringDecs = popDecs.copy()
        
        # Reinitialize selected particles
        offspringDecs[rows_to_mutate, cols_to_mutate] = np.random.uniform(self.problem.lb[0, cols_to_mutate], self.problem.ub[0, cols_to_mutate], size=n_to_reinit)
        
        return offspringDecs
