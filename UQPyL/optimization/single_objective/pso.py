# Particle Swarm Optimization <Single>

import numpy as np

from ...problems import ProblemABC as Problem
from ...DoE import LHS
from ..algorithmABC import Algorithm, Population, Verbose

class PSO(Algorithm):
    '''
    Particle Swarm Optimization
    -----------------------------
    This class implements a single-objective particle swarm optimization algorithm.
    
    Methods:
        run: Run the Particle Swarm Optimization.
    
    References:
        [1] J. Kennedy and R. Eberhart, Particle swarm optimization, in Proceedings of ICNN'95 - International Conference on Neural Networks, 1995.
        [2] J. Kennedy and R. Eberhart, Swarm Intelligence, Academic Press, 2001.
        [3] M. Clerc and J. Kennedy, The particle swarm - explosion, stability, and convergence in a multidimensional complex space, IEEE Transactions on Evolutionary Computation, 2002.
        [4] Y. Shi and R. C. Eberhart, A modified particle swarm optimizer, in Proceedings of the IEEE Congress on Evolutionary Computation, 1998.
    '''
    
    name = "PSO"
    type = "EA"
    
    def __init__(self, w: float = 0.1, c1: float = 0.5, c2: float = 0.5,
                 nPop: int = 50,
                 maxIterTimes: int = 1000,
                 maxFEs: int = 50000,
                 maxTolerateTimes: int = 1000, tolerate: float = 1e-6,
                 verboseFlag: bool = True, verboseFreq: int = 10, logFlag: bool = False, saveFlag: bool = True):
        '''
        Initialize the particle swarm optimization algorithm with user-defined parameters.
        
        :param w: Inertia weight.
        :param c1: Cognitive parameter.
        :param c2: Social parameter.
        :param nPop: Population size.
        :param maxIterTimes: Maximum number of iterations.
        :param maxFEs: Maximum number of function evaluations.
        :param maxTolerateTimes: Maximum number of tolerated iterations without improvement.
        :param tolerate: Tolerance for improvement.
        :param verbose: Flag to enable verbose output.
        :param verboseFreq: Frequency of verbose output.
        :param logFlag: Flag to enable logging.
        :param saveFlag: Flag to enable saving results.
        '''
        
        super().__init__(maxFEs = maxFEs, maxIterTimes = maxIterTimes, 
                         maxTolerateTimes = maxTolerateTimes, tolerate = tolerate, 
                         verboseFlag = verboseFlag, verboseFreq = verboseFreq, logFlag=logFlag, saveFlag=saveFlag)
        
        # Set user-defined parameters
        self.setPara('w', w)
        self.setPara('c1', c1)
        self.setPara('c2', c2)
        self.setPara('nPop', nPop)
                
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem):
        '''
        Execute the particle swarm optimization on the specified problem.

        :param problem: An instance of a class derived from ProblemABC.
                        This object defines the optimization problem, including
                        the number of inputs (n_input), upper bounds (ub), lower bounds (lb), and evaluation methods.
        
        :return: The result of the optimization process.
        '''
        
        # Initialization
        # Retrieve parameter values
        w, c1, c2 = self.getParaVal('w', 'c1', 'c2')
        nPop = self.getParaVal('nPop')
        
        # Set the problem to solve
        self.problem = problem
        
        # Initialize termination conditions
        self.FEs = 0; self.iters = 0; self.tolerateTimes = 0
        
        # Generate initial population
        pop = self.initialize(nPop)
        
        # Record initial population state
        self.record(pop)
        
        # Initialize personal best and global best
        pBestPop = pop  # Personal best
        gBestPop = pop[pop.argsort()[0]]  # Global best
        vel = pop.decs  # Velocity
        
        # Iterative process
        while self.checkTermination():
            # Perform PSO operation
            pop, vel = self._operationPSO(pop, vel, pBestPop, gBestPop, w, c1, c2)
            
            # Randomly reinitialize some particles
            pop = self._randomParticle(pop)
            
            # Evaluate the population
            self.evaluate(pop)
            
            # Update personal best
            replace = np.where(pop.objs < pBestPop.objs)[0]
            pBestPop.replace(replace, pop[replace])
            
            # Update global best
            gBestPop = pBestPop[pBestPop.argsort()[0]]
            
            # Record the current state of the population
            self.record(pop)
        
        # Return the final result
        return self.result
    
    def _operationPSO(self, pop, vel, pBestPop, gBestPop, w, c1, c2):
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
        
        popDecs = pop.decs
        pBestDecs = pBestPop.decs
        gBestDecs = gBestPop.decs
        
        N, D = pop.size()
        
        particleVel = vel
        
        # Random coefficients for stochastic behavior
        r1 = np.random.random((N, D))
        r2 = np.random.random((N, D))
        
        # Update velocity
        offVel = w * particleVel + (pBestDecs - popDecs) * c1 * r1 + (gBestDecs - popDecs) * c2 * r2
        
        # Update positions
        offspringDecs = popDecs + offVel
        np.clip(offspringDecs, self.problem.lb, self.problem.ub, out=offspringDecs)
        
        return Population(offspringDecs), offVel
    
    def _randomParticle(self, pop):
        '''
        Randomly reinitialize a portion of the population.

        :param pop: Current population.
        
        :return: Population with some particles reinitialized.
        '''
        
        popDecs = pop.decs
        N, D = pop.size()
        
        # Determine number of particles to reinitialize
        n_to_reinit = int(0.1 * N)
        n_to_reinit = n_to_reinit if n_to_reinit < D else D
        
        # Randomly select particles and dimensions to mutate
        rows_to_mutate = np.random.choice(N, size=n_to_reinit, replace=False)
        cols_to_mutate = np.random.choice(D, size=n_to_reinit, replace=False)

        offspringDecs = popDecs.copy()
        
        # Reinitialize selected particles
        offspringDecs[rows_to_mutate, cols_to_mutate] = np.random.uniform(self.problem.lb[0, cols_to_mutate], self.problem.ub[0, cols_to_mutate], size=n_to_reinit)
        
        return Population(offspringDecs)