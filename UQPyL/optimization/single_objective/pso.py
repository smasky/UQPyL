# Particle Swarm Optimization <Single>

import numpy as np

from ...problems import ProblemABC as Problem
from ...DoE import LHS
from ..algorithmABC import Algorithm, Population, Verbose
class PSO(Algorithm):
    '''
        Particle Swarm Optimization
        -----------------------------
        Attributes:
            problem: Problem
                the problem you want to solve, including the following attributes:
                n_input: int
                    the input number of the problem
                ub: 1d-np.ndarray or float
                    the upper bound of the problem
                lb: 1d-np.ndarray or float
                    the lower bound of the problem
                evaluate: Callable
                    the function to evaluate the input
            n_sample: int, default=50
                the number of samples as the population
            w: float, default=0.1
                the inertia weight
            c1: float, default=0.5
                the cognitive parameter
            c2: float, default=0.5
                the social parameter
            maxIterTimes: int, default=1000
                the maximum iteration times
            maxFEs: int, default=50000
                the maximum function evaluations
            maxTolerateTimes: int, default=1000
                the maximum tolerate times which the best objective value does not change
            tolerate: float, default=1e-6
                the tolerate value which the best objective value does not change
        Methods:
            run: run the Particle Swarm Optimization
        
        References:
            [1] J. Kennedy and R. Eberhart, Particle swarm optimization, in Proceedings of ICNN'95 - International Conference on Neural Networks, 1995.
            [2] J. Kennedy and R. Eberhart, Swarm Intelligence, Academic Press, 2001.
            [3] M. Clerc and J. Kennedy, The particle swarm - explosion, stability, and convergence in a multidimensional complex space, IEEE Transactions on Evolutionary Computation, 2002.
            [4] Y. Shi and R. C. Eberhart, A modified particle swarm optimizer, in Proceedings of the IEEE Congress on Evolutionary Computation, 1998.
        
    '''
    
    name= "PSO"
    type= "EA" 
    
    def __init__(self, w: float = 0.1, c1: float = 0.5, c2: float = 0.5,
                    nPop: int = 50,
                    maxIterTimes: int = 1000,
                    maxFEs: int = 50000,
                    maxTolerateTimes: int = 1000, tolerate: float = 1e-6,
                    verbose: bool = True, verboseFreq: int = 10, logFlag: bool = False, saveFlag: bool = True):
        
            super().__init__(maxFEs=maxFEs, maxIterTimes=maxIterTimes, 
                         maxTolerateTimes=maxTolerateTimes, tolerate=tolerate, 
                         verbose=verbose, verboseFreq=verboseFreq, logFlag=logFlag, saveFlag=saveFlag)
            
            #user-define setting
            self.setParameters('w', w)
            self.setParameters('c1', c1)
            self.setParameters('c2', c2)
            self.setParameters('nPop', nPop)
                
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem):
        
        #Initialization
        #Parameter Setting
        w, c1, c2 = self.getParaValue('w', 'c1', 'c2')
        nPop = self.getParaValue('nPop')
        
        #Problem 
        self.problem = problem
        
        #Termination Condition Setting
        self.FEs = 0; self.iters = 0; self.tolerateTimes = 0
        
        #Population Generation
        pop=self.initialize(nPop)
            
        #Record result
        self.record(pop)
        
        #Init vel and orient
        pBestPop=pop #Personal best 
        gBestPop=pop[pop.argsort()[0]] #Global Best
        vel=pop.decs #Velocity
        
        while self.checkTermination():
            
            pop, vel=self._operationPSO(pop, vel, pBestPop, gBestPop, w, c1, c2)
            pop=self._randomParticle(pop)
            self.evaluate(pop)
            
            replace=np.where(pop.objs<pBestPop.objs)[0]
            pBestPop.replace(replace, pop[replace])
            gBestPop=pBestPop[pBestPop.argsort()[0]]
            
            self.record(pop)
            
        return self.result
    
    def _operationPSO(self, pop, vel, pBestPop, gBestPop, w, c1, c2):
        
        popDecs = pop.decs
        pBestDecs = pBestPop.decs
        gBestDecs = gBestPop.decs
        
        N, D = pop.size()
        
        particleVel = vel
        
        r1 = np.random.random((N, D))
        r2 = np.random.random((N, D))
        
        offVel = w*particleVel+(pBestDecs-popDecs)*c1*r1+(gBestDecs-popDecs)*c2*r2
        
        offspringDecs = popDecs + offVel
        np.clip(offspringDecs, self.problem.lb, self.problem.ub, out=offspringDecs)
        
        return Population(offspringDecs), offVel
    
    def _randomParticle(self, pop):
        
        popDecs = pop.decs
        N, D = pop.size()
        
        n_to_reinit = int(0.1 * N)
        rows_to_mutate = np.random.choice(N, size=n_to_reinit, replace=False)
        cols_to_mutate = np.random.choice(D, size=n_to_reinit, replace=False)

        offspringDecs = popDecs.copy()
        
        offspringDecs[rows_to_mutate, cols_to_mutate] = np.random.uniform(self.problem.lb[0, cols_to_mutate], self.problem.ub[0, cols_to_mutate], size=n_to_reinit)
        
        return Population(offspringDecs)