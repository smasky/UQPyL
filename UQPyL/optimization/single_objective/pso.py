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
    
    def __init__(self, nInit: int=50, nPop: int=50,
                    w: float=0.1, c1: float=0.5, c2: float=0.5,
                    maxIterTimes: int=1000,
                    maxFEs: int=50000,
                    maxTolerateTimes: int=1000, tolerate: float=1e-6,
                    verbose: bool=True, verboseFreq: int=100, logFlag: bool=False, saveFlag=False):
        
            super().__init__(maxFEs=maxFEs, maxIterTimes=maxIterTimes, 
                         maxTolerateTimes=maxTolerateTimes, tolerate=tolerate, 
                         verbose=verbose, verboseFreq=verboseFreq, logFlag=logFlag, saveFlag=saveFlag)
            
            #user-define setting
            self.setParameters('w', w)
            self.setParameters('c1', c1)
            self.setParameters('c2', c2)
            self.setParameters('nInit', nInit)
            self.setParameters('nPop', nPop)
                
    @Verbose.decoratorRun
    @Algorithm.initializeRun
    def run(self, problem, xInit=None, yInit=None):
        
        #Initialization
        #Parameter Setting
        w, c1, c2 = self.getParaValue('w', 'c1', 'c2')
        nInit, nPop = self.getParaValue('nInit', 'nPop')
        #Problem 
        self.problem=problem
        
        #Termination Condition Setting
        self.FEs=0; self.iters=0; self.tolerateTimes=0
        
        #Population Generation
        if xInit is not None:
            if yInit is not None:
                pop=Population(xInit, yInit)
            else:
                pop=Population(xInit)
                self.evaluate(pop)
        else:
            pop=self.initialize(nInit)
            
        pop=pop.getTop(nPop)
        
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
        
        n, d=pop.size()
        
        particleVel=vel
        
        r1=np.random.random((n, d))
        r2=np.random.random((n, d))
        
        offVel=w*particleVel+(pBestPop.decs-pop.decs)*c1*r1+(gBestPop.decs-pop.decs)*c2*r2
        
        offSpring=pop+offVel
        
        offSpring.clip(self.problem.lb, self.problem.ub)
        
        return offSpring, offVel
    
    def _randomParticle(self, pop):
        
        n, d=pop.size()
        n_to_reinit = int(0.1 * n)
        rows_to_mutate = np.random.choice(n, size=n_to_reinit, replace=False)
        cols_to_mutate = np.random.choice(d, size=n_to_reinit, replace=False)

        pop.decs[rows_to_mutate, cols_to_mutate] = np.random.uniform(self.problem.lb[0, cols_to_mutate], self.problem.ub[0, cols_to_mutate], size=n_to_reinit)
        
        return pop