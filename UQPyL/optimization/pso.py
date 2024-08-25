import numpy as np
from tqdm import tqdm

from ..problems import Problem
from ..DoE import LHS

class PSO():
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
    def __init__(self, problem: Problem, n_samples: int=50,
                    w: float=0.1, c1: float=0.5, c2: float=0.5,
                    x_init=None, y_init=None,
                    maxIterTimes: int=1000,
                    maxFEs: int=50000,
                    maxTolerateTimes: int=1000,
                    tolerate=1e-6):
            #problem setting
            self.evaluate=problem.evaluate
            self.n_input=problem.n_input
            self.ub=problem.ub.reshape(1,-1);self.lb=problem.lb.reshape(1,-1)
            self.problem=problem
            
            #algorithm setting
            self.w=w;self.c1=c1;self.c2=c2
            self.n_samples=n_samples
            
            #
            self.x_init=x_init
            self.y_init=y_init
            
            #termination setting
            self.maxIterTimes=maxIterTimes
            self.maxFEs=maxFEs
            self.maxTolerateTimes=maxTolerateTimes
            self.tolerate=tolerate
    
    def run(self) -> dict:
        '''
            Run the Particle Swarm Optimization
            -------------------------------
            Returns:
                Result: dict
                    the result of the Particle Swarm Optimization, including the following keys:
                    decs: np.ndarray
                        the best decision of the Particle Swarm Optimization
                    objs: np.ndarray
                        the best objective value of the Particle Swarm Optimization
                    history_decs: np.ndarray
                        the history of the decision of the Particle Swarm Optimization
                    history_objs: np.ndarray
                        the history of the objective value of the Particle Swarm Optimization
                    iters: int
                        the iteration times of the Particle Swarm Optimization
                    FEs: int
                        the function evaluations of the Particle Swarm Optimization
        '''
        
        time=1
        iter=0
        FEs=0
        
        lhs=LHS('classic', problem=self.problem)
        if self.x_init is None:
            self.x_init=lhs.sample(self.n_samples, self.n_input)
        if self.y_init is None:
            self.y_init=self.evaluate(self.x_init)
        
        decs=self.x_init
        objs=self.y_init
        
        FEs+=objs.shape[0]
        
        history_best_decs={}
        history_best_objs={}
        Result={}
        
        P_best_decs=np.copy(decs)
        P_best_objs=np.copy(objs)
        ind=np.argmin(P_best_objs)
        G_best_dec=np.copy(P_best_decs[ind])      
        G_best_obj=np.copy(P_best_objs[ind])
        vel=np.copy(decs)
        
        # show_process=tqdm(total=self.maxIterTimes, desc="Particle Swarm Optimization")
        
        while iter<self.maxIterTimes and FEs<self.maxFEs and time<=self.maxTolerateTimes:
            decs, vel=self._operationPSO(decs, vel, P_best_decs, G_best_dec, self.w)
            decs=self._randomParticle(decs)
            objs=self.evaluate(decs)
            
            replace=np.where(objs<P_best_objs)[0]
            P_best_decs[replace]=np.copy(decs[replace])
            P_best_objs[replace]=np.copy(objs[replace])
            
            ind=np.argmin(P_best_objs)
            G_best_dec=np.copy(P_best_decs[ind])      
            G_best_obj=np.copy(P_best_objs[ind])
            
            iter+=1
            FEs+=objs.shape[0]
            
            # show_process.update(1)
            
            history_best_decs[FEs]=G_best_dec
            history_best_objs[FEs]=G_best_obj
            
        Result['best_decs']=G_best_dec
        Result['best_obj']=G_best_obj[0]
        Result['history_best_decs']=history_best_decs
        Result['history_best_objs']=history_best_objs
        Result['iters']=iter
        Result['FEs']=FEs
        
        return Result
            
    def _operationPSO(self, decs, vel, P_best_decs, G_best_dec, w):
        
        N, D=decs.shape
        
        PatricleVel=vel
        
        r1=np.random.rand(N,D)
        r2=np.random.rand(N,D)
        
        offVel=w*PatricleVel+self.c1*r1*(P_best_decs-decs)+self.c2*r2*(G_best_dec-decs)
        offDecs=decs+offVel
        
        offDecs = np.clip(offDecs, self.lb, self.ub)
        return offDecs, offVel
    
    
    def _randomParticle(self, decs):
        
        n_to_reinit = int(0.1 * decs.shape[0])
        rows_to_mutate = np.random.choice(decs.shape[0], size=n_to_reinit, replace=False)
        cols_to_mutate = np.random.choice(decs.shape[1], size=n_to_reinit, replace=False)

        decs[rows_to_mutate, cols_to_mutate] = np.random.uniform(self.lb[0, cols_to_mutate], self.ub[0, cols_to_mutate], size=n_to_reinit)
                
        return decs