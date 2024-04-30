### Multi-Objective Adaptive Surrogate Modelling-based Optimization
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

from ..DoE import LHS
from ..problems import Problem
from ..surrogates import Mo_Surrogates
from .nsga_ii import NSGAII

lhs=LHS("center")
class MOASMO():
    '''
    Multi-Objective Adaptive Surrogate Modelling-based Optimization
    -----------------------------------------------------------------
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
        surrogates: Surrogates
            the surrogates you want to use, you should implement Mo_Surrogate class
        Pct: float, default=0.2
            the percentage of the population to be selected for infilling
        n_init: int, default=50
            the number of initial samples
        n_pop: int, default=100
            the number of population for evolution optimizer
        maxFEs: int, default=1000
            the maximum number of function evaluations
        maxIter: int, default=100
            the maximum number of iterations
        x_init: 2d-np.ndarray, default=None
            the initial input samples
        y_init: 2d-np.ndarray, default=None
            the initial output samples
        advance_infilling: bool, default=False
            the switch to use advanced infilling or not
            
    Methods:
        run()
            run the optimization
    
    References:
        [1] W. Gong et al., Multiobjective adaptive surrogate modeling-based optimization for parameter estimation of large, complex geophysical models, 
                            Water Resour. Res., vol. 52, no. 3, pp. 1984–2008, Mar. 2016, doi: 10.1002/2015WR018230.
    '''
    def __init__(self, problem: Problem, surrogates: Mo_Surrogates,
                 Pct: float=0.2, n_init: int=50, n_pop: int=100, 
                 maxFEs: int=1000, maxIter: int=100,
                 x_init: int=None, y_init: int=None,
                 advance_infilling=False):
        #problem setting
        self.evaluate=problem.evaluate
        self.lb=problem.lb; self.ub=problem.ub
        self.n_input=problem.n_input
        self.n_output=problem.n_output
        
        #algorithm setting
        self.surrogates=surrogates
        self.n_init=n_init
        self.x_init=x_init
        self.y_init=y_init
        self.Pct=Pct
        self.n_pop=n_pop
        self.advance_infilling=advance_infilling
        self.subProblem=Problem(self.surrogates.predict, self.n_input, self.n_output, self.ub, self.lb)
        
        #termination setting
        self.maxFEs=maxFEs
        self.maxIter=maxIter
    def run(self):
        
        maxFEs=self.maxFEs
        show_process=tqdm(total=maxFEs)
        pct=self.Pct  
        n_init=self.n_init
        n_infilling=int(np.floor(n_init*pct))
        ub=self.ub; lb=self.lb
        
        if self.x_init is None:
            self.x_init=(ub-lb)*lhs(self.n_init, self.n_input)+lb
            
        if self.y_init is None:
            self.y_init=self.evaluate(self.x_init)
        
        FE=n_init
        XPop=self.x_init
        YPop=self.y_init
        show_process.update(FE)
        while FE<maxFEs:
            #build surrogate
            self.surrogates.fit(XPop, YPop)
            
            nsga_ii=NSGAII(self.subProblem, self.n_pop)
            #main optimization
            Result=nsga_ii.run()
            BestX=Result['pareto_X']
            BestY=Result['pareto_Y']
            CrowdDis=Result['crowdDis']
            
            if self.advance_infilling==False:
                #Origin version
                if BestY.shape[0]>n_infilling:
                    idx=CrowdDis.argsort()[::-1][:n_infilling]
                    BestX=np.copy(BestX[idx])
                    BestY=np.copy(BestY[idx])
            else:
                #Advanced version Using crowding-based strategy
                if BestY.shape[0]>n_infilling:
                    
                    Known_FrontNo, _ =nsga_ii.NDSort(YPop, YPop.shape[0])
                    Unknown_FrontNo, _=nsga_ii.NDSort(BestY, BestY.shape[0])
                    Known_best_Y=YPop[np.where(Known_FrontNo==1)]
                    Unknown_best_Y=BestY[np.where(Unknown_FrontNo==1)]
                    Unknown_best_X=BestX[np.where(Unknown_FrontNo==1)]

                    added_points_Y=[]
                    added_points_X=[]
                    for _ in range(n_infilling):
                        
                        if len(added_points_Y)==0:
                            distances = cdist(Unknown_best_Y, Known_best_Y)
                        else:
                            distances = cdist(Unknown_best_Y, np.append(Known_best_Y, added_points_Y, axis=0))

                        max_distance_index = np.argmax(np.min(distances, axis=1))

                        added_point = Unknown_best_Y[max_distance_index]
                        added_points_Y.append(added_point)
                        added_points_X.append(Unknown_best_X[max_distance_index])
                        Known_best_Y = np.append(Known_best_Y, [added_point], axis=0)

                        Unknown_best_Y = np.delete(Unknown_best_Y, max_distance_index, axis=0)
                        Unknown_best_X = np.delete(Unknown_best_X, max_distance_index, axis=0)
                    BestX=np.copy(np.array(added_points_X))
                    BestY=np.copy(np.array(added_points_Y))
            
            FE+=BestX.shape[0]
            show_process.update(BestX.shape[0])
            
            BestY=self.evaluate(BestX)
            XPop=np.vstack((XPop,BestX))
            YPop=np.vstack((YPop,BestY))
        
        FrontNo, _ =nsga_ii.NDSort(YPop, YPop.shape[0])
        idx=np.where(FrontNo==1)
        ND_XPop=XPop[idx]
        ND_YPop=YPop[idx]
        
        return ND_XPop, ND_YPop
          
        
                
        
            
        
            
            
        
        
        