### Multi-Objective Adaptive Surrogate Modelling-based Optimization
import numpy as np
from ..Experiment_Design import LHS
from ..Problems import Problem
from .nsga_ii import NSGAII
from tqdm import tqdm
from scipy.spatial.distance import cdist
lhs=LHS("center")
class MOASMO():
    def __init__(self, problem, surrogates,
                 Pct=0.2, NInit=50, NPop=100, XInit=None, YInit=None,
                 advance_infilling=False):
        self.evaluator=problem.evaluate
        self.lb=problem.lb; self.ub=problem.ub
        self.NInput=problem.dim
        self.NOutput=problem.NOutput
        
        self.surrogates=surrogates
        self.NInit=NInit
        self.XInit=XInit
        self.YInit=YInit
        self.Pct=Pct
        self.NPop=NPop
        self.advance_infilling=advance_infilling
        self.subProblem=Problem(self.surrogates.predict, self.NInput, self.NOutput, self.ub, self.lb)
        
    def run(self, maxFE=1000):
        
        show_process=tqdm(total=maxFE)
        pct=self.Pct  
        NInit=self.NInit
        N_Resample=int(np.floor(NInit*pct))
        ub=self.ub; lb=self.lb
        
        if self.XInit is None:
            self.XInit=(ub-lb)*lhs(self.NInit, self.NInput)+lb
            
        if self.YInit is None:
            self.YInit=self.evaluator(self.XInit)
        
        FE=NInit
        XPop=self.XInit
        YPop=self.YInit
        show_process.update(FE)
        while FE<maxFE:
            #build surrogate
            self.surrogates.fit(XPop, YPop)
            
            nsga_ii=NSGAII(self.subProblem, self.NPop)
            #main optimization
            BestX, BestY, FrontNo, CrowdDis=nsga_ii.run()

            if self.advance_infilling==False:
                #Origin version
                if BestY.shape[0]>N_Resample:
                    idx=CrowdDis.argsort()[::-1][:N_Resample]
                    BestX=np.copy(BestX[idx])
                    BestY=np.copy(BestY[idx])
            else:
                #Advanced version Using crowding-based strategy
                if BestY.shape[0]>N_Resample:
                    
                    Known_FrontNo, _ =nsga_ii.NDSort(YPop, YPop.shape[0])
                    Unknown_FrontNo, _=nsga_ii.NDSort(BestY, BestY.shape[0])
                    Known_best_Y=YPop[np.where(Known_FrontNo==1)]
                    Unknown_best_Y=BestY[np.where(Unknown_FrontNo==1)]
                    Unknown_best_X=BestX[np.where(Unknown_FrontNo==1)]

                    added_points_Y=[]
                    added_points_X=[]
                    for _ in range(N_Resample):
                        
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
            
            BestY=self.evaluator(BestX)
            XPop=np.vstack((XPop,BestX))
            YPop=np.vstack((YPop,BestY))
        
        FrontNo, _ =nsga_ii.NDSort(YPop, YPop.shape[0])
        idx=np.where(FrontNo==1)
        ND_XPop=XPop[idx]
        ND_YPop=YPop[idx]
        
        return ND_XPop, ND_YPop
          
        
                
        
            
        
            
            
        
        
        