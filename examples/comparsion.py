import sys
sys.path.append(".")
from scipy.io import loadmat
import os
os.chdir('./examples')
import pandas as pd
#tmp
import numpy as np
# seed_train=np.random.randint(0, 1000, (450,20))
# seed_test=np.random.randint(0, 1000, (450,20))
# np.savetxt("seed_train.txt", seed_train, fmt="%d")
# np.savetxt("seed_test.txt", seed_test, fmt="%d")
import numpy as np
seed_train=np.loadtxt("seed_train.txt", dtype=np.int64)
seed_test=np.loadtxt("seed_test.txt", dtype=np.int64)
from UQPyL.problems import (Sphere, Schwefel_2_22, Schwefel_1_22, Schwefel_2_21, Schwefel_2_26,
                            Rosenbrock, Step, Quartic, Rastrigin, Ackley, Griewank, Trid, Bent_Cigar,
                            Discus, Weierstrass)
from UQPyL.DoE import LHS
from UQPyL.utility.scalers import MinMaxScaler
from UQPyL.utility.metrics import r_square, rank_score
benchmarks={1: Sphere, 2: Schwefel_2_22, 3: Schwefel_1_22, 4: Schwefel_2_21, 5: Schwefel_2_26,
            6: Rosenbrock, 7:Step, 8: Quartic, 9:Rastrigin, 10:Ackley, 11:Griewank, 12:Trid, 
            13:Bent_Cigar, 14: Discus, 15:Weierstrass}

dimensions=[10, 20, 30, 50]
samples=[100, 200, 300, 500]
# columns = ['problem', 'surrogate', 'dimensions', 'samples', 'r_square', 'rank_score']
database = pd.read_excel("./database.xlsx", index_col=0)
#----------------------RBF-----------------------------#
from UQPyL.surrogates import RBF
from UQPyL.surrogates.rbf_kernels import Cubic, Gaussian, Linear, Multiquadric, Thin_plate_spline
from UQPyL.optimization import PSO, GA
from UQPyL.problems import Problem, ProblemABC
kernels=["Cubic", "Gaussian", "Linear", "Multiquadric", "Thin_plate_spline"]
index=0

class rbf_optimization(ProblemABC):
    def __init__(self, model, train_x, train_y, test_x, test_y):
        self.model=model
        self.train_x=train_x
        self.train_y=train_y
        self.test_x=test_x
        self.test_y=test_y
        
        super().__init__(n_input=2, n_output=1, lb=np.array([0, 0]), ub=np.array([4, 0.5]), disc_var=[1, 0], disc_range=[[0, 1, 2, 4], 0])
    
    def evaluate(self, X):
        n=X.shape[0]
        y=np.zeros((n,1))
        X=self.rescale_to_problems(X)
        for i in range(n):
            kernel=kernels[int(X[i, 0])]
            c_smooth=X[i, 1]
            self.model.kernel=eval(kernel)()
            self.model.C_smooth=c_smooth
            self.model.fit(self.train_x, self.train_y)
            pre_=self.model.predict(self.test_x)
            y[i,0]=-1*rank_score(self.test_y, pre_)
        return y
    
    def rescale_to_problems(self, X:np.ndarray):
        
        disc_var=self.disc_var
        disc_range=self.disc_range
        lb=self.lb
        ub=self.ub
        
        tmp_X=np.empty_like(X)
        # for continuous variables
        if isinstance(disc_var, list):
            disc_var=np.array(disc_var)
        pos_ind=np.where(disc_var==0)[0]
        tmp_X[:, pos_ind]=X[:, pos_ind]*(ub[:, pos_ind]-lb[:, pos_ind])+lb[:, pos_ind]
        
        # for disc variables
        for i in range(self.n_input):
            if disc_var[i]==1:
                num_interval=len(disc_range[i])
                bins=np.linspace(0, 1, num_interval+1)
                indices = np.digitize(X[:, i]/(self.ub[0, i]-self.lb[0, i])+self.lb[0, i], bins[1:])
                indices[indices >= num_interval] = num_interval - 1
                if isinstance(disc_range[i], list):
                    tmp_X[:, i]=np.array(disc_range[i])[indices]
                else:
                    tmp_X[:, i]=disc_range[i][indices]
        return tmp_X
    
       
for id, func in benchmarks.items():
    for dim in dimensions:
        for sample in samples:
            for time in range(20):
                problem=func(dim)
                lhs=LHS('classic', problem=problem)
                
                train_X=lhs.sample(sample, problem.n_input, random_seed=seed_train[index, time])
                train_Y=problem.evaluate(train_X)
                ################select kernel func##################
                indices = np.arange(train_X.shape[0])
                selected_indices = np.random.choice(indices, size=int(train_X.shape[0]/10), replace=False)
                unselected_indices = np.setdiff1d(indices, selected_indices)
                train_X_=train_X[unselected_indices]
                train_Y_=train_Y[unselected_indices]
                test_X_=train_X[selected_indices]
                test_Y_=train_Y[selected_indices]
                
                ga=GA(problem=rbf_optimization(RBF(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)), kernel=Cubic()), train_x=train_X_, train_y=train_Y_, test_x=test_X_, test_y=test_Y_), n_samples=50, maxFEs=1000)
                res=ga.run()
                kernel=eval(kernels[int(res["best_decs"][0,0])])()
                surrogate=RBF(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)), kernel=kernel)
                surrogate.C_smooth=res["best_decs"][0,1]
                
                test_X=lhs.sample(sample, problem.n_input, random_seed=seed_test[index, time])
                test_Y=problem.evaluate(test_X)

                surrogate.fit(train_X, train_Y)
                pre_Y=surrogate.predict(test_X)
                r2=r_square(test_Y, pre_Y)
                rank=rank_score(test_Y, pre_Y)
                
                database.loc[len(database)]=[problem.__class__.__name__, 'RBF-L2', index, time,dim, sample, r2, rank]
            index+=1
database.to_excel("./database.xlsx")
            






