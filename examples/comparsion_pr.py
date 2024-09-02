import sys
sys.path.append(".")
from scipy.io import loadmat
import os
os.chdir('./examples')
import pandas as pd
database=pd.read_excel("./database_pr.xlsx", index_col=0)
#tmp
import numpy as np

seed_train=np.loadtxt("seed_train.txt", dtype=np.int64)
seed_test=np.loadtxt("seed_test.txt", dtype=np.int64)

from UQPyL.problems import (Sphere, Schwefel_2_22, Schwefel_1_22, Schwefel_2_21, Schwefel_2_26,
                            Rosenbrock, Step, Quartic, Rastrigin, Ackley, Griewank, Trid, Bent_Cigar,
                            Discus, Weierstrass)

benchmarks={1: Sphere, 2: Schwefel_2_22, 3: Schwefel_1_22, 4: Schwefel_2_21, 5: Schwefel_2_26,
            6: Rosenbrock, 7:Step, 8: Quartic, 9:Rastrigin, 10:Ackley, 11:Griewank, 12:Trid, 
            13:Bent_Cigar, 14: Discus, 15:Weierstrass}

from UQPyL.DoE import LHS
from UQPyL.utility.metrics import r_square, rank_score
from UQPyL.utility.scalers import MinMaxScaler
from UQPyL.utility.model_selections import RandSelect

dimensions=[10, 20, 30, 50]
samples=[100, 200, 300, 500]

from UQPyL.surrogates import RBF
from UQPyL.surrogates.rbf_kernels import Cubic, Gaussian, Linear, Multiquadric, Thin_plate_spline
from UQPyL.problems import Problem, ProblemABC
from UQPyL.optimization import GA
kernels=["Lasso","Origin", "Ridge"]

Bootstrap_num=5


index=0
class pr_optimization(ProblemABC):
    def __init__(self, model, train_x, train_y, test_x, test_y):
        self.model=model
        self.train_x=train_x
        self.train_y=train_y
        self.test_x=test_x
        self.test_y=test_y
        
        super().__init__(n_input=3, n_output=1, lb=np.array([0, 0, 0]), ub=np.array([2, 1, 0.0001]), disc_var=[1, 1, 0], disc_range=[[0, 1, 2], [0, 1], 0])

    def evaluate(self, X):
        n=X.shape[0]
        y=np.zeros((n,1))
        X=self.rescale_to_problems(X)
        for i in range(n):
            kernel=kernels[round(X[i, 0])]
            interface_bool=[round(X[i,1])]
            alpha=X[i,2]
            if kernel=="Origin":
                self.model=PolynomialRegression(alpha=alpha, degree=2, loss_type=kernel, interaction_only=interface_bool)
            else:
                self.model=PolynomialRegression(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)),alpha=alpha, degree=2, loss_type=kernel, interaction_only=interface_bool)
            # self.model.loss_type=kernel
            # self.model.alpha=alpha
            # self.model.interaction_only=interface_bool
            # try:
            self.model.fit(self.train_x, self.train_y)
            pre_=self.model.predict(self.test_x)
            y[i,0]=-1*rank_score(self.test_y, pre_)
            # except:
            #     y[i,0]=1
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
        tmp_X[:, pos_ind]=X[:, pos_ind]
        
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

from UQPyL.surrogates import PolynomialRegression
counts=[]
for i in range(1, 15):
    id=i
    func=benchmarks[i]
    for dim in dimensions:
        for sample in samples:
            for time in range(20):
                problem=func(dim)
                lhs=LHS('classic', problem=problem)
                train_X=lhs.sample(sample, problem.n_input, random_seed=seed_train[index, time])
                train_Y=problem.evaluate(train_X)
                
                test_X=lhs.sample(sample, problem.n_input, random_seed=seed_test[index, time])
                test_Y=problem.evaluate(test_X)
                
                scores=np.zeros((Bootstrap_num, 3))
                randSelect=RandSelect(int(sample/10))
                train, test=randSelect.split(train_X)
                train_X_=train_X[train, :]
                train_Y_=train_Y[train, :]
                test_X_=test_X[test, :]
                test_Y_=test_Y[test, :]
                surrogate=PolynomialRegression(alpha=0.00005, degree=2, loss_type=kernels[0])
                opt=pr_optimization(surrogate, train_X_, train_Y_, test_X_, test_Y_)
                ga=GA(problem=opt, n_samples=50, maxFEs=10000)
                res=ga.run()
                x=res["best_decs"][0]
                loss=round(x[0])
                inter=round(x[1])
                
                
                
                
                # for j in range(Bootstrap_num):
                #     train, test=randSelect.split(train_X)
                #     train_X_=train_X[train, :]
                #     train_Y_=train_Y[train, :]
                #     test_X_=test_X[test, :]
                #     test_Y_=test_Y[test, :]
                    
                #     for k in range(3):
                #         surrogate=PolynomialRegression(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)), alpha=0.00005, degree=2, loss_type=kernels[k])
                #         surrogate.fit(train_X_, train_Y_)
                #         pre_=surrogate.predict(test_X_)
                #         scores[j,k]=rank_score(test_Y_, pre_)
                        
                # total_rank=np.mean(scores, axis=0)
                # k=np.argmax(total_rank)
                if loss==1:
                    surrogate=PolynomialRegression(alpha=x[2], degree=2, loss_type=kernels[loss], interaction_only=inter)
                else:
                    surrogate=PolynomialRegression(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)), alpha=x[2], degree=2, loss_type=kernels[loss], interaction_only=inter)
                    
                    # self.model=PolynomialRegression(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)),alpha=alpha, degree=2, loss_type=kernel, interaction_only=interface_bool)
                # surrogate=PolynomialRegression(alpha=x[2], degree=2, loss_type=kernels[loss], interaction_only=inter)
                
                surrogate.fit(train_X, train_Y)
                pre_Y=surrogate.predict(test_X)
                r2=r_square(test_Y, pre_Y)
                rank=rank_score(test_Y, pre_Y)
                
                database.loc[len(database)]=[problem.__class__.__name__, 'PR-M2', index, time, dim, sample, r2, rank]
            index=index+1
database.to_excel("./database_pr.xlsx")       
# for i in range(1,15):
#     count={"Cubic": 0, "Gaussian": 0, "Linear": 0, "Multiquadric": 0, "Thin_plate_spline": 0 }
#     id=i
#     func=benchmarks[i]
#     for dim in dimensions:
#         for sample in samples:
#             for time in range(5):
#                 problem=func(dim)
#                 lhs=LHS('classic', problem=problem)
                
#                 train_X=lhs.sample(sample, problem.n_input, random_seed=seed_train[index, time])
#                 train_Y=problem.evaluate(train_X)
                
#                 test_X=lhs.sample(sample, problem.n_input, random_seed=seed_test[index, time])
#                 test_Y=problem.evaluate(test_X)
        
#                 randSelect=RandSelect(int(sample/10))
                
#                 train, test=randSelect.split(train_X)
#                 train_X_=train_X[train, :]
#                 train_Y_=train_Y[train, :]
#                 test_X_=test_X[test, :]
#                 test_Y_=test_Y[test, :]
                
#                 surrogate=RBF(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)), kernel=Cubic())
#                 rbf_opt=rbf_optimization(model=surrogate, train_x=train_X_, train_y=train_Y_, test_x=test_X_, test_y=test_Y_)
#                 ga=GA(problem=rbf_opt, n_samples=50, maxFEs=1000)
#                 res=ga.run()
                
#                 kernel=eval(kernels[int(res["best_decs"][0,0])])()
#                 surrogate=RBF(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)), kernel=kernel)
#                 surrogate.C_smooth=res["best_decs"][0,2]
#                 surrogate.kernel.epsilon=np.power(10, res["best_decs"][0,1])  
                
#                 surrogate.fit(train_X, train_Y)
#                 pre_Y=surrogate.predict(test_X)
#                 r2=r_square(test_Y, pre_Y)
#                 rank=rank_score(test_Y, pre_Y)
                
#                 database.loc[len(database)]=[problem.__class__.__name__, 'RBF-M2', index, time, dim, sample, r2, rank]
#             index=index+1
# database.to_excel("./database_pr.xlsx")
      



# counts=[]
# for i in range(1,15):
#     count={"Cubic": 0, "Gaussian": 0, "Linear": 0, "Multiquadric": 0, "Thin_plate_spline": 0 }
#     id=i
#     func=benchmarks[i]
#     for dim in dimensions:
#         for sample in samples:
#             for time in range(20):
#                 problem=func(dim)
#                 lhs=LHS('classic', problem=problem)
                
#                 train_X=lhs.sample(sample, problem.n_input, random_seed=seed_train[index, time])
#                 train_Y=problem.evaluate(train_X)
                
#                 test_X=lhs.sample(sample, problem.n_input, random_seed=seed_test[index, time])
#                 test_Y=problem.evaluate(test_X)
                
#                 scores=np.zeros((Bootstrap_num, 5))
#                 randSelect=RandSelect(int(sample/10))
#                 for j in range(Bootstrap_num):
#                     train, test=randSelect.split(train_X)
#                     train_X_=train_X[train, :]
#                     train_Y_=train_Y[train, :]
#                     test_X_=test_X[test, :]
#                     test_Y_=test_Y[test, :]
                    
#                     for k in range(5):
#                         if k<4:
#                             surrogate=RBF(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)),kernel=eval(kernels[k])())
#                         else:
#                             surrogate=RBF(kernel=eval(kernels[k])())
#                         surrogate.fit(train_X_, train_Y_)
#                         pre_=surrogate.predict(test_X_)
#                         scores[j,k]=rank_score(test_Y_, pre_)
                
#                 total_rank=np.mean(scores, axis=0)
#                 k=np.argmax(total_rank)
                
                
#                 if k<4:
#                     surrogate=RBF(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)),kernel=eval(kernels[k])())
#                 else:
#                     surrogate=RBF(kernel=eval(kernels[k])())
                
#                 surrogate.fit(train_X, train_Y)
#                 pre_Y=surrogate.predict(test_X)
#                 r2=r_square(test_Y, pre_Y)
#                 rank=rank_score(test_Y, pre_Y)
                
#                 count[kernels[k]]+=1
                

#                 database.loc[len(database)]=[problem.__class__.__name__, 'RBF-TPS', index, time, dim, sample, r2, rank]
#             index=index+1
#     counts.append(count)
# database.to_excel("./database_rbf.xlsx")
                


               