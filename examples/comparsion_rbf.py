import sys
sys.path.append(".")
from scipy.io import loadmat
import os
os.chdir('./examples')
import pandas as pd
database=pd.read_excel("./database_rbf.xlsx", index_col=0)
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

dimensions=[10, 20, 30, 50]
samples=[100, 200, 300, 500]

from UQPyL.surrogates import RBF
from UQPyL.surrogates.rbf.kernels import Cubic, Gaussian, Linear, Multiquadric, Thin_plate_spline

kernels=["Cubic", "Gaussian", "Linear", "Multiquadric", "Thin_plate_spline"]

index=0
for i in range(1,2):
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
                
                surrogate=RBF(scalers=(MinMaxScaler(0,0.1),MinMaxScaler(0,0.1)),kernel=Multiquadric())
                surrogate.fit(train_X, train_Y)
                pre_Y=surrogate.predict(test_X)
                r2=r_square(test_Y, pre_Y)
                rank=rank_score(test_Y, pre_Y)
                
                database.loc[len(database)]=[problem.__class__.__name__, 'RBF-MLP', index, time, dim, sample, r2, rank]
database.to_excel("./database_rbf.xlsx")               
                
                