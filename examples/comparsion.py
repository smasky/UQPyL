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

index=0
for id, func in benchmarks.items():
    for dim in dimensions:
        for sample in samples:
            for time in range(20):
                problem=func(dim)
                lhs=LHS('classic', problem=problem)
                
                train_X=lhs.sample(sample, problem.n_input, random_seed=seed_train[index, time])
                train_Y=problem.evaluate(train_X)
                
                test_X=lhs.sample(sample, problem.n_input, random_seed=seed_test[index, time])
                test_Y=problem.evaluate(test_X)

                surrogate=RBF(kernel=Multiquadric())
                surrogate.fit(train_X, train_Y)
                pre_Y=surrogate.predict(test_X)
                r2=r_square(test_Y, pre_Y)
                rank=rank_score(test_Y, pre_Y)
                
                database.loc[len(database)]=[problem.__class__.__name__, 'RBF-MLP', index, time,dim, sample, r2, rank]
            index+=1
database.to_excel("./database.xlsx")
            






