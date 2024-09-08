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

dimensions=[5, 10, 20, 30, 40, 50]
samples=[100, 200, 300, 400, 500]
# columns = ['problem', 'surrogate', 'dimensions', 'samples', 'r_square', 'rank_score']
database = pd.read_excel("./database.xlsx", index_col=0)
#----------------------RBF-----------------------------#
from UQPyL.surrogates import RBF
from UQPyL.surrogates.rbf.kernels import Cubic, Gaussian, Linear, Multiquadric, Thin_plate_spline
from UQPyL.utility import MinMaxScaler
index=0
dim=50
sample=100
time=2
index=7
   
problem=Schwefel_2_22(dim)
lhs=LHS('classic', problem=problem)

train_X=lhs.sample(sample, problem.n_input, random_seed=seed_train[index, time])
train_Y=problem.evaluate(train_X)

test_X=lhs.sample(50, problem.n_input, random_seed=seed_test[index, time])
test_Y=problem.evaluate(test_X)

surrogate=RBF(kernel=Cubic())
surrogate.fit(train_X, train_Y)
pre_Y=surrogate.predict(test_X)
r2=r_square(test_Y, pre_Y)
rank=rank_score(test_Y, pre_Y)
print(r2, rank)

            






