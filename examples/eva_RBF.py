#evaluate RBF

import sys
sys.path.append(".")
from scipy.io import loadmat
import os
os.chdir('./examples')

import pandas as pd
import numpy as np

from UQPyL.DoE import LHS

from UQPyL.problems import (Sphere, Schwefel_2_22, Schwefel_1_22, Schwefel_2_21, Rosenbrock,
                            Step, Quartic, Schwefel_2_26, Rastrigin, Ackley, Griewank, 
                            Trid, Bent_Cigar, Discus, Weierstrass)
                         
from UQPyL.utility.metrics import r_square, rank_score

from UQPyL.surrogates import RBF
from UQPyL.surrogates.rbf_kernels import Cubic, Gaussian, Linear, Multiquadric, Thin_plate_spline

benchmark_list=[Sphere, Schwefel_2_22, Schwefel_1_22, Schwefel_2_21, Rosenbrock,
                Step, Quartic, Schwefel_2_26, Rastrigin, Ackley, Griewank, 
                Trid, Bent_Cigar, Discus, Weierstrass]

times=10
dimensions=[5, 10, 15, 20, 25, 30]
n_samples=[100, 150, 200, 250, 300, 350, 400, 450, 500]
database=pd.DataFrame(columns=['benchmark','dimensions','sampling_size','surrogate_name', 'kernel', 'random_seed1', 'random_seed2', 'r2_score', 'rank_score'])


from tqdm import tqdm

for benchmark in tqdm(benchmark_list):
    for n_sample in n_samples:
        for d in dimensions:
            for i in range(times):
                
                seed1=np.random.randint(0, 1000000)
                seed2=np.random.randint(0, 1000000)
                
                problem=benchmark(n_input=d)
                lhs=LHS(criterion='classic', problem=problem)

                kernel=Cubic()
                rbf=RBF(kernel=kernel)

                train_X=lhs.sample(n_sample, d, random_seed=seed1)
                train_Y=problem.evaluate(train_X)
                test_X=lhs.sample(int(n_sample/10), d, random_seed=seed2)
                test_Y=problem.evaluate(test_X)

                rbf.fit(train_X, train_Y)
                predict_Y=rbf.predict(test_X)

                r2_score=r_square(test_Y, predict_Y)
                rank_error=rank_score(test_Y, predict_Y)
                
                new_row={'benchmark':problem.__class__.__name__, 'dimensions': d, 'sampling_size': n_sample, 'surrogate_name': rbf.__class__.__name__, 'kernel': kernel.__class__.__name__, 'random_seed1': seed1, 'random_seed2': seed2, 'r2_score': r2_score, 'rank_score': rank_error}
                database.loc[len(database)]=new_row
database.to_excel('RBF_database.xlsx', index=False)
                



