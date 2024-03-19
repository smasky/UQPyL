import sys
sys.path.append(".")
from UQPyL.Experiment_Design import LHS
from UQPyL.Problems.Benchmarks import (Sphere, Schwefel_2_22, Schwefel_1_22, Schwefel_2_21, Rosenbrock, 
                         Step, Quartic, Schwefel_2_26, Rastrigin, Ackley, Griewank, 
                         Trid, Bent_Cigar, Discus, Weierstrass)
import os
os.chdir('./Test')
import numpy as np
import pickle
funcs=["Sphere", "Schwefel_2_22", "Schwefel_1_22", "Schwefel_2_21", "Rosenbrock",
       "Step", "Quartic", "Schwefel_2_26", "Rastrigin", "Ackley", "Griewank", 
       "Trid", "Bent_Cigar", "Discus", "Weierstrass" ]

lhs=LHS(criterion='classic')
Sampling={}
n_dims=[5, 10, 15, 20, 25, 30]
n_samples=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
n_samples_test=[50]*10
n_times=50

for func in funcs:
    for D in n_dims:
        for N in n_samples_test:
            name="{}_D{}_N{}".format(func, D, N)
            sample=[] 
            for _ in range(n_times):
                sample.append(lhs._generate(N, D))
            Sampling[name]=sample

with open('Samples_test.pickle', 'wb') as f:
    pickle.dump(Sampling,f)
            