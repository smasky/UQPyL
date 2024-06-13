import pandas as pd
import os
import numpy as np
os.chdir('./examples')
benchmarks={ 1: "Sphere", 2: "Schwefel_2_22", 3: "Schwefel_1_22", 4: "Schwefel_2_21", 5: "Schwefel_2_26",
             6: "Rosenbrock", 7: "Step", 8: "Quartic", 9: "Rastrigin", 10: "Ackley", 11: "Griewank", 12: "Trid", 
             13: "Bent_Cigar", 14: "Discus", 15: "Weierstrass" }
surrogate=["RBF-CB", "RBF-LN", "RBF-TPS"]
database = pd.read_excel("./database.xlsx", index_col=0)
dimensions=[5, 15, 30, 50]
samples=[100, 300, 500]

for sur in surrogate:
    for key, func in benchmarks.items():
            res=database[(database['surrogate'] == sur) & (database['problem'] == func)]['rank_score'].to_numpy()
            print(sur, func, "mean:", np.mean(res), "std:", np.std(res))
            