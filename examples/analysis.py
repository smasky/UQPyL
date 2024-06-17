import pandas as pd
import os
import numpy as np
from scipy.stats import kruskal, mannwhitneyu
os.chdir('./examples')
benchmarks={ 1: "Sphere", 2: "Schwefel_2_22", 3: "Schwefel_1_22", 4: "Schwefel_2_21", 5: "Schwefel_2_26",
             6: "Rosenbrock", 7: "Step", 8: "Quartic", 9: "Rastrigin", 10: "Ackley", 11: "Griewank", 12: "Trid", 
             13: "Bent_Cigar", 14: "Discus", 15: "Weierstrass" }
surrogate=["RBF-CB", "RBF-LN", "RBF-TPS", "RBF-MLP", "RBF-GAS", "RBF-L1"]
database = pd.read_excel("./database.xlsx", index_col=0)
dimensions=[10, 20, 30, 50]
samples=[100, 200, 300, 500]

for sur in surrogate:
    for dim in dimensions:
        for sample in samples:
            res=database[(database['surrogate'] == sur) & (database['dimensions'] == dim) & (database['samples'] == sample)]['rank_score'].to_numpy()
            # res[res<0]=0
            # count=np.sum(res==0)
            print(sur, "dim:", dim,"sample:", sample,"mean:", np.mean(res), "std:", np.std(res))

# for key, problem in benchmarks.items():
#     score_2={"RBF-CB": 0, "RBF-LN": 0, "RBF-TPS": 0, "RBF-MLP": 0, "RBF-GAS": 0, "RBF-L1": 0}
#     score_1={"RBF-CB": 0, "RBF-LN": 0, "RBF-TPS": 0, "RBF-MLP": 0, "RBF-GAS": 0, "RBF-L1": 0}
#     for dim in dimensions:
#         for sample in samples:
#             first=database[(database['surrogate'] == surrogate[0]) & (database['dimensions'] == dim) & (database['samples'] == sample) &(database['problem']==problem)]['r_square'].to_numpy()
#             name=[surrogate[0]]
#             for i in range(1, len(surrogate)):
#                 second=database[(database['surrogate'] == surrogate[i]) & (database['dimensions'] == dim) & (database['samples'] == sample) &(database['problem']==problem)]['r_square'].to_numpy()
#                 stat, p_value = mannwhitneyu(first, second, alternative='two-sided')
#                 if p_value > 0.05:
#                     name.append(surrogate[i])
#                 elif np.mean(second)>np.mean(first):
#                     name=[surrogate[i]]
#                     first=second
#             if(len(name)==1):
#                 score_1[name[0]]+=1
#             else:
#                 for n in name:
#                     score_2[n]+=1
#     print(problem)
#     print(score_1)
#     print(score_2)
            