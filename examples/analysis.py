import pandas as pd
import os
import numpy as np
from scipy.stats import kruskal, mannwhitneyu
os.chdir('./examples')
benchmarks={ 1: "Sphere", 2: "Schwefel_2_22", 3: "Schwefel_1_22", 4: "Schwefel_2_21", 5: "Schwefel_2_26",
             6: "Rosenbrock", 7: "Step", 8: "Quartic", 9: "Rastrigin", 10: "Ackley", 11: "Griewank", 12: "Trid", 
             13: "Bent_Cigar", 14: "Discus"}
# surrogate=["RBF-CB", "RBF-LN", "RBF-TPS", "RBF-MLP", "RBF-GAS", "RBF-M1"]
# surrogate=["GP-Matern-0.5", "GP-Matern-1.5", "GP-Matern-2.5", "GP-Exp", "GP-RQ", "GP-L1"]
# surrogate=["SVR-rbf", "SVR-linear", "SVR-poly"]
surrogate=["MARS-M0"]
database = pd.read_excel("./database_mar.xlsx", index_col=0)
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
#     for sur in surrogate:
#         for dim in dimensions:
#             for sample in samples:
#                 res=database[(database['surrogate'] == sur) & (database['dimensions'] == dim) & (database['samples'] == sample) &(database['problem']==problem)]['rank_score'].to_numpy()
#                 # res[res<0]=0
#                 # count=np.sum(res==0)
#                 print(sur, "dim:", dim,"sample:", sample,"problem:", problem ,"mean:", np.mean(res), "std:", np.std(res))

# score_2={"GP-Matern-0.5": 0, "GP-Matern-1.5": 0, "GP-Matern-2.5": 0, "GP-Exp": 0, "GP-RQ": 0, "GP-L1":0 }
# score_1={"GP-Matern-0.5": 0, "GP-Matern-1.5": 0, "GP-Matern-2.5": 0, "GP-Exp": 0, "GP-RQ": 0, "GP-L1":0}

signs=0
counts=np.zeros(3)
for key, problem in benchmarks.items():
    # score_2={"RBF-CB": 0, "RBF-LN": 0, "RBF-TPS": 0, "RBF-MLP": 0, "RBF-GAS": 0, "RBF-L1": 0}
    # score_1={"RBF-CB": 0, "RBF-LN": 0, "RBF-TPS": 0, "RBF-MLP": 0, "RBF-GAS": 0, "RBF-L1": 0}
    # score_2={"RBF-CB": 0, "RBF-LN": 0, "RBF-TPS": 0, "RBF-MLP": 0, "RBF-GAS": 0, "RBF-M1": 0}
    # score_1={"RBF-CB": 0, "RBF-LN": 0, "RBF-TPS": 0, "RBF-MLP": 0, "RBF-GAS": 0, "RBF-M1": 0}
    signs=0
    for dim in dimensions:
        for sample in samples:
            data=[]
            for i in range(0, len(surrogate)):
                tmp=database[(database['surrogate'] == surrogate[i]) & (database['dimensions'] == dim) & (database['samples'] == sample) &(database['problem']==problem)]['rank_score'].to_numpy()
                data.append(database[(database['surrogate'] == surrogate[i]) & (database['dimensions'] == dim) & (database['samples'] == sample) &(database['problem']==problem)]['rank_score'].to_numpy())
            data_array=np.column_stack(data)
            max_indexs=np.argmax(data_array, axis=1)
            max_counts = np.bincount(max_indexs, minlength=data_array.shape[1])
            counts+=np.array(max_counts)
            
            # sign=np.zeros(20)
            # max_values=np.max(data_array, axis=1)
            # M1=database[(database['surrogate'] == surrogate[-1]) & (database['dimensions'] == dim) & (database['samples'] == sample) &(database['problem']==problem)]['rank_score'].to_numpy()
            # sign[M1>=max_values]=1
            # signs+=np.sum(sign)
            # data=[]
            # first=database[(database['surrogate'] == surrogate[0]) & (database['dimensions'] == dim) & (database['samples'] == sample) &(database['problem']==problem)]['r_square'].to_numpy()
            # name=[surrogate[0]]
            # for i in range(1, len(surrogate)):
            #     second=database[(database['surrogate'] == surrogate[i]) & (database['dimensions'] == dim) & (database['samples'] == sample) &(database['problem']==problem)]['r_square'].to_numpy()
            #     # stat, p_value = mannwhitneyu(first, second, alternative='two-sided')
            #     # if p_value > 0.05:
            #     #     name.append(surrogate[i])
            #     # elif np.mean(second)>np.mean(first):
            #     #     name=[surrogate[i]]
            #     #     first=second
            #     if np.mean(second)>np.mean(first):
            #         name=[surrogate[i]]
            #         first=second
            # if(len(name)==1):
            #     score_1[name[0]]+=1
            # else:
            #     for n in name:
            #         score_2[n]+=1
    # print(problem)
    # print(score_1)
    # print(score_2)
    print(problem, counts)
    print(problem, signs)      