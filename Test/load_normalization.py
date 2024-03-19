import pickle
import os
os.chdir('./Test')
import numpy as np
with open('./total_RBF.pickle','rb') as f:
    re1=pickle.load(f)
    re2=pickle.load(f)

# with open('./total_SVR.pickle','rb') as f:
#     re3=pickle.load(f)
#     re4=pickle.load(f)

# re5={**re1,**re3}
# re6={**re2, **re4}
# with open('./total_KRG_GA.pickle','rb') as f:
#     re3=pickle.load(f)
#     re4=pickle.load(f)
# with open('./normalization_SVR_minmax.pickle','rb') as f:
#     re3=pickle.load(f)
#     re4=pickle.load(f)
    
# with open('./normalization_SVR_stand.pickle','rb') as f:
#     re5=pickle.load(f)
#     re6=pickle.load(f)
funcs=["Sphere", "Schwefel_2_22", "Schwefel_1_22", "Schwefel_2_21", "Rosenbrock", 
                         "Step", "Quartic", "Schwefel_2_26", "Rastrigin", "Ackley", "Griewank", 
                         "Trid", "Bent_Cigar", "Discus", "Weierstrass"]

Sampling={}
n_dims=[5, 15, 20, 30]
n_samples=[50, 150, 300, 400, 500]
Types=['CB', 'MQ', 'LN', 'TPS', 'GAS']
# Types=['Origin', 'Ridge', 'Lasso']
# Types=['rbf', 'poly', 'linear']
# Types=['Linear']
# Types=['CB', 'MQ', 'LN', 'TPS', 'GAS']
# Types=['linear', 'rbf', 'polynomial']
# Types=["Origin", "Ridge", "Lasso"]
# Types=[0.5,1.5,2.5,np.inf]
RE_R2=np.ones((15,4))*-100
RE_Rank=np.ones((15,4))*-100
RE_Type=[]
I=0
J=0
Samplings={5:150, 15:300, 30:500}
for i, func in enumerate(funcs):
    S_types=[]
    for j, D in enumerate(n_dims):
        total_R2=[]
        total_Rank=[]
        for kernel in Types:
            name="{}_D{}_N{}_{}".format(func, D, 10*D, kernel)
            Sub_R2=re1[name]
            Sub_Rank=re2[name]
            total_R2+=Sub_R2
            total_Rank+=Sub_Rank
        S_list=np.argsort(np.array(total_R2)*-1)[:20].tolist()
        S_types+=[Types[value//20] for value in S_list]
        
        S_R2_value=np.sort(total_R2)[::-1][:20]
        S_RP_Value=np.sort(total_Rank)[::-1][:20]
        RE_R2[i, j]=np.mean(S_R2_value)
        RE_Rank[i, j]=np.mean(S_RP_Value)
    RE_Type.append(S_types)
# for func in funcs:
#     for D in n_dims:
#         I=0
#         for kernel in Types:   
#             name="{}_D{}_N{}_{}".format(func, D, Samplings[D],kernel)
#             A=np.array(re1[name]).mean()
#             # B=np.array(re3[name]).mean()
#             # C=np.array(re5[name]).mean()
#             RE[J,3*I]=A
#             # RE[J,3*I+1]=B
#             # RE[J,3*I+2]=C
#             I+=1
#         J+=1
RE_count=[]
for types in RE_Type:
    CB=0
    MQ=0
    LN=0
    TPS=0
    GAS=0
    for t in types:
        if t=='CB':
            CB+=1
        elif t=='MQ':
            MQ+=1
        elif t=='LN':
            LN+=1
        elif t=='TPS':
            TPS+=1
        elif t=='GAS':
            GAS+=1
    RE_count.append([CB, MQ, LN, TPS, GAS]) #['CB', 'MQ', 'LN', 'TPS', 'GAS']    
RE=np.array(RE_count)
np.savetxt('RBF_count.txt', RE)