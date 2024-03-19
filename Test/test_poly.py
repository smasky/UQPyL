import sys
sys.path.append(".")
from UQPyL.Problems.Benchmarks import (Sphere, Schwefel_2_22, Schwefel_1_22, Schwefel_2_21, Rosenbrock, 
                         Step, Quartic, Schwefel_2_26, Rastrigin, Ackley, Griewank, 
                         Trid, Bent_Cigar, Discus, Weierstrass)
from UQPyL.Surrogates import LinearRegression as LR, SVR, RBF
from UQPyL.Surrogates.RBF_Kernel import Cubic, Multiquadric, Linear, Thin_plate_spline, Gaussian
from UQPyL.Utility.scalers import MinMaxScaler, StandardScaler
from UQPyL.Utility.metrics import r2_score, rank_score, sort_score
from UQPyL.Utility.polynomial_features import PolynomialFeatures

import pickle
import os
os.chdir('./Test')
with open('./Samples.pickle','rb') as f:
    datas=pickle.load(f)
with open('./Samples_test.pickle','rb') as f:
    test_datas=pickle.load(f)
    
funcs=["Sphere", "Schwefel_2_22", "Schwefel_1_22", "Schwefel_2_21", "Rosenbrock",
       "Step", "Quartic", "Schwefel_2_26", "Rastrigin", "Ackley", "Griewank", 
       "Trid", "Bent_Cigar", "Discus", "Weierstrass" ]

Sampling={}
n_dims=[5, 10, 15, 20, 25, 30]
n_samples=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
n_times=20

def compute_Y(func, dim, samples):
    F=eval(func)(dim=dim)
    Y=F.evaluate(samples)
    return Y

r2_result={}
rank_result={}

kernels=[Cubic(), Multiquadric(), Linear(), Thin_plate_spline(), Gaussian()]
poly=PolynomialFeatures(degree=2)
for kernel in kernels:
    for func in funcs[:]:
        for D in n_dims:
            for N in n_samples:
                name="{}_D{}_N{}".format(func, D, N)
                name2="{}_D{}_N{}".format(func, D, 50)
                re1=[]
                re2=[]
                re3=[]
                for I in range(n_times):
                    train_X=datas[name][I]
                    train_Y=compute_Y(func, D, train_X)
                    
                    test_X=test_datas[name2][I]
                    test_Y=compute_Y(func, D, test_X)
                    # model=LR(scalers=(MinMaxScaler(0, 1000), MinMaxScaler(0, 1000)), poly_feature=PolynomialFeatures(degree=2), alpha=0.1,Type='Lasso')
                    model=SVR(scalers=(MinMaxScaler(-10, 10), MinMaxScaler(-10, 10)), poly_feature=PolynomialFeatures(degree=2), kernel='linear')
                    #model=SVR(kernel='linear')
                    # model=RBF(poly_feature=poly,kernel=kernel)
                    model.fit(train_X, train_Y)
                    pre_Y=model.predict(test_X)
                    re1.append(r2_score(test_Y, pre_Y))
                    re2.append(rank_score(test_Y, pre_Y))
                r2_result[name]=re1
                rank_result[name]=re2
                
with open('./SVR_result_poly.pickle','wb') as f:
    pickle.dump(r2_result, f)
    pickle.dump(rank_result, f)
    