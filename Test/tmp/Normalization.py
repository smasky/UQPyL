import sys
sys.path.append(".")
from UQPyL.Problems.Single_Benchmarks import (Sphere, Schwefel_2_22, Schwefel_1_22, Schwefel_2_21, Rosenbrock, 
                         Step, Quartic, Schwefel_2_26, Rastrigin, Ackley, Griewank, 
                         Trid, Bent_Cigar, Discus, Weierstrass)
from UQPyL.Surrogates import LinearRegression as LR, SVR, RBF, PolynomialRegression as PR, KRG, GPR
from UQPyL.Surrogates.RBF_Kernel import Cubic, Multiquadric, Linear, Thin_plate_spline, Gaussian
# from UQPyL.Surrogates.GP_Kernel import Matern, RBF, RationalQuadratic as RQ, DotProduct as Linear
from UQPyL.Utility.scalers import MinMaxScaler, StandardScaler
from UQPyL.Utility.metrics import r2_score, rank_score, sort_score
from UQPyL.Utility.polynomial_features import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import pickle
import os
os.chdir('./Test')
with open('./Samples.pickle','rb') as f:
    datas=pickle.load(f)
with open('./Samples_test.pickle','rb') as f:
    test_datas=pickle.load(f)
    
funcs=["Sphere", "Schwefel_2_22", "Schwefel_1_22", "Schwefel_2_21", "Rosenbrock", 
                         "Step", "Quartic", "Schwefel_2_26", "Rastrigin", "Ackley", "Griewank", 
                         "Trid", "Bent_Cigar", "Discus", "Weierstrass"]
funcs=["Rosenbrock"]
Sampling={}
n_dims=[5, 15, 30]
n_samples=[50, 150, 500]
n_times=20

Standard=StandardScaler(100,10)
MinMax=MinMaxScaler(0,100)

def compute_Y(func, dim, samples):
    F=eval(func)(dim=dim)
    Y=F.evaluate(samples)
    X=samples*(F.ub-F.lb)+F.lb
    return X,Y

r2_result={}
rank_result={}
kernels=[Cubic(), Multiquadric(), Linear(), Thin_plate_spline(), Gaussian()]
# Types=['CB', 'MQ', 'LN', 'TPS', 'GAS']  #RBF
# Types=['Origin','Ridge','Lasso']
# Types=['rbf', 'polynomial']
# Types=['gaussian','exp']
# Types=['rbf', 'linear', 'poly']
# Types=['linear']
# Samplings={5:150, 15:300, 30:500}
# Types=["RBF"]
# nus=[0.5,1.5,2.5,np.inf]
Types=['GAS']
for i, type in enumerate(Types):
    for func in funcs[:]:
        for D in n_dims:
            name="{}_D{}_N{}".format(func, D, 10*D)
            name2="{}_D{}_N{}".format(func, D, 50)
            re1=[]
            re2=[]
            re3=[]
            print("++++===+++++++++++++++++++++++++++++++")
            print(D)
            for I in range(20):
                train_X=datas[name][I]
                train_X, train_Y=compute_Y(func, D, train_X)  
                test_X=test_datas[name2][I]
                test_X, test_Y=compute_Y(func, D, test_X)
                
                # poly=PolynomialFeatures(degree=2)
                # train_x_ploy=poly.transform(train_X)
                # theta0=np.random.random(D)*10000000
                # ub=np.ones(D)*1e9
                # lb=np.ones(D)*100000
                # kernel=eval(type)(length_scale=theta0, l_lb=lb, l_ub=ub)
                # model=PR(scalers=(MinMaxScaler(-100,100),MinMaxScaler(-100,100)),alpha=0.1,Type=type)
                #model=SVR(kernel=type)
                #model=SVR(poly_feature=PolynomialFeatures(degree=2),kernel='linear', C=10, epsilon=10)
                #model=SVR(kernel='linear')
                # model=RBF(scalers=(MinMaxScaler(0,100), MinMaxScaler(0,100)),kernel=type)
                #
                gamma=[1e-3,1e-2,1e-1,1,10,100,1000]
                print("------------------------")
                for g in gamma:
                    try:
                        # theta0=np.ones(D)*1e-3
                        # theta_ub=np.ones(D)*1e-1
                        # theta_lb=np.ones(D)*1e-9
                        # model=KRG(scalers=(MinMaxScaler(0,100),MinMaxScaler(0,100)), theta0=theta0, lb=theta_lb, ub=theta_ub, kernel=type, optimizer='GA', fitMode='predictError')
                        # model=GPR(scalers=(MinMaxScaler(0,1),MinMaxScaler(0,1)),optimizer='Boxmin', fitMode='likelihood', kernel=kernel)
                        # K=kernels[4]
                        # K.gamma=g
                        # model=RBF(scalers=(MinMaxScaler(-1,1), MinMaxScaler(-1,1)),kernel=K)
                        # if type=="Lasso":
                        #     model=PR(scalers=(MinMaxScaler(-100,100),MinMaxScaler(-100,100)), poly_feature=PolynomialFeatures(degree=2),degree=2,alpha=0.1,Type=type)
                        # else:
                        #     model=PR(alpha=0.1, degree=2, poly_feature=PolynomialFeatures(degree=2), Type=type)
                        # model=SVR(poly_feature=PolynomialFeatures(degree=2), kernel=type, C=10, epsilon=10)
                        # model=PR(scalers=(MinMaxScaler(-100,100),MinMaxScaler(-100,100)),alpha=g, degree=2, poly_feature=PolynomialFeatures(degree=2), Type='Ridge')
                        model=PR(scalers=(MinMaxScaler(-100,100),MinMaxScaler(-100,100)), poly_feature=PolynomialFeatures(degree=2),degree=2,alpha=g,Type='Lasso')
                        model.fit(train_X, train_Y)
                        
                        pre_Y=model.predict(test_X)
                        print(r2_score(test_Y, pre_Y), rank_score(test_Y, pre_Y))
                        # re1.append(r2_score(test_Y, pre_Y))
                        # re2.append(rank_score(test_Y, pre_Y))
                    except:
                        re1.append(-np.inf)
                        re2.append(-np.inf)
        #     r2_result[name+"_{}".format(type)]=re1
        #     rank_result[name+"_{}".format(type)]=re2
        # with open('./total_PR.pickle','wb') as f:
        #     pickle.dump(r2_result, f)
        #     pickle.dump(rank_result, f)
