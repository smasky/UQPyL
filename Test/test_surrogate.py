import sys
sys.path.append(".")
from scipy.io import loadmat
import os
os.chdir('./Test')
import numpy as np
from sklearn.neural_network import MLPRegressor
from UQPyL.Surrogates.RBF_Kernel import Cubic, Linear, Multiquadric, Gaussian
from UQPyL.Surrogates.GP_Kernel import RBF
from UQPyL.Surrogates import RBF as RBFN, GPR, LinearRegression, PolynomialRegression, SVR, FNN, KRG
from UQPyL.Utility import PolynomialFeatures
from UQPyL.Utility import r2_score, rank_score
from UQPyL.Utility.scalers import MinMaxScaler, StandardScaler
#Test Data
data=loadmat('../UQPyL/Surrogates/gp.mat')
X=data['pop']
Y=data['value']

# X=np.loadtxt('./1225_X.txt')
# Y=np.loadtxt('./1225_y.txt')

train_X=X[0:100, :]
train_Y=Y[0:100, 0:1]
test_X=X[280:, :]
test_Y=Y[280:, 0:1]

###########################MLP##############################
pf=PolynomialFeatures(degree=2, include_bias=False)
xx=pf.transform(train_X)
t_xx=pf.transform(test_X)
#########################FNN#########################
# fcnn=FNN(scalers=(MinMaxScaler(0,10),MinMaxScaler(0,10)), hidden_layer_sizes=[200,100],
#                         solver='adam', alpha=10, epoch=2000 ,activation_functions='relu')
# fcnn.fit(xx, train_Y)
# P_Y=fcnn.predict(t_xx)
# print("r2_score:", r2_score(test_Y, P_Y))
# print("rank_score", rank_score(test_Y, P_Y))

########################SVR###############################
svr=SVR(scalers=(MinMaxScaler(-1,1),MinMaxScaler(0,1)), kernel='linear', 
                C=1, epsilon=0.1, degree=3)
svr.fit(train_X, train_Y)
P_Y=svr.predict(test_X)
print("r2_score:", r2_score(test_Y, P_Y))
print("rank_score", rank_score(test_Y, P_Y))
temp=1
#########PolynomialRegression and LinerRegression#####
# PL=PolynomialRegression(scalers=(MinMaxScaler(0,10), MinMaxScaler(0,10)), fit_intercept=True, degree=2, interaction_only=False, Type='Lasso')
# PL.fit(train_X, train_Y)
# P_Y=PL.predict(test_X)
# print("r2_score:", r2_score(test_Y, P_Y))
# print("rank_score", rank_score(test_Y, P_Y))

# LLR=LinearRegression(scalers=(MinMaxScaler(0,10), MinMaxScaler(0,10)),Type='Lasso')
# LLR.fit(train_X, train_Y)
# P_Y=LLR.predict(test_X)
# print("r2_score:", r2_score(test_Y, P_Y))
# print("rank_score", rank_score(test_Y, P_Y))
# ################ Gaussian Process###################
dims=20
theta=np.random.random(dims)*1000
ub=np.ones(dims)*1e6
lb=np.ones(dims)*1e3
kernel=RBF(theta, ub, lb)
theta=kernel.theta
ub=kernel.theta_ub
lb=kernel.theta_lb
gp=GPR(scalers=(MinMaxScaler(0,10), MinMaxScaler(0,10)), optimizer='Boxmin', fitMode='likelihood', kernel=kernel)
gp.fit(train_X, train_Y)
P_Y=gp.predict(test_X)
print("r2_score:", r2_score(test_Y, P_Y))
print("rank_score", rank_score(test_Y, P_Y))
#####################Kriging########################
dims=20
theta=np.random.random(dims)*100
ub=np.ones(dims)*1e3
lb=np.ones(dims)*1
dace_obj1 = KRG(theta, ub, lb, scalers=(MinMaxScaler(0,10), MinMaxScaler(0,10)), regression='poly1', kernel='gaussian', optimizer='Boxmin', fitMode='likelihood')
dace_obj1.fit(train_X,train_Y)
P_Y,ss=dace_obj1.predict(test_X)
print("r2_score:", r2_score(test_Y, P_Y))
print("rank_score", rank_score(test_Y, P_Y))
#######################RBF##########################
rbf=RBFN(scalers=(MinMaxScaler(0,10), MinMaxScaler(0,10)), kernel=Linear())
rbf.fit(train_X,train_Y)
P_Y=rbf.predict(test_X)
print("r2_score:", r2_score(test_Y, P_Y))
print("rank_score", rank_score(test_Y, P_Y))
